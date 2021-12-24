#SNN
from snntorch import spikegen

#PyTorch
import torch
import random
import time
import math
from config import *
from log import *
import numba
from numba import njit
from numba.typed import List

translation_table = torch.tensor([[0,1,2,3,4,5,6,7,8,9],[20,20,20,20,20,20,20,20,20,20]]).to("cuda:0")

def calc_acc(net,args,device,test_loader,output = False):

    reset_class_guesses()
    total = 0
    correct_spike = 0
    correct_mem = 0
    zero_activity = 0
    total_activity = 0
    with torch.no_grad():
      net.eval()
      for data, targets in test_loader:
        spiking_data = spikegen.rate(data.to(device), num_steps=args.num_steps)
        targets = targets.to(device)

        #forward pass
        test_spk_all, test_mem, _, _ = net(spiking_data[:,:,0,:,:].view(args.num_steps,args.batch_size,NUM_INPUTS),args, prediction = targets)
        test_spk = test_spk_all[len(test_spk_all)-1]
        test_spk_hidden = test_spk_all[len(test_spk_all)-2].swapaxes(0,1)
        #calculate total accuracy
        _, predicted_spike = test_spk.sum(dim=0).max(1)
        _, predicted_mem = test_mem.sum(dim=0).max(1)

        if args.use_stdp:
            global translation_table
            for i in range(targets.size(0)):
                if translation_table[0][predicted_spike[i]] != targets[i] and translation_table[1][predicted_spike[i]] < 500:
                    translation_table[1][predicted_spike[i]] -= 1
                elif translation_table[1][predicted_spike[i]] < 500:
                    translation_table[1][predicted_spike[i]] += 1

                if translation_table[1][predicted_spike[i]] <= 0:
                    target_index = (translation_table[0] == targets[i]).nonzero().item()
                    if translation_table[1][target_index] >= 500:
                        continue
                    elif translation_table[1][target_index] > 50:
                        translation_table[1][target_index].div(2)
                    else:
                        translation_table[0][target_index] = translation_table[0][predicted_spike[i]]
                        translation_table[0][predicted_spike[i]] = targets[i]
                        translation_table[1][target_index] = 100
                        translation_table[1][predicted_spike[i]] = 100


                track_class_guesses(targets[i],predicted_spike[i])
                predicted_spike[i] = translation_table[0][predicted_spike[i]]
                predicted_mem[i] = translation_table[0][predicted_mem[i]]

        total += targets.size(0)
        correct_spike += (predicted_spike == targets).sum().item()
        correct_mem += (predicted_mem == targets).sum().item()
        zero_activity += (test_spk_hidden.sum(1).sum(1)==torch.zeros(args.batch_size,device=device)).sum().item()
        total_activity += test_spk_hidden.sum().item()

    if(output):
        print("\n")
        print("--Layer 1--")
        print("Min: {:.4}".format(net.fc1.weight.min().item()))
        print("Max: {:.4}".format(net.fc1.weight.max().item()))
        print("Average: {:.4}".format(net.fc1.weight.mean().item()))
        print("STD: {:.4}".format(net.fc1.weight.std().item()))
        print("--Layer 2--")
        print("Min: {:.4}".format(net.fc2.weight.min().item()))
        print("Max: {:.4}".format(net.fc2.weight.max().item()))
        print("Average: {:.4}".format(net.fc2.weight.mean().item()))
        print("\n")
        print("STD: {:.4}".format(net.fc2.weight.std().item()))
        print("\n")
        print("Zero-Activity percentage: {:.2f}%".format((zero_activity/total)*100))
        print("Average-Activity: {:.2f}".format(total_activity/(total*args.num_steps)))
        print("\n")
        track_correct_labels(targets)
        print_epoch(correct_spike,total,translation_table)

    return (correct_spike / total * 100),(correct_mem / total * 100)


def train_backprop(net, args, device, train_loader, test_loader, optimizer, loss, epoch):
    train_batch = iter(train_loader)

    #Minibatch training loop
    for batch_idx, (data, targets) in enumerate(train_batch):
        spiking_data = spikegen.rate(data.to(device), num_steps=args.num_steps)
        targets = targets.to(device)

        #forward pass
        net.train()
        spk_rec, mem_rec, _, _= net(spiking_data.view(args.num_steps,args.batch_size, -1),args, prediction = targets)
        spk_rec = spk_rec[len(spk_rec)-1]

        #initialize the loss & sum over time
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(args.num_steps):
            loss_val += loss(mem_rec[step], targets) / args.num_steps

        #Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        #Store loss history for future plotting
        track_train_loss(loss_val.item())

        #Test set
        with torch.no_grad():
            net.eval()
            test_data, test_targets = next(iter(test_loader))
            test_data = test_data.to(device)
            spiking_test_data = spikegen.rate(test_data, num_steps=args.num_steps)
            test_targets = test_targets.to(device)

            #Test set forward pass
            test_spk, test_mem, _, _ = net(spiking_test_data.view(args.num_steps,args.batch_size, -1),args, prediction = targets)
            test_spk = test_spk[len(test_spk)-1]

            #Test set loss
            test_loss = torch.zeros((1), dtype=dtype, device=device)
            for step in range(args.num_steps):
                test_loss += loss(test_mem[step], test_targets) / args.num_steps
            track_test_loss(test_loss.item())
            #Print train/test loss/accuracy
            if batch_idx  % args.log_interval == 0:
                accuracy_spike, accuracy_mem = calc_acc(net,args,device,test_loader)
                train_printer(epoch,batch_idx,accuracy_spike,accuracy_mem)#,train_loss = train_loss_hist(len(train_loader)*epoch + batch_idx), test_loss = test_loss_hist(len(train_loader)*epoch + batch_idx))
                track_accuracy(accuracy_spike,accuracy_mem)

def train_stdp(net, args, device, train_loader, epoch, out = True, layer = 0):
    train_batch = iter(train_loader)

    #Minibatch training loop
    accuracy_spike, accuracy_mem = 0,0
    for batch_idx, (data, targets) in enumerate(train_batch):
        spiking_data = spikegen.rate(data.to(device), num_steps=args.num_steps)

        #Pass Batch through SNN
        spk_rec, mem_rec, pre_rec, post_rec = net(spiking_data.view(args.num_steps,args.batch_size, -1),args,stdp = True, layer = layer,prediction = targets)
        if out:
            printProgressBar(batch_idx, len(train_batch), "Spk: {:.2f}%, Mem: {:.2f}%".format(accuracy_spike,accuracy_mem), "Complete", length=50)

        if batch_idx  % args.log_interval == 0:
            #accuracy_spike, accuracy_mem = calc_acc(net,args,device,test_loader)
            track_accuracy(0,0)
    print()
    print("Epoch {} Done!".format(epoch))

def calc_pre_weight_change(args, spk, pre, weights):
    adjustments = torch.einsum("bi,bj->ij", pre, spk) / args.batch_size
    adjusted_weights = (weights + (weights * adjustments)).clamp(MIN_WEIGHT,MAX_WEIGHT)
    return adjusted_weights

def calc_post_weight_change(args, spk, post, weights):
    adjustments = torch.einsum("bi,bj->ij", spk, post) / args.batch_size
    adjusted_weights = (weights + (weights * adjustments)).clamp(MIN_WEIGHT,MAX_WEIGHT)
    return adjusted_weights

def calc_weight_change_last_layer(args, spk_out, spk_hidden, spk_out_rec, spk_hidden_rec, weights, predicted_spikes):
    adjustments_pos = torch.bmm(spk_out.unsqueeze(2), spk_hidden_rec.unsqueeze(1)) #128,10,256
    adjustments_neg = torch.bmm(spk_out_rec.unsqueeze(2), 1-spk_hidden.unsqueeze(1)) / 9 #128,10,256
    adjustments = torch.zeros_like(adjustments_pos)
    adjustments[torch.arange(adjustments.size(0)), predicted_spikes] = adjustments_pos[torch.arange(adjustments.size(0)), predicted_spikes] + adjustments_neg[torch.arange(adjustments.size(0)), predicted_spikes]#128,256
    adjustments = adjustments.sum(dim=0) / args.batch_size
    adjusted_weights = (weights + (weights * adjustments)).clamp(MIN_WEIGHT,MAX_WEIGHT)
    return adjusted_weights
