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

voting_table = torch.zeros(NUM_OUTPUTS,2)

def calc_acc(net,args,device,test_loader,output = False, last = False):
    total = 0
    correct_spike = 0
    correct_mem = 0
    zero_activity_hidden = 0
    full_activity_hidden = 0
    total_activity_hidden = 0
    zero_activity_out = 0
    full_activity_out = 0
    total_activity_out = 0
    with torch.no_grad():
      net.eval()
      for batch_idx, (data, targets) in enumerate(test_loader):
        spiking_data = spikegen.rate(data.to(device), num_steps=args.num_steps)
        targets = targets.to(device)

        #forward pass
        test_spk_all, test_mem, _, _ = net(spiking_data[:,:,0,:,:].view(args.num_steps,args.batch_size,NUM_INPUTS),args)
        test_spk = test_spk_all[len(test_spk_all)-1]
        test_spk_hidden = test_spk_all[len(test_spk_all)-2].swapaxes(0,1)
        test_spk_out = test_spk_all[len(test_spk_all)-1].swapaxes(0,1)
        #calculate total accuracy
        _, predicted_spike = test_spk.sum(dim=0).max(1)
        _, predicted_mem = test_mem.sum(dim=0).max(1)

        total += targets.size(0)

        if args.use_stdp and last:
            voting_table.to(device)
            spk_guesses = test_spk_out.sum(1)
            for i in range(spk_guesses.size(0)):
                out_guesses = torch.zeros(10).to(device)
                for j in range(spk_guesses[i].size(0)):
                    out_guesses[voting_table[j,0].long()] += voting_table[j,1] * spk_guesses[i][j]
                correct_spike += out_guesses.argmax() == targets[i]
                correct_mem = 0

            print("--Current Accuracy (Batch {})--".format(batch_idx))
            print("{:.2f}".format(correct_spike*100/total))
        else:
            correct_spike += (predicted_spike == targets).sum().item()
            correct_mem += (predicted_mem == targets).sum().item()

        zero_activity_hidden += (test_spk_hidden.sum(1).sum(1)==torch.zeros(args.batch_size,device=device)).sum().item()
        total_activity_hidden += test_spk_hidden.sum().item()
        full_activity_hidden += ((test_spk_hidden.sum(1).sum(1) / args.num_steps) == torch.ones(args.batch_size,device=device) * NUM_HIDDEN/2).sum().item()
        zero_activity_out += (test_spk_out.sum(1).sum(1)==torch.zeros(args.batch_size,device=device)).sum().item()
        total_activity_out += test_spk_out.sum().item()
        full_activity_out += ((test_spk_out.sum(1).sum(1) / args.num_steps) == torch.ones(args.batch_size,device=device) * NUM_OUTPUTS/2).sum().item()

        if args.use_stdp and batch_idx % args.log_interval == 0 and last:
            print("Batch {} Done!".format(batch_idx))

    zero_activity_hidden = (zero_activity_hidden/total)*100
    full_activity_hidden = (full_activity_hidden/total)*100
    total_activity_hidden = ((total_activity_hidden/(total*args.num_steps))*2/NUM_HIDDEN)*100
    zero_activity_out = (zero_activity_out/total)*100
    full_activity_out = (full_activity_out/total)*100
    total_activity_out = ((total_activity_out/(total*args.num_steps))*2/NUM_OUTPUTS)*100
    garabge_activity = [zero_activity_hidden + full_activity_hidden,zero_activity_out+full_activity_out]
    average_activity = [total_activity_hidden,total_activity_out]
    track_firing_rates(zero_activity_hidden,full_activity_hidden,total_activity_hidden)
    track_correct_labels(targets)
    if(output):
        output_layer_stats(net,
                            [zero_activity_hidden,zero_activity_out],
                            [full_activity_hidden,full_activity_out],
                            [total_activity_hidden,total_activity_out]
                            )
        print_epoch(correct_spike,total)
    return (correct_spike / total * 100),(correct_mem / total * 100),garabge_activity,average_activity


def train_backprop(net, args, device, train_loader, test_loader, optimizer, loss, epoch):
    train_batch = iter(train_loader)

    #Minibatch training loop
    for batch_idx, (data, targets) in enumerate(train_batch):
        spiking_data = spikegen.rate(data.to(device), num_steps=args.num_steps)
        targets = targets.to(device)

        #forward pass
        net.train()
        spk_rec, mem_rec, _, _= net(spiking_data.view(args.num_steps,args.batch_size, -1),args)
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
            test_spk, test_mem, _, _ = net(spiking_test_data.view(args.num_steps,args.batch_size, -1),args)
            test_spk = test_spk[len(test_spk)-1]

            #Test set loss
            test_loss = torch.zeros((1), dtype=dtype, device=device)
            for step in range(args.num_steps):
                test_loss += loss(test_mem[step], test_targets) / args.num_steps
            track_test_loss(test_loss.item())
            #Print train/test loss/accuracy
            if batch_idx  % args.log_interval == 0:
                accuracy_spike, accuracy_mem, _, _ = calc_acc(net,args,device,test_loader)
                train_printer(epoch,batch_idx,accuracy_spike,accuracy_mem)#,train_loss = train_loss_hist(len(train_loader)*epoch + batch_idx), test_loss = test_loss_hist(len(train_loader)*epoch + batch_idx))
                track_accuracy(accuracy_spike,accuracy_mem)

def train_stdp(net, args, device, train_loader, test_loader, epoch, out = True, layer = 0):
    train_batch = iter(train_loader)

    #Minibatch training loop
    accuracy_spike, accuracy_mem = 9,9
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(train_batch):
            spiking_data = spikegen.rate(data.to(device), num_steps=args.num_steps)

            #Pass Batch through SNN
            spk_rec, mem_rec, pre_rec, post_rec = net(spiking_data.view(args.num_steps,args.batch_size, -1),args,stdp = True, layer = layer)
            if out:
                printProgressBar(batch_idx, len(train_batch), "Spk: {:.2f}%, Mem: {:.2f}%".format(accuracy_spike,accuracy_mem), "Complete", length=50)

            if batch_idx  % args.log_interval == 0:
                accuracy_spike, accuracy_mem, _, _ = 9,9,0,0#calc_acc(net,args,device,test_loader)
                track_accuracy(accuracy_spike,accuracy_mem)
    print()
    print("Epoch {} Done!".format(epoch))

def calculate_voting(net,args,train_loader,device):
    global voting_table
    voting_table.to(device)
    neuron_votes = torch.zeros(10,NUM_OUTPUTS).to(device)
    train_batch = iter(train_loader)
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(train_batch):
            spiking_data = spikegen.rate(data.to(device), num_steps=args.num_steps)

            #Pass Batch through SNN
            spk_rec, _, _, _ = net(spiking_data.view(args.num_steps,args.batch_size, -1),args)
            spk_out=spk_rec[len(spk_rec)-1]
            firings = spk_out.sum(0)
            for i,batch in enumerate(firings):
                if batch.sum(0) != 0:
                    neuron_votes[targets[i]] += (batch / batch.max())

            if batch_idx  % args.log_interval == 0:
                print("Batch {} Done!".format(batch_idx))

        neuron_votes_tmp = torch.zeros_like(neuron_votes)
        for prediction in range(neuron_votes.size(0)):
            for neuron in range(neuron_votes[prediction].size(0)):
                neuron_votes_tmp[prediction,neuron] = neuron_votes[prediction,neuron] / neuron_votes.swapaxes(0,1)[neuron].sum()

    neuron_votes = neuron_votes_tmp

    for neuron in range(voting_table.size(0)):
        voting_table[neuron][0] = neuron_votes.swapaxes(0,1)[neuron].argmax()
        voting_table[neuron][1] = neuron_votes.swapaxes(0,1)[neuron].max()

    sums = torch.zeros(10).to(device)
    for i in range(voting_table.size(0)):
        sums[voting_table[i,0].long()] += voting_table[i,1]

    for i in range(voting_table.size(0)):
        voting_table[i,1] = voting_table[i,1] / sums[voting_table[i,0].long()]

    #torch.set_printoptions(profile = "full")
    #print(voting_table)
