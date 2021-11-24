#SNN
from snntorch import spikegen

#PyTorch
import torch

from config import *
from log import *
import numba
from numba import njit
from numba.typed import List

def calc_acc(net,args,device,test_loader,output = False):
    total = 0
    correct_spike = 0
    correct_mem = 0
    with torch.no_grad():
      net.eval()
      for data, targets in test_loader:
        spiking_data = spikegen.rate(data.to(device), num_steps=args.num_steps)
        targets = targets.to(device)

        #forward pass
        test_spk, test_mem = net(spiking_data[:,:,0,:,:].view(args.num_steps,args.batch_size,NUM_INPUTS))
        test_spk = test_spk[len(test_spk)-1]
        #calculate total accuracy
        _, predicted_spike = test_spk.sum(dim=0).max(1)
        _, predicted_mem = test_mem.sum(dim=0).max(1)
        total += targets.size(0)
        correct_spike += (predicted_spike == targets).sum().item()
        correct_mem += (predicted_mem == targets).sum().item()

    if(output):
        track_correct_labels(targets)
        print_epoch(correct_spike,total)

    return (correct_spike / total * 100),(correct_mem / total * 100)

def train_backprop(net, args, device, train_loader, test_loader, optimizer, loss, epoch):
    train_batch = iter(train_loader)

    #Minibatch training loop
    for batch_idx, (data, targets) in enumerate(train_batch):
        spiking_data = spikegen.rate(data.to(device), num_steps=args.num_steps)
        targets = targets.to(device)

        #forward pass
        net.train()
        spk_rec, mem_rec = net(spiking_data.view(args.num_steps,args.batch_size, -1))
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
            test_spk, test_mem = net(spiking_test_data.view(args.num_steps,args.batch_size, -1))
            test_spk = test_spk[len(test_spk)-1]

            #Test set loss
            test_loss = torch.zeros((1), dtype=dtype, device=device)
            for step in range(args.num_steps):
                test_loss += loss(test_mem[step], test_targets) / args.num_steps
            track_test_loss(test_loss.item())
            #Print train/test loss/accuracy
            if batch_idx  % args.log_interval == 0:
                accuracy_spike, accuracy_mem = calc_acc(net,args,device,test_loader)
                train_printer(epoch,batch_idx,train_loss_hist(len(train_loader)*epoch + batch_idx),test_loss_hist(len(train_loader)*epoch + batch_idx),accuracy_spike,accuracy_mem)
                track_accuracy(accuracy_spike,accuracy_mem)
                if args.dry_run:
                    break

def train_stdp(net, args, device, train_loader, test_loader, optimizer, loss, epoch):
    train_batch = iter(train_loader)

    #Minibatch training loop
    for batch_idx, (data, targets) in enumerate(train_batch):
        spiking_data = spikegen.rate(data.to(device), num_steps=args.num_steps)

        #Pass Batch through SNN
        spk_rec, mem_rec = net(spiking_data.view(args.num_steps,args.batch_size, -1))
        spike_indices = []
        for layer in spk_rec:
            indices = layer.nonzero()
            spike_indices.append(indices)

        for layer_index in range(1,len(spk_rec)-1):
            calc_spikes(spk_rec[layer_index].to("cpu").detach().numpy(),spk_rec[layer_index-1].to("cpu").detach().numpy(),args.num_steps,args.batch_size,5)

        if batch_idx  % args.log_interval == 0:
            print("Batch Done! ({})".format(batch_idx))
            if args.dry_run:
                break

    print("Epoch Done!")

@njit()
def calc_spikes(layer,prev_layer,num_steps,batch_size,stdp_range):
    for time_step in range(num_steps-1):
        for batch_index in range(batch_size-1):
            if 1 in layer[time_step][batch_index]:
                spiked_neurons = layer[time_step][batch_index].nonzero()[0]
                presynaptic_spikes = calc_presynaptic_spikes(prev_layer,time_step,batch_index,stdp_range-1)
                postsynaptic_spikes = calc_postsynaptic_spikes(prev_layer,time_step+1,batch_index,stdp_range-1)
    #TODO

@njit()
def calc_presynaptic_spikes(layer,time_step,batch_index,stdp_range):
    #Das ist ein Witz, dass das funktioniert (Sonst mecker njit wegen Typ)
    if time_step < 0:
        return [np.reshape(np.array(x),0) for x in range(0)]
    if stdp_range <= 0:
        return [layer[time_step][batch_index].nonzero()[0]]

    return [layer[time_step][batch_index].nonzero()[0]] + (calc_presynaptic_spikes(layer,time_step-1,batch_index,stdp_range-1))

@njit()
def calc_postsynaptic_spikes(layer,time_step,batch_index,stdp_range):
    if time_step >= layer.shape[0]:
        return [np.reshape(np.array(x),0) for x in range(0)]
    if stdp_range <= 0:
        return [layer[time_step][batch_index].nonzero()[0]]
    return [layer[time_step][batch_index].nonzero()[0]] + (calc_postsynaptic_spikes(layer,time_step+1,batch_index,stdp_range-1))
