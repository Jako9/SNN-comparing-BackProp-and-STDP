#SNN
from snntorch import spikegen

#PyTorch
import torch
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
    with torch.no_grad():
      net.eval()
      for data, targets in test_loader:
        spiking_data = spikegen.rate(data.to(device), num_steps=args.num_steps)
        targets = targets.to(device)

        #forward pass
        test_spk, test_mem, _, _ = net(spiking_data[:,:,0,:,:].view(args.num_steps,args.batch_size,NUM_INPUTS),args)
        test_spk = test_spk[len(test_spk)-1]
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

    if(output):
        print("AVG Layer 1")
        print(net.fc1.weight.mean())
        print("AVG Layer 2")
        print(net.fc2.weight.mean())
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
                accuracy_spike, accuracy_mem = calc_acc(net,args,device,test_loader)
                train_printer(epoch,batch_idx,accuracy_spike,accuracy_mem,train_loss = train_loss_hist(len(train_loader)*epoch + batch_idx), test_loss = test_loss_hist(len(train_loader)*epoch + batch_idx))
                track_accuracy(accuracy_spike,accuracy_mem)
                if args.dry_run:
                    break

def train_stdp(net, args, device, train_loader, test_loader, optimizer, loss, epoch, out = True, layer = 0):
    train_batch = iter(train_loader)

    #Minibatch training loop
    accuracy_spike, accuracy_mem = 0,0
    for batch_idx, (data, targets) in enumerate(train_batch):
        spiking_data = spikegen.rate(data.to(device), num_steps=args.num_steps)

        #Pass Batch through SNN
        spk_rec, mem_rec, pre_rec, post_rec = net(spiking_data.view(args.num_steps,args.batch_size, -1),args,stdp = True, layer = layer)
        #spike_indices = []
        #for layer in spk_rec:
        #    indices = layer.nonzero()
        #    spike_indices.append(indices)

        #layers_weights = []
        #for layer in net.children():
        #    if isinstance(layer, torch.nn.Linear):
        #        layers_weights.append(layer.weight.to("cpu").detach().numpy())
        #
        #for layer_index in range(1,len(spk_rec)):
        #    adjustments = calc_spikes(layers_weights[layer_index-1],layers_weights,spk_rec[layer_index].to("cpu").detach().numpy(),spk_rec[layer_index-1].to("cpu").detach().numpy(),args.num_steps,args.batch_size,STDP_RANGE)
        #    with torch.no_grad():
        #        linear_layer_index = 0
        #        for layer in net.children():
        #            if isinstance(layer, torch.nn.Linear):
        #                if (linear_layer_index + 1 == layer_index):
        #                    weights = layer.weight.to("cpu").detach().numpy()
        #                    updated_weights = torch.nn.parameter.Parameter(torch.tensor(np.add(weights,adjustments,dtype=np.float32)).to(device))
        #                    layer.weight = updated_weights
        #                    break
        #                linear_layer_index += 1

        if out:
            printProgressBar(batch_idx, len(train_batch), "Spk: {:.2f}%, Mem: {:.2f}%".format(accuracy_spike,accuracy_mem), "Complete", length=50)

        if batch_idx  % args.log_interval == 0:
            #accuracy_spike, accuracy_mem = calc_acc(net,args,device,test_loader)
            track_accuracy(0,0)
            if args.dry_run:
                break
    print()
    print("Epoch {} Done!".format(epoch))

@njit()
def calc_spikes(weights,layers_weights,layer,prev_layer,num_steps,batch_size,stdp_range):
    adjustments = np.zeros(weights.shape)
    for time_step in range(num_steps):
        for batch_index in range(batch_size):
            if 1 in layer[time_step][batch_index]:
                presynaptic_spikes = calc_presynaptic_spikes(prev_layer,time_step,batch_index,stdp_range-1)
                postsynaptic_spikes = calc_postsynaptic_spikes(prev_layer,time_step+1,batch_index,stdp_range-1)
                for post_neuron_index in range(layer[time_step][batch_index].size):
                    if(layer[time_step][batch_index][post_neuron_index] == 1):
                        for pre_neuron_index in range(prev_layer[time_step][batch_index].size):
                            adjustments[post_neuron_index][pre_neuron_index] += (calc_weight_adjustment(weights,post_neuron_index,pre_neuron_index,presynaptic_spikes[pre_neuron_index],postsynaptic_spikes[pre_neuron_index])/ batch_size * num_steps)

    return adjustments

@njit()
def calc_weight_adjustment(weights,post_neuron,pre_neuron,pre_count,post_count):
    adjustedment = STDP_LR * (math.exp((pre_count - post_count) / STDP_RANGE) - STDP_OFFSET) * (MAX_WEIGHT - weights[post_neuron][pre_neuron]) * (weights[post_neuron][pre_neuron] - MIN_WEIGHT)
    #print("Pre, Post: {}, {}".format(pre_count,post_count))
    #print("Exp: {}".format((math.exp((pre_count - post_count) / STDP_RANGE) - STDP_OFFSET)))
    #print("Weight: {}".format(weights[post_neuron][pre_neuron]))
    #print("Adjustment: {}".format(adjustedments))
    return adjustedment

@njit()
def calc_presynaptic_spikes(layer,time_step,batch_index,stdp_range):
    if time_step < 0:
        return np.zeros(layer[0][0].size).astype(np.float32)
    if stdp_range <= 0:
        return layer[time_step][batch_index]
    return np.add(layer[time_step][batch_index],calc_presynaptic_spikes(layer,time_step-1,batch_index,stdp_range-1))

@njit()
def calc_postsynaptic_spikes(layer,time_step,batch_index,stdp_range):
    if time_step >= layer.shape[0]:
        return np.zeros(layer[0][0].size).astype(np.float32)
    if stdp_range <= 0:
        return layer[time_step][batch_index]
    return np.add(layer[time_step][batch_index],calc_postsynaptic_spikes(layer,time_step+1,batch_index,stdp_range-1))

def calc_pre_weight_change(args, spk, pre, weights):
    adjustments = (torch.bmm(pre.unsqueeze(2), spk.unsqueeze(1)).sum(dim=0)) / args.batch_size
    adjusted_weights = (weights + (weights * adjustments)).clamp(MIN_WEIGHT,MAX_WEIGHT)
    return adjusted_weights

def calc_post_weight_change(args, spk, post, weights):
    adjustments = (torch.bmm(spk.unsqueeze(2), post.unsqueeze(1)).sum(dim=0)) / args.batch_size
    adjusted_weights = (weights + (weights * adjustments)).clamp(MIN_WEIGHT,MAX_WEIGHT)
    return adjusted_weights
