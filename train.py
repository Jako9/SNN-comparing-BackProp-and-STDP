#SNN
from snntorch import spikegen

#PyTorch
import torch

from config import *
from log import *

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
    iter_counter = 0
    train_batch = iter(train_loader)

    #Minibatch training loop
    for data, targets in train_batch:
        spiking_data = spikegen.rate(data.to(device), num_steps=args.num_steps)
        targets = targets.to(device)

        #forward pass
        net.train()
        spk_rec, mem_rec = net(spiking_data.view(args.num_steps,args.batch_size, -1))

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

            #Test set loss
            test_loss = torch.zeros((1), dtype=dtype, device=device)
            for step in range(args.num_steps):
                test_loss += loss(test_mem[step], test_targets) / args.num_steps
            track_test_loss(test_loss.item())
            #Print train/test loss/accuracy
            if iter_counter  % args.log_interval == 0:
                accuracy_spike, accuracy_mem = calc_acc(net,args,device,test_loader)
                train_printer(epoch,iter_counter,train_loss_hist(len(train_loader)*epoch + iter_counter),test_loss_hist(len(train_loader)*epoch + iter_counter),accuracy_spike,accuracy_mem)
                track_accuracy(accuracy_spike,accuracy_mem)
                if args.dry_run:
                    break
            iter_counter +=1

def train_stdp(net, args, device, train_loader, test_loader, optimizer, loss, epoch):
    print("Hi :')")
    #TODO
