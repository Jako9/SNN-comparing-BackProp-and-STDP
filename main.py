#SNN
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import backprop
import math

#PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd._functions import Resize
from torch.nn import functional as F

#Plotting
import matplotlib.pyplot as plt
import numpy as np
import itertools
from IPython.display import HTML
from scipy.interpolate import interp1d

import argparse

#Corresponding Items to nn Classes
classes = ["T-Shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle Boot"]

data_path='/data/mnist'
# Network Architecture
num_inputs = 28*28
num_hidden = 1000
num_outputs = 10

# Temporal Dynamics
num_steps = 50

dtype = torch.float

#for plotting
spk0_hist = []
spk1_hist = []
spk2_hist = []
mem0_hist = []
mem1_hist = []
mem2_hist = []
thrs0_hist = torch.zeros(num_inputs)
thrs1_hist = torch.zeros(num_hidden)
thrs2_hist = torch.zeros(num_outputs)
loss_hist = []
test_loss_hist = []
accuracies_spike = []
accuracies_mem = []
counter = 0
correct_labels = torch.zeros(1)
fig, ax = plt.subplots(3,3,figsize=(20,8))

# Define Network
class Net(nn.Module):
    def __init__(self,beta,device,learn_beta,learn_threshold):
        super().__init__()

        if(learn_threshold):
            threshold0 = nn.Parameter(torch.rand(num_inputs,requires_grad=True).to(device))
            threshold1 = nn.Parameter(torch.rand(num_hidden,requires_grad=True).to(device))
            self.register_parameter(name = "threshold0", param = threshold0)
            self.register_parameter(name = "threshold1", param = threshold1)
        else:
            threshold0 = torch.ones(num_inputs).to(device)
            threshold1 = torch.ones(num_hidden).to(device)

        threshold2 = torch.ones(num_outputs).to(device)

        # Initialize layers
        self.lif0 = snn.Leaky(beta=beta,learn_beta=learn_beta,threshold = threshold0, reset_mechanism="zero")
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta,learn_beta=learn_beta,threshold = threshold1, reset_mechanism="zero")
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta,learn_beta=False,threshold = threshold2, reset_mechanism="zero")

    def forward(self, x):
        # Initialize hidden states at t=0
        mem0 = self.lif0.init_leaky()
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        #Clear log-history
        spk0_hist.clear()
        spk1_hist.clear()
        spk2_hist.clear()
        mem0_hist.clear()
        mem1_hist.clear()
        mem2_hist.clear()
        global thrs0_hist
        global thrs1_hist
        global thrs2_hist
        thrs0_hist = self.lif0.threshold
        thrs1_hist = self.lif1.threshold
        thrs2_hist = self.lif2.threshold

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            spk0, mem0 = self.lif0(x[step],mem0)
            cur1 = F.relu(self.fc1(spk0))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = F.relu(self.fc2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            #Track Spike-Activity and Membrane Potential
            mem0_hist.append(mem0)
            spk0_hist.append(spk0)
            mem1_hist.append(mem1)
            spk1_hist.append(spk1)
            mem2_hist.append(mem2)
            spk2_hist.append(spk2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


def train_printer(epoch,iter_counter,loss,test_loss,accuracy_spike,accuracy_mem):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss:.2f}")
    print(f"Test Set Loss: {test_loss:.2f}")
    print(f"Test Set Accuracy with Spikes: {accuracy_spike:.2f}%")
    print(f"Test Set Accuracy with Membrane Potential: {accuracy_mem:.2f}%")
    print("\n")

def train_backprop(args, net, device, train_loader, test_loader, optimizer, loss, epoch):
    iter_counter = 0
    train_batch = iter(train_loader)

    # Minibatch training loop
    for data, targets in train_batch:
        spiking_data = spikegen.rate(data.to(device), num_steps=num_steps)
        targets = targets.to(device)

        # forward pass
        net.train()
        spk_rec, mem_rec = net(spiking_data.view(num_steps,args.batch_size, -1))

        # initialize the loss & sum over time
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val += loss(mem_rec[step], targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        with torch.no_grad():
            net.eval()
            test_data, test_targets = next(iter(test_loader))
            test_data = test_data.to(device)
            spiking_test_data = spikegen.rate(test_data, num_steps=num_steps)
            test_targets = test_targets.to(device)

            # Test set forward pass
            test_spk, test_mem = net(spiking_test_data.view(num_steps,args.batch_size, -1))

            # Test set loss
            test_loss = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                test_loss += loss(test_mem[step], test_targets)
            test_loss_hist.append(test_loss.item())
            # Print train/test loss/accuracy
            global counter
            if counter % args.log_interval == 0:
                accuracy_spike, accuracy_mem = calc_acc(net,device,test_loader,args.batch_size)
                train_printer(epoch,iter_counter,loss_hist[counter],test_loss_hist[counter],accuracy_spike,accuracy_mem)
                accuracies_spike.append(accuracy_spike)
                accuracies_mem.append(accuracy_mem)
                if args.dry_run:
                    break
            counter += 1
            iter_counter +=1

def calc_acc(net,device,test_loader,batch_size):
    total = 0
    correct_spike = 0
    correct_mem = 0
    with torch.no_grad():
      net.eval()
      for data, targets in test_loader:
        data = data.to(device)
        spiking_data = spikegen.rate(data, num_steps=num_steps)
        targets = targets.to(device)

        # forward pass
        test_spk, test_mem = net(spiking_data[:,:,0,:,:].view(num_steps,batch_size,num_inputs))
        # calculate total accuracy
        _, predicted_spike = test_spk.sum(dim=0).max(1)
        _, predicted_mem = test_mem.sum(dim=0).max(1)
        total += targets.size(0)
        correct_spike += (predicted_spike == targets).sum().item()
        correct_mem += (predicted_mem == targets).sum().item()

    return (correct_spike / total * 100),(correct_mem / total * 100)


def test(net,device,test_loader):
    total = 0
    correct = 0
    with torch.no_grad():
      net.eval()
      for data, targets in test_loader:
        data = data.to(device)
        spiking_data = spikegen.rate(data, num_steps=num_steps)
        targets = targets.to(device)

        # forward pass
        test_spk, test_mem = net(spiking_data[:,:,0,:,:].view(num_steps,128,num_inputs))
        # calculate total accuracy
        _, predicted = test_mem.sum(dim=0).max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        global correct_labels
        correct_labels = targets

    print(f"Total correctly classified test set images: {correct}/{total}")
    print(f"Test Set Accuracy: {100 * correct / total:.2f}%")
    return 100 * correct / total;

def plotProgress(args,currentAccuracy):
    spike_hist0 = sum(spk0_hist)[0]
    spike_hist1 = sum(spk1_hist)[0]
    spike_hist2 = sum(spk2_hist)[0]
    mem_hist0 = sum(mem0_hist)[0]/num_steps
    mem_hist1 = sum(mem1_hist)[0]/num_steps
    mem_hist2 = sum(mem2_hist)[0]/num_steps

    x_scaled = np.linspace(0, args.log_interval*len(accuracies_spike),  len(accuracies_spike),endpoint=True)
    x_scaledi = np.linspace(0, args.log_interval*len(accuracies_spike), len(accuracies_spike)*10,endpoint=True)

    smoothen_spike = interp1d(x_scaled, accuracies_spike,kind="cubic")
    yi_spike = smoothen_spike(x_scaledi)

    smoothen_mem = interp1d(x_scaled, accuracies_mem,kind="cubic")
    yi_mem = smoothen_mem(x_scaledi)

    ax[0][0].clear()
    ax[0][0].bar(range(spike_hist0.size()[0]),spike_hist0.to("cpu"))
    ax[0][0].set_title("Spike Activity for Sample {}".format(correct_labels[0]))
    ax[0][0].set_ylabel("Input Layer")

    ax[1][0].clear()
    ax[1][0].bar(range(spike_hist1.size()[0]),spike_hist1.to("cpu"))
    ax[1][0].set_ylabel("Hidden Layer")

    ax[2][0].clear()
    ax[2][0].bar(range(spike_hist2.size()[0]),spike_hist2.to("cpu"))
    ax[2][0].set_ylabel("Output Layer")
    ax[2][0].set_xlabel("Neuron")

    ax[0][1].clear()
    ax[0][1].bar(range(mem_hist0.size()[0]),mem_hist0.to("cpu"))
    ax[0][1].plot(thrs0_hist.to("cpu").detach().numpy(), color = "red", linewidth = 1, linestyle= "dashed")
    ax[0][1].set_title("Average Neuron-Potential for Sample {}".format(correct_labels[0]))

    ax[1][1].clear()
    ax[1][1].bar(range(mem_hist1.size()[0]),mem_hist1.to("cpu"))
    ax[1][1].plot(thrs1_hist.to("cpu").detach().numpy(), color = "red", linewidth = 1, linestyle= "dashed")

    ax[2][1].clear()
    ax[2][1].bar(range(mem_hist2.size()[0]),mem_hist2.to("cpu"))
    ax[2][1].plot(thrs2_hist.to("cpu").detach().numpy(), color = "red", linewidth = 1, linestyle= "dashed")
    ax[2][1].set_xlabel("Neuron")

    ax[0][2].clear()
    ax[0][2].plot(x_scaledi,yi_spike, label = "Accuracy Spike")
    ax[0][2].plot(x_scaledi,yi_mem, label = "Accuracy Potential")
    ax[0][2].legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True, shadow=True)
    ax[0][2].set_title("Learning Progress Analysis, Current Accuracy: {:.2f}%".format(currentAccuracy))
    ax[0][2].set_ylabel("Accuracy in %")
    ax[0][2].set_ylim([0,100])

    ax[1][2].clear()
    ax[1][2].plot(loss_hist, label = "Train Loss")
    ax[1][2].plot(test_loss_hist, label = "Test Loss")
    ax[1][2].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    ax[1][2].set_ylabel("Loss")

    ax[2][2].set_xlabel("Iteration")
    plt.show(block=False)
    plt.pause(1)

def main(learn_beta,learn_threshold):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=14, metavar='B',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--beta', type=float, default=0.95, metavar='N',
                        help='default decay rate for Leaky neurons (default: 0.95)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--load', action='store_true', default=False,
                        help='demonstrate the nn using a pre-trained version')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #Setup Datasets
    transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))
                ])

    mnist_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Load the network onto CUDA if available
    net = Net(args.beta,device,learn_beta,learn_threshold).to(device)
    torch.manual_seed(args.seed)

    print("Learnable parameters")
    length = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)
            length += param.size(0)
    print("In summary {} parameters".format(length))
    print()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    loss = nn.CrossEntropyLoss()

    # Outer training loop
    best = 0
    for epoch in range(args.epochs):
        train_backprop(args,net,device,train_loader,test_loader,optimizer,loss,epoch)
        current = test(net,device,test_loader)
        if(current > best):
            best = current
        plotProgress(args,current)
    if args.save_model:
        torch.save(net.state_dict(), "fashion_mnist_snn.pt")

    return best


if __name__ == '__main__':
    hyperparam = main(False,False)
    print("Beta: Hyperparam, Threshold: Hyperparam, Accuracy: {:2f}".format(hyperparam))
    plt.show(block=True)
