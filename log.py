import torch

#Plotting
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from config import *

#for plotting
spk0_hist = []
spk1_hist = []
spk2_hist = []
mem0_hist = []
mem1_hist = []
mem2_hist = []
thrs0_hist = torch.zeros(NUM_INPUTS)
thrs1_hist = torch.zeros(NUM_HIDDEN)
thrs2_hist = torch.zeros(NUM_OUTPUTS)
train_loss_hist_ = []
test_loss_hist_ = []
accuracies_spike = []
accuracies_mem = []
correct_labels = torch.zeros(1)
bestAcc = 0


fig, ax = plt.subplots(3,3,figsize=(20,8))


def reset_log(threshold0,threshold1,threshold2):
    spk0_hist.clear()
    spk1_hist.clear()
    spk2_hist.clear()
    mem0_hist.clear()
    mem1_hist.clear()
    mem2_hist.clear()
    global thrs0_hist
    global thrs1_hist
    global thrs2_hist
    thrs0_hist = threshold0
    thrs1_hist = threshold1
    thrs2_hist = threshold2

def write_log(mem0,spk0,mem1,spk1,mem2,spk2):
    mem0_hist.append(mem0)
    spk0_hist.append(spk0)
    mem1_hist.append(mem1)
    spk1_hist.append(spk1)
    mem2_hist.append(mem2)
    spk2_hist.append(spk2)

def track_train_loss(loss):
    train_loss_hist_.append(loss)

def track_test_loss(loss):
    test_loss_hist_.append(loss)

def train_loss_hist(counter):
    return train_loss_hist_[counter]

def test_loss_hist(counter):
    return test_loss_hist_[counter]

def track_accuracy(accuracy_spike,accuracy_mem):
    accuracies_spike.append(accuracy_spike)
    accuracies_mem.append(accuracy_mem)

def track_correct_labels(targets):
    global correct_labels
    correct_labels = targets

def track_best(current):
    global bestAcc
    if(current > bestAcc):
        bestAcc = current

def best():
    return bestAcc

def plotProgress(args,currentAccuracy,learn_threshold):
    spike_hist0 = sum(spk0_hist)
    spike_hist1 = sum(spk1_hist)
    spike_hist2 = sum(spk2_hist)
    mem_hist0 = sum(mem0_hist)/args.num_steps
    mem_hist1 = sum(mem1_hist)/args.num_steps
    mem_hist2 = sum(mem2_hist)/args.num_steps
    mem2_decay = torch.swapaxes(torch.stack(mem2_hist),0,1)

    x_scaled = np.linspace(0, args.log_interval*len(accuracies_spike),  len(accuracies_spike),endpoint=True)
    x_scaledi = np.linspace(0, args.log_interval*len(accuracies_spike), len(accuracies_spike)*10,endpoint=True)

    xpot_scaled = np.linspace(0, mem2_decay.size(1),  mem2_decay.size(1),endpoint=True)
    xpot_scaledi = np.linspace(0, mem2_decay.size(1), mem2_decay.size(1)*10,endpoint=True)

    mempot_interp = []
    for i in range(mem2_decay.size(0)):
        smoothen = interp1d(xpot_scaled, mem2_decay[i].to("cpu"),kind="cubic")
        mempot_interp.append(smoothen(xpot_scaledi))

    smoothen_spike = interp1d(x_scaled, accuracies_spike,kind="cubic")
    yi_spike = smoothen_spike(x_scaledi)

    smoothen_mem = interp1d(x_scaled, accuracies_mem,kind="cubic")
    yi_mem = smoothen_mem(x_scaledi)

    ax[0][0].clear()
    ax[0][0].bar(range(spike_hist0.size(0)),spike_hist0.to("cpu"))
    ax[0][0].set_title("Spike Activity for Sample {}".format(correct_labels[0]))
    ax[0][0].set_ylabel("Input Layer")

    ax[1][0].clear()
    ax[1][0].bar(range(spike_hist1.size(0)),spike_hist1.to("cpu"))
    ax[1][0].set_ylabel("Hidden Layer")

    ax[2][0].clear()
    ax[2][0].bar(range(spike_hist2.size(0)),spike_hist2.to("cpu"))
    ax[2][0].set_ylabel("Output Layer")
    ax[2][0].set_xlabel("Neuron")

    ax[0][1].clear()
    ax[0][1].bar(range(mem_hist0.size(0)),mem_hist0.to("cpu"))
    ax[0][1].plot(thrs0_hist.to("cpu").detach().numpy(), color = "red", linewidth = 1, linestyle= "dashed")
    ax[0][1].set_title("Average Neuron-Potential for Sample {}".format(correct_labels[0]))

    ax[1][1].clear()
    ax[1][1].bar(range(mem_hist1.size(0)),mem_hist1.to("cpu"))
    ax[1][1].plot(thrs1_hist.to("cpu").detach().numpy(), color = "red", linewidth = 1, linestyle= "dashed")

    ax[2][1].clear()
    ax[2][1].bar(range(mem_hist2.size(0)),mem_hist2.to("cpu"))
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
    ax[1][2].plot(train_loss_hist_, label = "Train Loss")
    ax[1][2].plot(test_loss_hist_, label = "Test Loss")
    ax[1][2].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    ax[1][2].set_ylabel("Loss")

    ax[2][2].clear()
    for i in range(len(mempot_interp)):
        ax[2][2].plot(xpot_scaledi,mempot_interp[i], label = "Neuron {}".format(i))
    if not LEARN_THRESHOLD:
        ax[2][2].plot(xpot_scaledi,torch.ones(len(xpot_scaledi)).to("cpu"), color = "red", linewidth = 1, linestyle= "dashed")
    ax[2][2].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    ax[2][2].set_ylabel("Membrane Potential Output")
    ax[2][2].set_xlabel("Time Step")
    plt.show(block=False)
    plt.pause(1)

def train_printer(epoch,iter_counter,loss,test_loss,accuracy_spike,accuracy_mem):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss:.2f}")
    print(f"Test Set Loss: {test_loss:.2f}")
    print(f"Test Set Accuracy with Spikes: {accuracy_spike:.2f}%")
    print(f"Test Set Accuracy with Membrane Potential: {accuracy_mem:.2f}%")
    print("\n")

def print_epoch(correct_spike,total):
    print(f"Total correctly classified test set images: {correct_spike}/{total}")
    print(f"Test Set Accuracy: {100 * correct_spike / total:.2f}%")
    print("\n")

def print_params(net):
    print("---Learnable parameters---")
    length = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)
            length += param.size(0)
    print("In summary {} parameters".format(length))
    print()
