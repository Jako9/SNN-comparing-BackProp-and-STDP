#PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os
import math

from config import *
from log import *
from train import *
import snn_net

def main():
    #Training settings
    args = parse()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #Setup Datasets
    transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))
                ])

    mnist_train = datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(DATA_PATH, train=False, download=True, transform=transform)

    #Create DataLoaders
    train_loader = DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size, shuffle=True, drop_last=True)

    #Load the network onto CUDA if available
    net = snn_net.Net(args,device).to(device)
    torch.manual_seed(args.seed)

    print_params(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    loss = nn.CrossEntropyLoss()

    setup_plot()

    #Outer training loop
    train_hidden_layer = not os.path.isfile("model/saved_layer_hidden.net") or args.train_layer <= 0
    train_output_layer = not os.path.isfile("model/saved_layer_output.net")  or args.train_layer <= 1
    epoch = 0
    while epoch < args.epochs:
        if(args.use_stdp):
            if(epoch < math.ceil(args.epochs * EPOCHS_FIRST_LAYER_POTION)):#train hidden layer
                if train_hidden_layer:
                    train_stdp(net,args,device,train_loader,epoch,layer = 0)
                    if epoch == math.ceil(args.epochs * EPOCHS_FIRST_LAYER_POTION) - 1 and not arg.ghost:
                        print("-------SAVING FIRST LAYER-------")
                        torch.save(net.state_dict(), "model/saved_layer_hidden.net")
                else:
                    print("-------LOADING PRE-TRAINED FIRST LAYER-------")
                    epoch = math.ceil(args.epochs * EPOCHS_FIRST_LAYER_POTION)
                    net = snn_net.Net(args,device).to(device)
                    net.load_state_dict(torch.load("model/saved_layer_hidden.net"))
                    continue

            else:#train output layer
                if train_output_layer:
                    #train_stdp(net,args,device,train_loader,epoch,layer = 1)
                    train_backprop(net,args,device,train_loader,test_loader,optimizer,loss,epoch)
                    if epoch == args.epochs - 1 and not arg.ghost:
                        print("-------SAVING SECOND LAYER-------")
                        torch.save(net.state_dict(), "model/saved_layer_output.net")
                else:
                    print("-------LOADING PRE-TRAINED SECOND LAYER-------")
                    epoch = args.epochs
                    net = snn_net.Net(args,device).to(device)
                    net.load_state_dict(torch.load("model/saved_layer_output.net"))
                    continue
        else:
            train_backprop(net,args,device,train_loader,test_loader,optimizer,loss,epoch)
        current , _ = calc_acc(net,args,device,test_loader,output=True)
        track_best(current)
        plotProgress(args,current,LEARN_THRESHOLD,epoch)
        epoch += 1


if __name__ == '__main__':
    main()
    print("Beta: Hyperparam, Threshold: Hyperparam, best accuracy: {:.2f}".format(best()))
    plt.show(block=True)
