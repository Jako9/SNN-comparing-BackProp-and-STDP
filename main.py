#PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os

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
    finished_first_layer = False

    while epoch < args.epochs:
        #train via backpropagation
        if not args.use_stdp:
            train_backprop(net,args,device,train_loader,test_loader,optimizer,loss,epoch)

        elif not finished_first_layer: #train hidden layer via stdp
            if train_hidden_layer:
                train_stdp(net,args,device,train_loader,test_loader,epoch,layer = 0)
            else: #load pre-trained hidden layer
                print("-------LOADING PRE-TRAINED FIRST LAYER-------")
                finished_first_layer = True
                net = snn_net.Net(args,device).to(device)
                net.load_state_dict(torch.load("model/saved_layer_hidden.net"))
                continue

        else: #train output layer via stdp
                if train_output_layer:
                    train_stdp(net,args,device,train_loader,test_loader,epoch,layer = 1)
                    #train_backprop(net,args,device,train_loader,test_loader,optimizer,loss,epoch)
                else: #load pre-trained output layer
                    print("-------LOADING PRE-TRAINED SECOND LAYER-------")
                    epoch = args.epochs
                    net = snn_net.Net(args,device).to(device)
                    net.load_state_dict(torch.load("model/saved_layer_output.net"))
                    continue

        #evaluate results of trained epoch
        current , _ , garbage, average = calc_acc(net,args,device,test_loader,output=True)

        #Saving hidden layer if trained
        if garbage[0] < 0.001 and average[0] < 50 :
            if not finished_first_layer and not args.ghost:
                print("-------SAVING FIRST LAYER-------")
                torch.save(net.state_dict(), "model/saved_layer_hidden.net")
            finished_first_layer = True
        #Saving output layer if trained
        if (finished_first_layer and garbage[1] < 0.001 and average[1] < 20) or epoch == args.epochs - 1:
            if not args.ghost:
                print("-------SAVING OUT LAYER-------")
                torch.save(net.state_dict(), "model/saved_layer_output.net")
            epoch = args.epochs

        #track training stats of current epoch
        track_best(current)
        plotProgress(args,current,LEARN_THRESHOLD,epoch)
        epoch += 1

    #Calculate output classes
    if args.use_stdp:
        print("-------CALCULATING NEURON VOTINGS-------")
        calculate_voting(net,args,train_loader,device)
    calc_acc(net,args,device,test_loader,output=True, last = True)

if __name__ == '__main__':
    main()
    print("Beta: Hyperparam, Threshold: Hyperparam, best accuracy: {:.2f}".format(best()))
    plt.show(block=True)
