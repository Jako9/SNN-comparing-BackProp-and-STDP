#PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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

    #Outer training loop
    for epoch in range(args.epochs):
        if(args.use_stdp):
            if(epoch < 25):
                train_stdp(net,args,device,train_loader,test_loader,optimizer,loss,epoch,layer = 0)
            else:
                train_stdp(net,args,device,train_loader,test_loader,optimizer,loss,epoch,layer = 1)
        else:
            train_backprop(net,args,device,train_loader,test_loader,optimizer,loss,epoch)
        current , _ = calc_acc(net,args,device,test_loader,output=True)
        track_best(current)
        plotProgress(args,current,LEARN_THRESHOLD)

    if args.save_model:
        torch.save(net.state_dict(), "fashion_mnist_snn.pt")


if __name__ == '__main__':
    main()
    print("Beta: Hyperparam, Threshold: Hyperparam, best accuracy: {:.2f}".format(best()))
    plt.show(block=True)
