import argparse
import torch

# Network Architecture
NUM_INPUTS = 28*28
NUM_HIDDEN = 256
NUM_OUTPUTS = 10

EPOCHS = 100
EPOCHS_FIRST_LAYER_POTION = 1/20

BATCH_SIZE = 128
BETA = 0.95
LR = 5e-4
LEARN_BETA = False
LEARN_THRESHOLD = False
# Temporal Dynamics
NUM_STEPS = 50
dtype = torch.float

#STDP
MIN_WEIGHT = 0
MAX_WEIGHT = 1
A_PLUS = 0.1
A_MINUS = A_PLUS * 1.04
STDP_DECAY = 10
DT = 1

#Corresponding Items to nn Classes
classes = ["T-Shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle Boot"]

DATA_PATH='/data/mnist'

def parse():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, metavar='N',
                        help='input batch size for training (default: {})'.format(BATCH_SIZE))
    parser.add_argument('--epochs', type=int, default=EPOCHS, metavar='N',
                        help='number of epochs to train (default: {})'.format(EPOCHS))
    parser.add_argument('--beta', type=float, default=BETA, metavar='N',
                        help='default decay rate for Leaky neurons (default: {})'.format(BETA))
    parser.add_argument('--num-steps', type=int, default=NUM_STEPS, metavar='N',
                        help='default number of simulation iteration per prediction (default: {})'.format(NUM_STEPS))
    parser.add_argument('--lr', type=float, default=LR, metavar='LR',
                        help='learning rate (default: {})'.format(LR))
    parser.add_argument('--use-stdp', action='store_true', default=False,
                        help='train snn using STDP. If not set, snn is trained using Backpropagation')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--ghost', action='store_true', default=False,
                        help='Training the nn without saving the progress')
    parser.add_argument('--train-layer', type=int, default=3, metavar='N',
                        help='Retraining the trained nn starting at specified layer. If not set, no layer is being retrained')
    parser.add_argument('--demo', action='store_true', default=False,
                        help='demonstrate the nn using a pre-trained version')
    return parser.parse_args()
