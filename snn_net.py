#SNN
import snntorch as snn
import numpy as np

#PyTorch
import torch
import torch.nn as nn
from torch.nn import functional as F

from config import *
from log import *

#Define Network
class Net(nn.Module):
    def __init__(self,beta,device):
        super().__init__()

        if(LEARN_THRESHOLD):
            #threshold0 = nn.Parameter(torch.rand(NUM_INPUTS,requires_grad=True).to(device))
            threshold1 = nn.Parameter(torch.rand(NUM_HIDDEN,requires_grad=True).to(device))
            #self.register_parameter(name = "threshold0", param = threshold0)
            self.register_parameter(name = "threshold1", param = threshold1)
        else:
            #threshold0 = torch.ones(NUM_INPUTS).to(device)
            threshold1 = torch.ones(NUM_HIDDEN).to(device)
        threshold2 = torch.ones(NUM_OUTPUTS).to(device)

        #Initialize layers
        #self.lif0 = snn.Leaky(beta=beta,learn_beta=LEARN_BETA,threshold = threshold0, reset_mechanism="zero")
        self.fc1 = nn.Linear(NUM_INPUTS, NUM_HIDDEN,bias = False)
        self.lif1 = snn.Leaky(beta=beta,learn_beta=LEARN_BETA,threshold = threshold1, reset_mechanism="zero")
        self.fc2 = nn.Linear(NUM_HIDDEN, NUM_OUTPUTS,bias = False)
        self.lif2 = snn.Leaky(beta=beta,learn_beta=False,threshold = threshold2, reset_mechanism="zero")

    def forward(self, x):
        #Initialize hidden states at t=0
        #mem0 = self.lif0.init_leaky()
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        #Clear log-history
        reset_log(self.lif1.threshold,self.lif2.threshold)

        #Record the final layer
        spk0_rec = []
        spk1_rec = []
        spk2_rec = []
        spk_out = []
        mem2_rec = []

        for step in range(x.size(0)):
            #spk0, mem0 = self.lif0(x[step],mem0)
            cur1 = F.relu(self.fc1(x[step]))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = F.relu(self.fc2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            #Track Spike-Activity and Membrane Potential
            write_log(x[step][0],mem1[0],spk1[0],mem2[0],spk2[0])
            spk0_rec.append(x[step])
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        spk_out.append(torch.stack(spk0_rec,dim=0))
        spk_out.append(torch.stack(spk1_rec,dim=0))
        spk_out.append(torch.stack(spk2_rec,dim=0))
        return spk_out, torch.stack(mem2_rec, dim=0)
