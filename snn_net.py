#SNN
import snntorch as snn

import numpy as np

#PyTorch
import torch
import torch.nn as nn
from torch.nn import functional as F

from config import *
from log import *

class LeakySTDP(snn.Leaky):
    def __init__(
        self,
        beta,
        threshold=1.0,
        spike_grad=None,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        reset_mechanism="subtract",
        output=False,
    ):
        super().__init__(
            beta,
            threshold,
            spike_grad,
            init_hidden,
            inhibition,
            learn_beta,
            reset_mechanism,
            output,
        )

    def forward(self, input_, mem=False,pre=False,post=False):
        if pre.size(0) == 0:
            pre = torch.zeros_like(input_)
        if post.size(0) == 0:
            post = torch.zeros_like(input_)
        spk, mem = super().forward(input_,mem)
        dpre = -(DT / STDP_DECAY) * pre + A_PLUS * spk
        dpost = -(DT / STDP_DECAY) * post - A_MINUS * spk
        pre = pre + dpre
        post = post + dpost
        return spk, mem, pre, post

    def init_leaky(self):
        mem = super().init_leaky()
        pre = snn._SpikeTensor(init_flag=False)
        post = snn._SpikeTensor(init_flag=False)
        return mem, pre, post

#Define Network
class Net(nn.Module):
    def __init__(self,args,device):
        super().__init__()

        if(LEARN_THRESHOLD):
            threshold0 = nn.Parameter(torch.rand(NUM_INPUTS,requires_grad=True).to(device))
            threshold1 = nn.Parameter(torch.rand(NUM_HIDDEN,requires_grad=True).to(device))
            threshold2 = nn.Parameter(torch.rand(NUM_OUTPUTS,requires_grad=True).to(device))
            self.register_parameter(name = "threshold0", param = threshold0)
            self.register_parameter(name = "threshold1", param = threshold1)
            self.register_parameter(name = "threshold2", param = threshold2)
        else:
            threshold0 = torch.ones(NUM_INPUTS).to(device)
            threshold1 = torch.ones(NUM_HIDDEN).to(device)
            threshold2 = torch.ones(NUM_OUTPUTS).to(device)

        #Initialize layers
        self.lif0 = LeakySTDP(beta=args.beta,learn_beta=LEARN_BETA,threshold = 0, reset_mechanism="subtract")
        self.fc1 = nn.Linear(NUM_INPUTS, NUM_HIDDEN,bias = False)
        self.lif1 = LeakySTDP(beta=args.beta,learn_beta=LEARN_BETA,threshold = threshold1, reset_mechanism="zero")
        self.fc2 = nn.Linear(NUM_HIDDEN, NUM_OUTPUTS,bias = False)
        self.lif2 = LeakySTDP(beta=args.beta,learn_beta=False,threshold = threshold2, reset_mechanism="zero")

        if args.use_stdp:
            self.fc1.weight.requires_grad = False
            self.fc2.weight.requires_grad = False

            neg_weights_hidden = (F.dropout(torch.ones_like(self.fc1.weight),p=0.63) * -0.38 * 2) + 1
            neg_weights_out = (F.dropout(torch.ones_like(self.fc2.weight),p=0.63) * -0.38 * 2) + 1

            self.fc1.weight = torch.nn.parameter.Parameter(F.dropout(torch.rand_like(self.fc1.weight) * MAX_WEIGHT, p=0.6) * (1-0.6) * neg_weights_hidden, requires_grad = False)
            self.fc2.weight = torch.nn.parameter.Parameter(F.dropout(torch.rand_like(self.fc2.weight) * MAX_WEIGHT, p=0.6) * (1-0.6) * neg_weights_out, requires_grad = False)

    def forward(self, x, args, stdp = False, layer = 0):
        #Initialize hidden states at t=0
        mem0, pre0, post0 = self.lif0.init_leaky()
        mem1, pre1, post1 = self.lif1.init_leaky()
        mem2, pre2, post2 = self.lif2.init_leaky()

        #Clear log-history
        reset_log(self.lif1.threshold,self.lif2.threshold)

        #Record layer activities
        spk0_rec = []
        pre0_rec = []
        post0_rec = []

        spk1_rec = []
        pre1_rec = []
        post1_rec = []

        spk2_rec = []
        mem2_rec = []
        pre2_rec = []
        post2_rec = []

        for step in range(args.num_steps):
            spk0, mem0, pre0, post0 = self.lif0(x[step],mem0, pre0, post0)
            cur1 = F.relu(self.fc1(spk0))
            spk1, mem1, pre1, post1 = self.lif1(cur1, mem1, pre1, post1)
            cur2 = F.relu(self.fc2(spk1))
            spk2, mem2, pre2, post2 = self.lif2(cur2, mem2, pre2, post2)

            #Track Spike-Activity and Membrane Potential
            write_log(spk0[0],mem1[0],spk1[0],mem2[0],spk2[0])
            spk0_rec.append(spk0)
            pre0_rec.append(pre0)
            post0_rec.append(post0)

            spk1_rec.append(spk1)
            pre1_rec.append(pre1)
            post1_rec.append(post1)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
            pre2_rec.append(pre2)
            post2_rec.append(post2)


            #Update via stdp-rule
            if stdp:
                if(layer == 0):
                    dw1 = calc_pre_weight_change(args,spk0,post1,self.fc1.weight)
                    dw1 = calc_post_weight_change(args,spk1,pre0,dw1)
                    self.fc1.weight = torch.nn.parameter.Parameter(dw1)
                if(layer == 1):
                    dw2 = calc_pre_weight_change(args,spk1,post2,self.fc2.weight)
                    dw2 = calc_post_weight_change(args,spk2,pre1,dw2)
                    self.fc2.weight = torch.nn.parameter.Parameter(dw2)

        spk_return = []
        pre_return = []
        post_return = []

        spk_return.append(torch.stack(spk0_rec,dim=0))
        spk_return.append(torch.stack(spk1_rec,dim=0))
        spk_return.append(torch.stack(spk2_rec,dim=0))

        pre_return.append(torch.stack(pre0_rec,dim=0))
        pre_return.append(torch.stack(pre1_rec,dim=0))
        pre_return.append(torch.stack(pre2_rec,dim=0))

        post_return.append(torch.stack(post0_rec,dim=0))
        post_return.append(torch.stack(post1_rec,dim=0))
        post_return.append(torch.stack(post2_rec,dim=0))
        return spk_return, torch.stack(mem2_rec, dim=0), pre_return, post_return

def calc_pre_weight_change(args, spk, pre, weights):
    adjustments = torch.einsum("bi,bj->ij", pre, spk) / args.batch_size
    adjusted_weights = (weights + (abs(weights) * adjustments)).clamp(-MAX_WEIGHT,MAX_WEIGHT)
    return adjusted_weights

def calc_post_weight_change(args, spk, post, weights):
    adjustments = torch.einsum("bi,bj->ij", spk, post) / args.batch_size
    adjusted_weights = (weights + (abs(weights) * adjustments)).clamp(-MAX_WEIGHT,MAX_WEIGHT)
    return adjusted_weights
