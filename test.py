import numpy as np
from numpy import random
import quadprog
from models import SeqSleepNet, TinySeqSleepNet
from models import init_weight
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.param = nn.Parameter(torch.randn(5))
        self.c = torch.zeros(5)

    def clear(self):
        for name, param in self.named_parameters():
            if name == 'param':
                param.data *= self.c


if __name__ == '__main__':
    net = SeqSleepNet()
    net.apply(init_weight)
    torch.save(net.state_dict(), 'saved_network.pt')
