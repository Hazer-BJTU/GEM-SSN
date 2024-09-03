import numpy as np
from numpy import random
import quadprog
from models import SeqSleepNet, TinySeqSleepNet
import torch
import torch.nn as nn


if __name__ == '__main__':
    '''
    g = random.normal(loc=0.0, scale=1.0, size=2064)
    G = random.normal(loc=0.0, scale=1.0, size=(10, 2064))
    H = np.eye(2064)
    f = g
    A = G.T
    b = np.zeros(10)
    x = quadprog.solve_qp(H, f, A, b)
    print(x[0])
    '''
    net = SeqSleepNet()
    for param in net.parameters():
        print(param.detach())
        break
