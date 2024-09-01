import numpy as np
from numpy import random
import quadprog
from models import SeqSleepNet
import torch
import torch.nn as nn


if __name__ == '__main__':
    '''
    g = random.normal(loc=0.0, scale=1.0, size=40000)
    G = random.normal(loc=0.0, scale=1.0, size=(10, 40000))
    H = np.eye(40000)
    f = g
    A = G.T
    b = np.zeros(10)
    x = quadprog.solve_qp(H, f, A, b)
    print(x)
    x = x[0]
    print(np.dot(x, G[0]))
    print(np.dot(x, G[1]))
    print(np.dot(x, G[2]))
    print(np.linalg.norm(x - g) ** 2)
    '''
    net = SeqSleepNet()
    X = torch.zeros((5, 10, 129, 48), dtype=torch.float32)
    y = torch.zeros((5, 10), dtype=torch.int64)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    loss = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    y_hat = net(X)
    y = y.view(-1)
    L = loss(y_hat, y)
    L.backward()
    for param in net.parameters():
        if param.grad is not None:
            grad = param.grad.view(-1)
            print(grad.data.pow(2))
