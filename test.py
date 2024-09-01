import numpy as np
from numpy import random
import quadprog


if __name__ == '__main__':
    g = random.normal(loc=0.0, scale=1.0, size=5)
    G = random.normal(loc=0.0, scale=1.0, size=(3, 5))
    H = np.eye(5)
    f = g
    A = G.T
    b = np.zeros(3)
    x = quadprog.solve_qp(H, f, A, b)
    print(x)
    x = x[0]
    print(np.dot(x, G[0]))
    print(np.dot(x, G[1]))
    print(np.dot(x, G[2]))
    print(np.linalg.norm(x - g) ** 2)
