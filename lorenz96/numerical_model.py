import numpy as np

def lorenz96(t, x, F=8):
    """Lorenz 96 model with constant forcing"""
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return dxdt
