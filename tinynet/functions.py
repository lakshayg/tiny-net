import numpy as np

## Loss functions (return loss and derivative)
def mse(yd, yp):
    e = yp - yd
    loss = 0.5 * np.sum(e**2, axis=1)
    return loss, e

def xentropy(yd, yp):
    l = np.log(yp)
    loss = -np.sum(yd * l, axis=1)
    deriv = -(yd/yp)
    return loss, deriv

# activation functions
def relu(x):
    x[x<0] = 0
    return x
def d_relu(y):
    y[y>0] = 1
    return y
def tanh(x):
    return np.tanh(x)
def d_tanh(y):
    return 1 - y**2

def sigmoid(x):
    return 1./(1 + np.exp(-x))
def d_sigmoid(y):
    return y - y**2

