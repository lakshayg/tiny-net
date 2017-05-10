#!/usr/bin/python3 -u
import numpy as np

import sys; sys.path.append('..')
from tinynet.net import Sequential
from tinynet.layers import Dense, Activation, Dropout
from tinynet.functions import mse
from tinynet.optimizers import adam

def train_xor_net():
    x = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.double)
    y = np.array([[0],[1],[1],[0]], dtype=np.double)
    x[x==0] = 0.01; x[x==1] = 0.99;
    y[y==0] = 0.01; y[y==1] = 0.99;

    net = Sequential([
        Dense(input_dim=2, output_dim=7),
        Activation('tanh'),
        Dropout(ratio=0.5),

        Dense(input_dim=7, output_dim=1),
        Activation('sigmoid')
    ])

    net.configure(batch_size=2, objective=mse,
                  optimizer=adam(lr=1e-2))
    net.train(x, y, epochs=200)
    print(net.predict(x))

train_xor_net()
