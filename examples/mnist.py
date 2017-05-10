#!/usr/bin/python3 -u

import sys; sys.path.append('..')
from tinynet.net import Sequential
from tinynet.layers import *
from tinynet.optimizers import *
from tinynet.functions import *

import sklearn.datasets
import numpy as np
import time

def num2onehot(y):
    # m = 4*np.eye(10) + np.ones(10)
    # for i in range(10):
        # e = np.exp(m[i])
        # m[i] = e / e.sum()
    m = np.eye(10, dtype=np.float64)
    return m[y.astype(int)]

def onehot2num(y):
    return np.argmax(y, axis=1)

def load_mnist_data():
    mnist = sklearn.datasets.fetch_mldata('MNIST original', data_home='./data')
    perm = list(range(mnist['data'].shape[0])); np.random.shuffle(perm)
    X = mnist['data'][perm] / 255.0
    y = mnist['target'][perm]
    X_train = X[:50000,:]
    y_train = y[:50000]
    X_val = X[50000:60000,:]
    y_val = y[50000:60000]
    X_test = X[60000:,:]
    y_test = y[60000:]
    return X_train, y_train, X_val, y_val, X_test, y_test

def eval_mnist_accuracy(net, x_test, y_test):
    yp = np.array([net.predict(x[None]).argmax() for x in x_test])
    return np.mean(yp == y_test)

def train_mnist_mlp():
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data()
    y_train = num2onehot(y_train)
    net = Sequential([
        Dense(784, 78),
        Activation('relu'),
        Dropout(ratio=0.4),

        Dense(78 ,10),
        Softmax()
    ])
    batch_size = int(input('batch size: '))
    net.configure(batch_size=batch_size,
                  optimizer=adam(lr=1e-3),
                  objective=xentropy)

    best_acc = 0
    bad_count = 0
    netfile = input('netfile: ') #'net/lenet5_bs32-2.p'
    for epoch in range(200): # max 200 iterations
        t1 = time.time()
        net.train(X_train, y_train, epochs=1)
        print('epoch', epoch+1, 'time', time.time()-t1)
        val_acc = eval_mnist_accuracy(net, X_val, y_val)
        print('val accuracy', val_acc)

        # if val_acc < best_acc:
            # bad_count += 1
        # else: # save net if it performed good
            # net.saveas(netfile)
            # best_acc = val_acc
            # bad_count = 0
        # if bad_count > 1:
            # break
    net = Sequential(netfile)
    print('test accuracy', eval_mnist_accuracy(net, X_test, y_test))

def train_mnist_cnn():
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data()
    X_train = X_train.reshape((-1, 1, 28, 28))
    X_val = X_val.reshape((-1, 1, 28, 28))
    X_test = X_test.reshape((-1, 1, 28, 28))
    y_train = num2onehot(y_train)
    net = Sequential([
        Conv(input_dim=(1,28,28), n_filters=6, size=5),
        Activation(name='relu'),
        Maxpool(input_dim=(6,28,28), size=2, stride=2),

        Conv(input_dim=(6,14,14), n_filters=16, size=5, padding=0),
        Activation(name='relu'),
        Maxpool(input_dim=(16,10,10), size=2, stride=2),

        Dropout(ratio=0.5),
        Conv2Dense(output_dim=16*5*5),

        Dense(input_dim=16*5*5, output_dim=120),
        Activation(name='relu'),

        Dense(input_dim=120, output_dim=84),
        Activation(name='relu'),

        Dense(input_dim=84, output_dim=10),
        Softmax()
    ])

    batch_size = int(input('batch size: '))
    learning_rate = 5e-4
    net.configure(batch_size=batch_size,
                  optimizer=adam(weight_decay=1e-7, lr=learning_rate),
                  objective=xentropy)

    best_acc = 0
    bad_count = 0
    netfile = input('netfile: ')
    for epoch in range(200): # max 200 iterations
        t1 = time.time()
        net.train(X_train, y_train, epochs=1, display=True)
        print('epoch', epoch+1, 'time', time.time()-t1)
        val_acc = eval_mnist_accuracy(net, X_val, y_val)
        print('val accuracy', val_acc)

        if val_acc < best_acc:
            bad_count += 1
        else: # save net if it performed good
            net.saveas(netfile)
            best_acc = val_acc
            bad_count = 0
        if bad_count > 1:
            break
    net = Sequential(netfile)
    print('test accuracy', eval_mnist_accuracy(net, X_test, y_test))

if __name__ == '__main__':
    # train_mnist_mlp()
    train_mnist_cnn()
