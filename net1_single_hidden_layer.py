#!/usr/bin/python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as pl

from load_fn import load
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

net1 = NeuralNet(
    layers=[ # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),],
    # layer parameters:
    input_shape=(None, 9216),  # 96x96 input pixels per batch
    hidden_num_units=100,  # number of units in hidden layer
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=30,  # 30 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=400,  # we want to train this many epochs
    verbose=1,)

X, y = load()
print X.shape, y.shape

net1.fit(X, y)

import cPickle as pickle
with open('net1.pickle', 'wb') as f:
    pickle.dump(net1, f, -1)

train_loss = np.array([i["train_loss"] for i in net1.train_history_])
valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
pl.plot(train_loss, linewidth=3, label="train")
pl.plot(valid_loss, linewidth=3, label="valid")
pl.grid()
pl.legend()
pl.xlabel("epoch")
pl.ylabel("loss")
pl.ylim(1e-3, 1e-2)
pl.yscale("log")
pl.savefig('training_loss.png')
