#!/usr/bin/python

import numpy as np

import matplotlib
matplotlib.use('Agg')
import pylab as pl

from load_fn import load
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

import cPickle as pickle

with open('net5.pickle', 'rb') as f:
    net5 = pickle.load(f)

with open('net6.pickle', 'rb') as f:
    net6 = pickle.load(f)

train_loss5 = np.array([i["train_loss"] for i in net5.train_history_])
valid_loss5 = np.array([i["valid_loss"] for i in net5.train_history_])
pl.plot(train_loss5, linewidth=3, label="net5 train")
pl.plot(valid_loss5, linewidth=3, label="net5 valid")

train_loss6 = np.array([i["train_loss"] for i in net6.train_history_])
valid_loss6 = np.array([i["valid_loss"] for i in net6.train_history_])
pl.plot(train_loss6, linewidth=3, label="net6 train")
pl.plot(valid_loss6, linewidth=3, label="net6 valid")
pl.grid()
pl.legend()
pl.xlabel("epoch")
pl.ylabel("loss")
pl.ylim(1e-3, 1e-2)
pl.yscale("log")
pl.savefig('training_loss_net6.png')
