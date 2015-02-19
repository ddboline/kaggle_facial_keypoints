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

with open('net3.pickle', 'rb') as f:
    net3 = pickle.load(f)

with open('net4.pickle', 'rb') as f:
    net4 = pickle.load(f)

train_loss3 = np.array([i["train_loss"] for i in net3.train_history_])
valid_loss3 = np.array([i["valid_loss"] for i in net3.train_history_])
pl.plot(train_loss3, linewidth=3, label="net3 train")
pl.plot(valid_loss3, linewidth=3, label="net3 valid")

train_loss4 = np.array([i["train_loss"] for i in net4.train_history_])
valid_loss4 = np.array([i["valid_loss"] for i in net4.train_history_])
pl.plot(train_loss4, linewidth=3, label="net4 train")
pl.plot(valid_loss4, linewidth=3, label="net4 valid")
pl.grid()
pl.legend()
pl.xlabel("epoch")
pl.ylabel("loss")
pl.ylim(1e-3, 1e-2)
pl.yscale("log")
pl.savefig('training_loss_net4.png')
