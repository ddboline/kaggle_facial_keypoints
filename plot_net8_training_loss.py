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

with open('net7.pickle', 'rb') as f:
    net7 = pickle.load(f)

with open('net8.pickle', 'rb') as f:
    net8 = pickle.load(f)

train_loss7 = np.array([i["train_loss"] for i in net7.train_history_])
valid_loss7 = np.array([i["valid_loss"] for i in net7.train_history_])
pl.plot(train_loss7, linewidth=3, label="net7 train")
pl.plot(valid_loss7, linewidth=3, label="net7 valid")

train_loss8 = np.array([i["train_loss"] for i in net8.train_history_])
valid_loss8 = np.array([i["valid_loss"] for i in net8.train_history_])
pl.plot(train_loss8, linewidth=3, label="net8 train")
pl.plot(valid_loss8, linewidth=3, label="net8 valid")
pl.grid()
pl.legend()
pl.xlabel("epoch")
pl.ylabel("loss")
pl.ylim(1e-3, 1e-2)
pl.yscale("log")
pl.savefig('training_loss_net8.png')
