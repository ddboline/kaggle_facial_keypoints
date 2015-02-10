#!/usr/bin/python

import matplotlib
matplotlib.use('Agg')
import pylab as pl

from load_fn import load
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

import cPickle as pickle

with open('net1.pickle', 'rb') as f:
    net1 = pickle.load(f)

with open('net2.pickle', 'rb') as f:
    net2 = pickle.load(f)

with open('net3.pickle', 'rb') as f:
    net3 = pickle.load(f)

train_loss1 = np.array([i["train_loss"] for i in net1.train_history_])
valid_loss1 = np.array([i["valid_loss"] for i in net1.train_history_])
pl.plot(train_loss1, linewidth=3, label="net1 train")
pl.plot(valid_loss1, linewidth=3, label="net1 valid")

train_loss2 = np.array([i["train_loss"] for i in net2.train_history_])
valid_loss2 = np.array([i["valid_loss"] for i in net2.train_history_])
pl.plot(train_loss2, linewidth=3, label="net2 train")
pl.plot(valid_loss2, linewidth=3, label="net2 valid")

train_loss3 = np.array([i["train_loss"] for i in net3.train_history_])
valid_loss3 = np.array([i["valid_loss"] for i in net3.train_history_])
pl.plot(train_loss3, linewidth=3, label="net3 train")
pl.plot(valid_loss3, linewidth=3, label="net3 valid")

pl.grid()
pl.legend()
pl.xlabel("epoch")
pl.ylabel("loss")
pl.ylim(1e-3, 1e-2)
pl.yscale("log")
pl.savefig('training_loss_net3.png')

