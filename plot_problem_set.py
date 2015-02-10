#!/usr/bin/python

import matplotlib
matplotlib.use('Agg')
import pylab as pl

from load_fn import load

import cPickle as pickle

with open('net1.pickle', 'rb') as f:
    net1 = pickle.load(f)

with open('net2.pickle', 'rb') as f:
    net2 = pickle.load(f)

sample1 = load(test=True)[0][6:7]
sample2 = load2d(test=True)[0][6:7]
y_pred1 = net1.predict(sample1)[0]
y_pred2 = net2.predict(sample2)[0]

fig = pl.figure(figsize=(6, 3))
ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
plot_sample(sample1[0], y_pred1, ax)
ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
plot_sample(sample1[0], y_pred2, ax)
pl.savefig('problem_set.png')
