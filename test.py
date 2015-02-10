#!/usr/bin/python

import os

from load_fn import load

x, y = load()
print x.shape, x.min(), x.max()
print y.shape, y.min(), y.max()
