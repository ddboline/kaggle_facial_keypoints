#!/bin/bash

sudo apt-get update
sudo apt-get install -y gcc g++ gfortran build-essential
sudo apt-get install -y git wget linux-image-generic libopenblas-dev
sudo apt-get install -y python-dev python-nose python-numpy python-scipy
sudo apt-get install -y cython linux-image-extra-virtual htop
sudo apt-get install -y python-matplotlib python-pandas python-virtualenv

sudo bash -c "echo deb http://ddbolineathome.mooo.com/deb/trusty ./ > /etc/apt/sources.list.d/py2deb.list"
sudo apt-get update
sudo apt-get install -y --force-yes ipython python-blaze python-theano python-lasagne python-nolearn python-pylearn2
