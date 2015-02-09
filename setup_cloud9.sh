#!/bin/bash

sudo apt-get update
sudo apt-get install -y python-pip
sudo apt-get install -y gcc g++ gfortran build-essential git wget linux-image-generic libopenblas-dev python-dev python-pip python-nose python-numpy python-scipy

sudo pip install --upgrade theano

wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_6.5-14_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_6.5-14_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda


# sudo pip install numpy
# sudo pip install theano
# sudo pip install git+https://github.com/benanne/Lasagne
# sudo pip install nolearn
# sudo pip install git+https://github.com/lisa-lab/pylearn2.git
# 
# 
# sudo pip install -r requirements.txt
# sudo pip install -r requirements-2.txt
