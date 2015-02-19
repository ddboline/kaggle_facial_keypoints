#!/bin/bash

sudo apt-get remove -y linux-headers-3.13.0-36 linux-headers-3.13.0-36-generic linux-image-3.13.0-36-generic
sudo apt-get install -y --reinstall nvidia-340 nvidia-340-dev nvidia-340-uvm

sudo modprobe nvidia-340
sudo modprobe nvidia-340-uvm

# virtualenv venv
# source venv/bin/activate

# pip install -r requirements.txt
# pip install -r requirements-2.txt

### getting this to work may require:
### installing from https://github.com/Theano/Theano.git
### changing a line in theano/sandbox/cuda/cuda_ndarray.cuh (more likely the former than the latter)
sudo pip install --upgrade theano 2>&1 > theano_build.log
sudo pip install git+https://github.com/benanne/Lasagne.git 2>&1 > lasagne_build.log
sudo pip install nolearn 2>&1 > nolearn_build.log

for F in training.zip test.zip IdLookupTable.csv SampleSubmission.csv;
do
    scp ubuntu@ddbolineinthecloud.mooo.com:~/setup_files/build/kaggle_facial_keypoints/$F .
done

for F in training.zip test.zip;
do
    unzip -x $F;
done

cd $HOME
git clone pip install https://github.com/lisa-lab/pylearn2.git
### this doesn't really work any other way
cd pylearn2
echo "You will need to build pylearn2 manually (setup.py written by idiot...)"
