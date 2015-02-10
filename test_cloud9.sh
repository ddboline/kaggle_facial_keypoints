#!/bin/bash

sudo modprobe nvidia-340
sudo modprobe nvidia-340-uvm

# sudo pip install --upgrade theano
# sudo pip install git+https://github.com/benanne/Lasagne
# sudo pip install nolearn
# sudo pip install git+https://github.com/lisa-lab/pylearn2.git

virtualenv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-2.txt

for F in training.zip test.zip IdLookupTable.csv SampleSubmission.csv;
do
    scp ubuntu@ddbolineinthecloud.mooo.com:~/setup_files/build/kaggle_facial_keypoints/$F .
done

for F in training.zip test.zip;
do
    unzip -x $F;
done

./test.py
./single_hidden_layer.py
./plot_net1_training_loss.py
./convolutions.py
./plot_net2_training_loss.py
./convolutions_flip.py
./plot_net3_training_loss.py

scp *.png *.pickle ddboline@ddbolineathome.mooo.com:~/setup_files/build/kaggle_facial_keypoints/
