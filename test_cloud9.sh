#!/bin/bash

./test.py
./net1_single_hidden_layer.py
./plot_net1_training_loss.py
./net2_convolutions.py
./plot_net2_training_loss.py
./net3_convolutions_flip.py
./plot_net3_training_loss.py

scp *.png *.pickle ddboline@ddbolineathome.mooo.com:~/setup_files/build/kaggle_facial_keypoints/
