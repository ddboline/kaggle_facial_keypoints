#!/bin/bash

# for F in training.zip test.zip IdLookupTable.csv SampleSubmission.csv;
# do
#     scp ubuntu@ddbolineinthecloud.mooo.com:~/setup_files/build/kaggle_facial_keypoints/$F .
# done

./test.py
./single_hidden_layer.py
./plot_net1_training_loss.py
./convolutions.py
./plot_net2_training_loss.py
./convolutions_flip.py
./plot_net3_training_loss.py
