#!/bin/bash

sudo modprobe nvidia-340
sudo modprobe nvidia-340-uvm

touch output.out
touch output.err
if [ -z $1 ]; then
    ./test.py >> output.out 2>> output.err
    ./net1_single_hidden_layer.py >> output.out 2>> output.err
    ./plot_net1_training_loss.py >> output.out 2>> output.err
    ./net2_convolutions.py >> output.out 2>> output.err
    ./plot_net2_training_loss.py >> output.out 2>> output.err
elif [ $1 = "2" ]; then
    ./net3_convolutions_flip.py >> output.out 2>> output.err
    ./plot_net3_training_loss.py >> output.out 2>> output.err
    ./net4_update_learning_rate.py >> output.out 2>> output.err
    ./plot_net3_training_loss.py >> output.out 2>> output.err
elif [ $1 = "3" ]; then
    ./net5_update_learning_rate_flipped.py >> output.out 2>> output.err
    ./plot_net3_training_loss.py >> output.out 2>> output.err
    ./net6_dropout.py >> output.out 2>> output.err
    ./plot_net3_training_loss.py >> output.out 2>> output.err
elif [ $1 = "4" ]; then
    ./net7_bigger.py >> output.out 2>> output.err
    ./plot_net3_training_loss.py >> output.out 2>> output.err
    ./net8_moar_epochs.py >> output.out 2>> output.err
    ./plot_net3_training_loss.py >> output.out 2>> output.err
fi

if [ -z $1 ]; then
    D=`date +%Y%m%d%H%M%S`
else
    D=$1_`date +%Y%m%d%H%M%S`
fi
ssh ddboline@ddbolineathome.mooo.com "mkdir -p ~/setup_files/build/kaggle_facial_keypoints/output_${D}"
scp output.out output.err *.png *.pickle ddboline@ddbolineathome.mooo.com:~/setup_files/build/kaggle_facial_keypoints/output_${D}
ssh ddboline@ddbolineathome.mooo.com "~/bin/send_to_gtalk DONE_${D}"
sudo shutdown now
