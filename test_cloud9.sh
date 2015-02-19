#!/bin/bash

touch output.out
if [ -z $1 ]; then
    ./test.py 2>&1 >> output.out
    ./net1_single_hidden_layer.py 2>&1 >> output.out
    ./plot_net1_training_loss.py 2>&1 >> output.out
    ./net2_convolutions.py 2>&1 >> output.out
    ./plot_net2_training_loss.py 2>&1 >> output.out
elif [ $1 = "2" ]; then
    ./net3_convolutions_flip.py 2>&1 >> output.out
    ./plot_net3_training_loss.py 2>&1 >> output.out
    ./net4_update_learning_rate.py 2>&1 >> output.out
    ./plot_net3_training_loss.py 2>&1 >> output.out
elif [ $1 = "3" ]; then
    ./net5_update_learning_rate_flipped.py 2>&1 >> output.out
    ./plot_net3_training_loss.py 2>&1 >> output.out
    ./net6_dropout.py 2>&1 >> output.out
    ./plot_net3_training_loss.py 2>&1 >> output.out
elif [ $1 = "4" ]; then
    ./net7_bigger.py 2>&1 >> output.out
    ./plot_net3_training_loss.py 2>&1 >> output.out
    ./net8_moar_epochs.py 2>&1 >> output.out
    ./plot_net3_training_loss.py 2>&1 >> output.out
fi

if [ -z $1 ]; then
    D=`date +%Y%m%d%H%M%S`
else
    D=$1_`date +%Y%m%d%H%M%S`
fi
ssh ddboline@ddbolineathome.mooo.com "mkdir -p ~/setup_files/build/kaggle_facial_keypoints/output_${D}"
scp output.out *.png *.pickle ddboline@ddbolineathome.mooo.com:~/setup_files/build/kaggle_facial_keypoints/output_${D}
ssh ddboline@ddbolineathome.mooo.com "~/bin/send_to_gtalk DONE_${D}"
sudo shutdown
