#!/bin/bash

sudo modprobe nvidia-340
sudo modprobe nvidia-340-uvm

export PATH="/usr/local/cuda-6.5/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-6.5/lib64"

rm *.pickle *.png

touch output.out
touch output.err
./test.py >> output.out 2>> output.err
./my_model.py >> output.out 2>> output.err

if [ -z $1 ]; then
    D=`date +%Y%m%d%H%M%S`
else
    D=$1_`date +%Y%m%d%H%M%S`
fi
ssh ddboline@ddbolineathome.mooo.com "mkdir -p ~/setup_files/build/kaggle_facial_keypoints/output_${D}"
scp output.out output.err *.png *.pickle ddboline@ddbolineathome.mooo.com:~/setup_files/build/kaggle_facial_keypoints/output_${D}
ssh ddboline@ddbolineathome.mooo.com "~/bin/send_to_gtalk DONE_${D}"
# sudo shutdown now
