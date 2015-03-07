#!/bin/bash

for F in training.zip test.zip IdLookupTable.csv SampleSubmission.csv;
do
    scp ubuntu@ddbolineinthecloud.mooo.com:~/setup_files/build/kaggle_facial_keypoints/$F .
done

for F in training.zip test.zip;
do
    unzip -x $F;
done

rm *.pickle *.png

touch output.out
touch output.err
./test.py >> output.out 2>> output.err
./my_model.py $1 >> output.out 2>> output.err

# if [ -z $1 ]; then
#     D=`date +%Y%m%d%H%M%S`
# else
#     D=$1_`date +%Y%m%d%H%M%S`
# fi
# ssh ddboline@ddbolineathome.mooo.com "mkdir -p ~/setup_files/build/kaggle_facial_keypoints/output_${D}"
# scp output.out output.err *.png *.pickle submission*.csv ddboline@ddbolineathome.mooo.com:~/setup_files/build/kaggle_facial_keypoints/output_${D}
# ssh ddboline@ddbolineathome.mooo.com "~/bin/send_to_gtalk DONE_${D}"
# sudo shutdown now
