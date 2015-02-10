#!/bin/bash

for F in "training.zip test.zip IdLookupTable.csv SampleSubmission.csv";
do
    scp ubuntu@ddbolineinthecloud.mooo.com:~/setup_files/build/kaggle_facial_keypoints/$F .
done

