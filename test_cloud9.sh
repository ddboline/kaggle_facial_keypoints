#!/bin/bash

for F in "training.zip test.zip IdLookupTable.csv SampleSubmission.csv";
do
    scp ubuntu@ddbolineinthecloud.mooo.com:~/$F .
done

