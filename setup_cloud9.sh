#!/bin/bash

sudo apt-get update
sudo apt-get install -y python-pip

sudo pip install -r requirements.txt
sudo pip install -r requirements-2.txt
