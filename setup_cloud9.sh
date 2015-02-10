#!/bin/bash

sudo apt-get update
sudo apt-get install -y python-pip
sudo apt-get install -y gcc g++ gfortran build-essential 
sudo apt-get install -y git wget linux-image-generic libopenblas-dev 
sudo apt-get install -y python-dev python-pip python-nose python-numpy python-scipy 
sudo apt-get install -y cython linux-image-extra-virtual

sudo pip install --upgrade theano

wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_6.5-14_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_6.5-14_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda

sudo pip install git+https://github.com/benanne/Lasagne
sudo pip install nolearn
sudo pip install git+https://github.com/lisa-lab/pylearn2.git

echo -e "\n[global]\nfloatX=float32\ndevice=gpu\n[mode]=FAST_RUN\n\n[nvcc]\nfastmath=True\n\n[cuda]\nroot=/usr/local/cuda" >> ~/.theanorc

cat > blacklist-nouveau.conf << EOL
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off
EOL

sudo mv blacklist-nouveau.conf /etc/modprobe.d/

echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf

sudo update-initramfs -u
sudo reboot

# cd /usr/local/cuda/samples/1_Utilities/deviceQuery
# sudo make
# sudo ./deviceQuery
