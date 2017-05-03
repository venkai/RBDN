#!/bin/bash

# This will fetch caffe, pretrained models, inference data and setup directory
# structure & symlinks for all of the training/inference scripts.
# (Nothing that is created here will be tracked by git)

##############################################################################
# ---------------------- Fetch caffe for RBDN --------------------------------
# (caffe folders will not be tracked by git)
echo "fetch caffe for RBDN"
git clone -b rec_conv --single-branch https://github.com/venkai/caffe.git
# You should compile this caffe library with your own Makefile.config file)


##############################################################################
# ----------------------- Set up inference data ------------------------------
# (data folders will not be tracked by git)
echo "fetch inference data for denoising, relighting, colorization experiments"
mkdir -p data/training && cd data
wget --no-check-certificate http://www.umiacs.umd.edu/~venkai/rbdn/data/inference.tar.gz
tar xvzf inference.tar.gz && rm -f inference.tar.gz
cd ..


##############################################################################
# ----------------------- Set up pretrained models ---------------------------
# (models folder will not be tracked by git)
echo "fetch pretrained models for denoising, relighting, colorization experiments"
wget --no-check-certificate http://www.umiacs.umd.edu/~venkai/rbdn/models/models.tar.gz
tar xvzf models.tar.gz && rm -f models.tar.gz


##############################################################################
# ----------- Set up directory structure and symlinks ------------------------
echo "set up directory structure and symlinks"
for F in ./inference/*/
do
  rm -rf ${F}results && mkdir -p ${F}results
done

for F in ./training/*/
do
  rm -rf ${F}snapshot && mkdir -p ${F}snapshot
  rm -rf ${F}training_log && mkdir -p ${F}training_log
done

rm -f ./inference/caffe && ln -sf ../caffe ./inference/caffe
rm -f ./training/caffe && ln -sf ../caffe ./training/caffe

rm -f ./inference/Data && ln -sf ../data/inference ./inference/Data
rm -f ./training/Data && ln -sf ../data/training ./training/Data

