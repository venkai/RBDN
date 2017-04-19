#!/bin/bash

# Start training an RBDN model for relighting. You can pause training at any moment
# with Ctrl+C and most recent snapshot will be saved in
# ./snapshot/trn_iter_[*].solverstate
# Running ./start_train.sh again will automatically resume from that snapshot.
# Training Data is expected in ../Data/multipie (can be a symlink)
# ImageList is expected in ../Data/multipie/imgset/train_relighting.txt 
# (relative to root ../Data/multipie)

CAFFEDIR="../caffe"
LOGDIR=./training_log
CAFFE=${CAFFEDIR}/build/tools/caffe
SOLVER=./solver.prototxt

# LOAD CAFFE DEPENDENCIES TO LD_LIBRARY_PATH
source ${CAFFEDIR}/load_caffe_dependencies.sh

mkdir -p $LOGDIR
mkdir -p snapshot

# Check if snapshot exists; if not train from scratch.
if [ "$(ls -A ./snapshot/)" ]; then
    CURR_SNAPSHOT=`ls ./snapshot/*solverstate -t | head -n 1`
    echo "Resuming training from snapshot ${CURR_SNAPSHOT}"
    echo "If you want to train from scratch, delete everything in ./snapshot"
    GLOG_log_dir=$LOGDIR $CAFFE train -snapshot $CURR_SNAPSHOT -solver $SOLVER -gpu 0
else
    GLOG_log_dir=$LOGDIR $CAFFE train -solver $SOLVER -gpu 0
fi
