#!/bin/bash

# Pull most recent snapshot and prepare it for inference by computing global
# mean/variance for all the batch-normalization layers by passing training
# data through the network. If the most recent snapshot is in
# ./snapshot/trn_iter_[N].caffemodel, then the final inference-ready weights
# will be saved in ./tst_[N].caffemodel
# Training Data is expected in ../Data (can be a symlink)
# ImageList is expected in ../Data/imgset/train.txt 
# (relative to root ../Data)

# Choose most recent snapshot for inference
CAFFEDIR="../caffe_colorization"
CURR_SNAPSHOT=`ls ./snapshot/trn_iter_*.caffemodel -t | head -n 1`
F=${CURR_SNAPSHOT/"./snapshot/trn_iter_"/}
ITER=${F/".caffemodel"/}
ITER=${ITER/"*"/}
P_ITER=`printf "%08g\n" ${ITER}`

WEIGHTSA_TMP=./train_curr.caffemodel
WEIGHTSA_INF_TMP=./train_curr_inference.caffemodel
WEIGHTSA_INF=./tst_${P_ITER}.caffemodel

echo $WEIGHTSA
echo $WEIGHTSA_INF

# LOAD CAFFE DEPENDENCIES TO LD_LIBRARY_PATH
source ${CAFFEDIR}/load_caffe_dependencies.sh

rm -f $WEIGHTSA_TMP && cp -f $CURR_SNAPSHOT $WEIGHTSA_TMP

# ADJUST BATCH NORMALIZATION FOR INFERENCE
export PYTHONPATH=${CAFFEDIR}/resources:${PYTHONPATH}
python make_bn.py
python add_wt_classab.py

rm -f ${WEIGHTSA_TMP}
mv ${WEIGHTSA_INF_TMP} ${WEIGHTSA_INF} 
