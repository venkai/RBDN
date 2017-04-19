#!/bin/bash

# This will open an interactive matlab session after loading caffe dependencies.
# You can then navigate to any of the 3 experiment folders from within matlab:
# "relighting/denoising/colorization" and run the inference script "get_pred.m"
# Each experiment folder EXP has its own "get_pred.m", "test.prototxt"
# EXP/get_pred.m saves results in EXP/results (not tracked by git)


# LOAD CAFFE DEPENDENCIES TO LD_LIBRARY_PATH (this also sets env-variable MATLAB_DIR)
source ./caffe/load_caffe_dependencies.sh

: "${MATLAB_DIR:?needs to be set in \"./caffe/load_caffe_dependencies.sh\".
This is the same path you specified in caffe\'s Makefile.config 
Please see \"./caffe/load_caffe_dependencies.sh.example\" for reference.}"
echo $MATLAB_DIR

# If you don't want an interactive matlab session, you can uncomment the line below
# and replace run_demo.m with the inference script of your choice.

# ${MATLAB_DIR}/bin/matlab -nodisplay -nosplash -r "run('./run_demo.m'); exit"

${MATLAB_DIR}/bin/matlab
