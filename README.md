# RBDN (Recursively Branched Deconvolutional Network)
Code for paper [**Generalized Deep Image to Image Regression**](https://arxiv.org/abs/1612.03268).

## Usage
- **Setup:** After cloning this repository, run `./setup.sh`. This will fetch caffe, download pretrained caffe models for all 3 experiments (denoising/relighting/colorization) and inference data, as well as set up the directory structure and symbolic links for all the training/inference scripts.
- **Install Caffe:** Note that `setup.sh` pulls 2 different branches of caffe into 2 separate directories: namely *caffe_colorization* used for colorization and *caffe_rbdn* which is used for both denoising/relighting experiments. Both these branches will eventually be merged with the master branch in *BVLC/caffe*. However for now, you would have to separately install both these caffe versions if you want to perform all 3 experiments.

- **Data:** 

  - Inference data is automatically downloaded by `setup.sh`.
  
  - Training data/imglist for relighting experiment can be downloaded from either of these mirrors: [*mirror1*](https://drive.google.com/file/d/0B3PoH3B39H2reWxzd3VDZDFVSlE/view?usp=sharing)/[*mirror2*](https://drive.google.com/file/d/0B4c0dYlyY36JY3EwWUo3Y2MtNm8/view?usp=sharing)
  This downloads the file *multipie.tar.gz*. Copy it to ./data/training and run `tar xvzf multipie.tar.gz && rm multipie.tar.gz`

  - Denoising/colorization experiments use the same training data/imglist: which is every single *unresized* train & validation image from both ImageNet ILSVRC2012 and MS-COCO2014 whose smallest spatial dimension is greater than 128 (~1.7 million images in total). You can simply download these datasets from their respective sources and place/symlink them within `./data/training/` without any preprocessing whatsoever. Place the appropriate imglist in `./data/training/imgset/train.txt` with the image-paths in `train.txt` being relative to `./data/training` 

  - Note that data folders are not tracked by git.

- **Inference:** Each experiment (denoising/relighting/colorization) has its own folder in `./inference` that contains an experiment specific MATLAB inference script `get_pred.m` which uses the matcaffe interface to evaluate pretrained models in `./models`. The script `./inference/run_matcaffe.sh` can be used to load caffe dependencies to `LD_LIBRARY_PATH` and then start MATLAB interactively.

- **Training:** Each experiment (denoising/relighting/colorization) has its own folder in `./training` that contain 2 key experiment specific scripts:
  - `start_train.sh`: This starts training an RBDN model, either from scratch or from the most recent snapshot in the `snapshot` directory. You can pause training at any moment with `Ctrl+C` and most recent snapshot will be saved in `./snapshot/trn_iter_[*].solverstate`. Running `./start_train.sh` again will automatically resume from that snapshot. 
  - `run_bn.sh`: This takes the most recent snapshot in `./snapshot` and prepares it for inference by passing training data through the network and computing global mean/variance for all the *batch-normalization* layers in the network. The resulting inference-ready model is saved as `./tst_[ITER].caffemodel`