# RBDN (Recursively Branched Deconvolutional Network)
![placeholder](https://github.com/venkai/RBDN/blob/gh-pages/assets/placeholder.png)
**RBDN** is an architecture for [**Generalized Deep Image to Image Regression**](https://arxiv.org/abs/1612.03268) which features 
* a memory-efficient recursive branched scheme with extensive parameter sharing that computes an early learnable multi-context representation of the input, 
* end-to-end preservation of local correspondences from input to output and 
* ability to choose context-vs-locality based on task as well as apply a per-pixel multi-context non-linearity. 

## Architecture
![pipeline](https://github.com/venkai/RBDN/blob/gh-pages/assets/pipeline.png)
**RBDN** gives state-of-the-art performance on 3 diverse *image-to-image regression* tasks: **Denoising**, **Relighting**, **Colorization**.

## Installation & Usage

- **Clone:** Run `git clone -b master --single-branch  https://github.com/venkai/RBDN.git`

- **Setup:** Go to repository `cd RBDN` and run `./setup.sh`. This will fetch caffe, download pretrained caffe models for all 3 experiments (**denoising/relighting/colorization**) and inference data, as well as set up the directory structure and symbolic links for all the training/inference scripts.

- **Install Caffe:** Note that `setup.sh` pulls 2 different branches of caffe into 2 separate directories: namely `caffe_colorization` used for **colorization** and `caffe_rbdn` which is used for both **denoising/relighting** experiments. Both these branches will eventually be merged with the master branch in [**venkai/caffe**](https://github.com/venkai/caffe). However for now, you would have to separately install both these caffe versions if you want to perform all 3 experiments.

- **Data:** 

  - Inference data is automatically downloaded by `setup.sh`.
  
  - Training data/imglist for **relighting** experiment can be downloaded from either of these mirrors: [**[1]**](https://drive.google.com/file/d/0B3PoH3B39H2reWxzd3VDZDFVSlE/view?usp=sharing)/[**[2]**](https://drive.google.com/file/d/0B4c0dYlyY36JY3EwWUo3Y2MtNm8/view?usp=sharing)  
  This downloads the file `multipie.tar.gz`. Move it to `./data/training` and run `tar xvzf multipie.tar.gz && rm multipie.tar.gz`

  - **Denoising/colorization** experiments use the same training data/imglist: which is every single *unresized* train & validation image from both [**ImageNet ILSVRC2012**](https://arxiv.org/abs/1409.0575) and [**MS-COCO2014**](http://mscoco.org/) whose smallest spatial dimension is greater than **128** (**~1.7 million** images in total). You can simply download these datasets from their respective sources and place/symlink them within `./data/training/` without any preprocessing whatsoever. Place the appropriate imglist in `./data/training/imgset/train.txt` with the image-paths in `train.txt` being relative to `./data/training` 

  - Note that data folders are not tracked by git.

- **Inference:** Each experiment (**denoising/relighting/colorization**) has its own folder in `./inference` that contains an experiment specific MATLAB inference script `get_pred.m` which uses the [**Matcaffe**](http://caffe.berkeleyvision.org/tutorial/interfaces.html#matlab) interface to evaluate pretrained models in `./models`. The script `./inference/run_matcaffe.sh` can be used to load caffe dependencies to `LD_LIBRARY_PATH` and then start MATLAB interactively.

- **Training:** Each experiment (**denoising/relighting/colorization**) has its own folder in `./training` that contain 2 key experiment specific scripts:
  - `start_train.sh`: This starts training an **RBDN** model, either from scratch or from the most recent snapshot in the `snapshot` directory. You can pause training at any moment with `Ctrl+C` and most recent snapshot will be saved in `./snapshot/trn_iter_[*].solverstate`. Running `./start_train.sh` again will automatically resume from that snapshot. 
  - `run_bn.sh`: This takes the most recent snapshot in `./snapshot` and prepares it for inference by passing training data through the network and computing global mean/variance for all the *batch-normalization* layers in the network. The resulting inference-ready model is saved as `./tst_[ITER].caffemodel`, where `ITER` is the iteration corresponding to the most recent snapshot.

# License & Citation
**RBDN** is released under a variant of the [BSD 2-Clause license](https://github.com/venkai/RBDN/blob/master/LICENSE). 

If you find **RBDN** useful in your research, please consider citing our paper:

```
@article{santhanam2016generalized,
  title={Generalized Deep Image to Image Regression},
  author={Santhanam, Venkataraman and Morariu, Vlad I and Davis, Larry S},
  journal={arXiv preprint arXiv:1612.03268},
  year={2016}
}
```
  
# Acknowledgments
* We would like to thank [Yangqing Jia](http://daggerfs.com/), [Evan Shelhamer](http://imaginarynumber.net/) and the [**BVLC/BAIR**](http://bair.berkeley.edu/) team for creating & maintaining [**caffe**](http://caffe.berkeleyvision.org/), [Richard Zhang](https://richzhang.github.io/) for [colorization layers in caffe](https://github.com/richzhang/colorization) and [Hyeonwoo Noh](http://cvlab.postech.ac.kr/~hyeonwoonoh/), [Seunghoon Hong](http://cvlab.postech.ac.kr/~maga33/), [Dmytro Mishkin](https://github.com/ducha-aiki) for several useful caffe layers, all of which were instrumental in creating **RBDN**. 

* This research is based upon work supported by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via IARPA R&D Contract No. 2014-14071600012. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of the ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon.