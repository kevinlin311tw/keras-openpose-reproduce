# keras-openpose-reproduce

This is a keras implementation of [Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation).


## Prerequisites

  0. Keras and Tensorflow (tested on Linux machine)
  0. Python3
  0. GPU with at least `11GB` memory
  0. More than `250GB` of disk space for training data


## Download COCO 2014 Dataset

Please download the COCO dataset and the official COCO evaluation API. Go to folder `dataset` and simply run the following commands:

    $ cd dataset
    $ ./step1_download_coco2014.sh
    $ ./step2_setup_coco_api.sh


## Prepare Training Data 

Before model training, we convert the images to the specific data format for efficient training. We generate the heatmaps, part affinity maps, and then convert them to HDF5 files. Go to the folder `training`, and run the scripts. The process takes around 2 hours.

    $ cd training
    $ python3 generate_masks_coco2014.py
    $ python3 generate_hdf5_coco2014.py

After the pre-processing, you will find the files `train_dataset_2014.h5` and `val_dataset_2014.h5`. The files are about `182GB` and `3.8GB`, respectively.

## Training

Simply go to folder `training` and run the training script:

    $ cd training
    $ python3 train_pose.py


## Evaluation on COCO Keypoint Datasets

Evaluation codes and instructions coming soon!


## Evaluation Summary

We empirically trained the model for `100 epochs (2 weeks)` and achieved comparable performance to the results reported in the original paper. We also compared with the original implementation which is [online avialable](https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation#converting-caffe-model-to-keras-model). Note that the validation list `COCO2014-Val-1K` is provided by [the official Openpose](https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose/blob/master/image_info_val2014_1k.txt).


|     Method      |      Validation       |     AP    | 
|-----------------|:---------------------:|:---------:|
|  [Openpose paper](https://arxiv.org/pdf/1611.08050.pdf) |  COCO2014-Val-1k   |    58.4   | 
|  [Openpose model](https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation#converting-caffe-model-to-keras-model) |    COCO2014-Val-1k    |    56.3   |     
|    This repo    |    COCO2014-Val-1k    |    58.9   |


We also evaluated the performance on the full COCO2014 validation set.

|     Method      |      Validation       |     AP    | 
|-----------------|:---------------------:|:---------:|  
|  [Openpose model](https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation#converting-caffe-model-to-keras-model) |      COCO2014-Val     |    58.9   |    
|    This repo    |      COCO2014-Val     |    59.0   |   


You may find our trained model at [Dropbox](https://www.dropbox.com/s/76b3r8rj82wicik/weights.0100.h5?dl=0)
You may also find our prediction results on COCO2014 validation (json format w/o images) [Dropbox](https://www.dropbox.com/s/snaot6xva6ei5ge/val2014_ours_result.json?dl=0)


## Acknowledgment
This repo is based upon [@anatolix](https://github.com/anatolix)'s repo [keras_Realtime_Multi-Person_Pose_Estimation](https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation), and [@michalfaber](https://github.com/michalfaber)'s repo [keras_Realtime_Multi-Person_Pose_Estimation](https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation)


## Citation

Please cite the paper in your publications if it helps your research:

    @inproceedings{cao2017realtime,
      author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
      year = {2017}
      }
    
    @inproceedings{wei2016cpm,
      author = {Shih-En Wei and Varun Ramakrishna and Takeo Kanade and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Convolutional pose machines},
      year = {2016}
      }
