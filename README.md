# keras-openpose-reproduce

This is a keras implementation of [Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation).


## Acknowledgment
This repo is based upon [@anatolix](https://github.com/anatolix)'s repo [keras_Realtime_Multi-Person_Pose_Estimation](https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation), and [@michalfaber](https://github.com/michalfaber)'s repo [keras_Realtime_Multi-Person_Pose_Estimation](https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation)

## Prerequisites

  0. Keras and Tensorflow (tested on Linux machine)
  0. Python3
  0. GPU with at least `11GB` memory
  0. More than `250GB` of disk space free for training data


## Download COCO 2014 Dataset

Please download the COCO dataset and the official COCO evaluation API. Go to folder `dataset` and simply run the following commands:

    $ cd dataset
    $ ./step1_download_coco2014.sh
    $ ./step2_setup_coco_api.sh


## Prepare Training Data 

Before model training, we need to convert the images to the specific data format. We first generate the heatmaps, part affinity maps, and then convert them to HDF5 file format. Go to the folder `training`, and run the scripts:

    $ cd training
    $ python3 generate_masks_coco2014.py
    $ python3 generate_hdf5_coco2014.py

After the data conversion, you may find the training files `train_dataset_2014.h5` and `val_dataset_2014.h5`. Note that these files are about `182GB` and `3.87 GB`, respectively.

## Training

Simply go to folder `training` and run the training script:

    $ cd training
    $ python3 train_pose.py


## Evaluation on COCO Keypoint Datasets

Evaluation codes and instructions coming soon!


## Summary of the Evaluation

We empirically trained the model with `100 epochs` and achieved comparable performance to the results reported in the original paper. We also compared with the original implementation which is [online avialable](https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation#converting-caffe-model-to-keras-model). Note the validation set `COCO2014-Val-1K` can be found in the [original caffe implementation](https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose/blob/master/image_info_val2014_1k.txt).


|     Method      |       Validation      |     AP    | 
|-----------------|:---------------------:|:---------:|
|  [Openpose paper](https://arxiv.org/pdf/1611.08050.pdf) |  COCO2014-Val-1k   |    58.4   | 
|  [Openpose model](https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation#converting-caffe-model-to-keras-model) |    COCO2014-Val-1k    |    56.3   |     
|    This repo    |    COCO2014-Val-1k    |    58.9   |


We also evaluated the performance on the full COCO2014 validation set.

|     Method      |       Validation      |     AP    | 
|-----------------|:---------------------:|:---------:|  
|  [Openpose model](https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation#converting-caffe-model-to-keras-model) |      COCO2014-Val     |    58.9   |    
|    This repo    |      COCO2014-Val     |    59.0   |   



