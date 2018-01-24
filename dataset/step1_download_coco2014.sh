#!/usr/bin/env bash

echo "Downloading train2014..."
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
rm train2014.zip 

echo "Downloading val2014..."
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
rm val2014.zip

echo "Downloading annotations..."
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
rm annotations_trainval2014.zip

git https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
sudo python setup.py install

