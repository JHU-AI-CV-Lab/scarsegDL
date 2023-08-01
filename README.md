# A 3D Convolutional Neural Network Model for Scar Segmentation 

This repository contains weights and models for 3 different Scar Segmentation Models:
  1) U-Net: https://www.nature.com/articles/s41592-018-0261-2
  2) Cascaded U-Net: https://link.springer.com/chapter/10.1007/978-3-030-33226-6_12
  3) U-Net ++: https://arxiv.org/abs/1807.10165

## How to Use 

### Dependencies 

Implementation requires: 
  1) Tensorflow
  2) Keras

Data Preprocessing: Images should be compressed to unsigned 8-bit images of size 256x256x16 pixels. Images were shrunk with nearest-neighbor interpolation or re-sized with zero-padding. Images shoud be normalized with 0 mean pixel intensity and division by 255. The mean intensity ratio for normalization should be taken from scar_mean.npy. 

To use the code: Open the ipython notebook. Choose your model. Make sure you data noramlized data is in a numpy array. Input the path to your data and the path of the weights of interest. Be sure that the model you initialize matches the architecture of the correspodning weights. 



