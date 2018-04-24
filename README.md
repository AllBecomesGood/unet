# Implementation of deep learning framework -- Unet, using Keras

Inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

---

## Overview

### Data

MRI scans .nii (nifti format)

### Pre-processing

data.py reads the scans from folder TumourData and turns it into numpy arrays

### Model

![img/u-net-architecture.png](img/u-net-architecture.png)

This deep neural network is implemented with Keras functional API, which makes it extremely easy to experiment with different interesting architectures.

Output from the network is an image being the mask outlining where a tumour is predicted to be located in the MRI slices/images.

### Training

python3 unet.py

---

## How to use

### Dependencies

This tutorial depends on the following libraries:

* Tensorflow
* Keras 2

I use a virtual environment created with Anaconda and Python version 3.5.

You can also set up a virtual enviroment and run your code from there:
* Create virtual enviroment folder: ```conda create -n unet python=3.5```
* Activate your virtual enviroment: ```source activate unet```
* Install packages needed in enviroment: ```pip3 install -r req.txt```
* Check installed packages: ```pip3 freeze```

### Prepare the data

load the .nii MRI scans and save as numpy arrays (data.py)
### Define the model

* Check out ```get_unet()``` in ```unet.py``` to modify the model, optimizer and loss function.

### Train the model and generate masks for test images

* Run ```python3 unet.py``` to train the model.

After this script finishes, in ```imgs_mask_test.npy``` masks for corresponding images in ```imgs_test.npy```
should be generated. I suggest you examine these masks for getting further insight of your model's performance.

### Results

currently crap

## Folder structure

* ```./TumourData```: Contains folders with MRI scans in .nii format.
* ```./npydata```: Contains the ```.npy``` files generated from ```.tif``` images from our deform folder.
* ```./results```: Outputs from our unet are saved in this folder.

## From clone to your first results

* After cloning this project the first step is loading up the TumourData folder. ```TumourData/train_nii/mriScan1/flair.nii.gz``` and ```TumourData/test_nii/mriScan1/flair.nii.gz``` also a ```/mask_flair.nii.gz```

* ```python3 data.py``` to generate ```.npy``` files from your input data. This should produce 3 files in your ```./npydata``` folder.

* ```python3 unet.py``` to build your model. This should create a ```unet.hdf5``` file which is your model and can be loaded in your future programs. It also generates the results from the test dataset.





## About Keras.

Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
supports both convolutional networks and recurrent networks, as well as combinations of the two.
supports arbitrary connectivity schemes (including multi-input and multi-output training).
runs seamlessly on CPU and GPU.
Read the documentation [Keras.io](http://keras.io/)

Keras is compatible with: Python 2.7-3.5.
