# Image segmentation on domestic objects

![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white) ![Tensor Flow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

This repository contains the python project for my honours project at Edinburgh Napier University. This code was used for testing and researching an ingredients dataset being developed by Edinburgh Napier University.
The code uses four models :

* U-Net
* U-Net++ / Nested U-Net
* SegNet
* DeepLabV3+

The code uses COCO datasets, if another type of dataset was to be used then the code would need to be altered.

The code can be split up into 3 sections:

* Data loading
* Model Training
* Model Predictions


# Data loading

The code for the data loading is found in the data.py file. To access and use the COCO dataset, I used the python package pycoco which makes it easier to access images and annotations. 
