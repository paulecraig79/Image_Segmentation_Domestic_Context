# Image segmentation on domestic objects

![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white) ![Tensor Flow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

This repository contains the python project for my honours project at Edinburgh Napier University. This code was used for testing and researching binary image segmentation using an ingredients dataset being developed by Edinburgh Napier University.
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

The code for the data loading is found in the data.py file. To access and use the COCO dataset, I used the python package pycoco which makes it easier to access images and annotations. Pycoco uses the annotations to generate masks for the segmentations. The images and masks are then loaded into a dataloader which can then be used by the models to train or predict.

# Model Training

For training, each model is imported and initialised with in-channels and out-channels. The loss function which I have implemented for training is Binary Cross Entropy with logits loss and the optimzer which has been implemented is Adam. The validation dataset is used to track the models performance during training. 

# Model Predictions

The checkpoints which have been trained from model training are loaded into each model for predicting. The models will predict every image in the test dataset then output the average pixel accuracy, average precision, average recall, average F1-score, average time taken per image and total time taken.
