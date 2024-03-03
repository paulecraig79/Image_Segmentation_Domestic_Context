import torch
import torch.nn as nn
from pycocotools.coco import COCO

from torch.utils.data import TensorDataset, DataLoader
from utils import save_predictions_as_imgs

import cv2
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch.nn.functional as F


#Models
from Models.UNet import UNET
from Models.Seg import SegNet
from Models.Unetplus import NestedUNet
from Models.DeepLabV3plus import DeepLab


#parameters
NUM_CLASSES = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_PATH = "Ingredients-11/test/"
THRESHOLD = 0.5
ANN_DIR = "Ingredients-11/test/_annotations.coco.json"
IMG_DIR ="Ingredients-11/test/"
IMAGE_HEIGHT, IMAGE_WIDTH = 1280, 768





def extend_image(img, channels=None):
    height, width = img.shape[0], img.shape[1]
    delta = IMAGE_WIDTH - width
    if channels:
        padding = np.zeros((height, int(delta / 2), channels), np.uint8)
    else:
        padding = np.zeros((height, int(delta / 2)), np.uint8)
    img = np.concatenate((padding, img, padding), axis=1)
    return img

def load_data(ann_dir,images_folder_path):

    coco = COCO(ann_dir)

    image_ids = coco.getImgIds(imgIds=coco.getImgIds())

    images = []
    masks = []

    for img_id in image_ids:
        img_info = coco.loadImgs(ids=img_id)[0]
        image_path = f"{images_folder_path}/{img_info['file_name']}"
        image = plt.imread(image_path)

        ann_ids = coco.getAnnIds(imgIds=img_info['id'])
        annotations = coco.loadAnns(ann_ids)


        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in annotations:
            if 'segmentation' not in ann or not ann['segmentation']:
                # Skip annotations with empty segmentations
                continue
            coco_mask = coco.annToMask(ann)
            mask = np.maximum(mask, coco_mask * ann['category_id'])

        image = extend_image(image, 3)
        mask = extend_image(mask)

        target_height = 448  # Example height
        target_width = 448
        resized_img = cv2.resize(image, (target_width, target_height))
        resized_mask = cv2.resize(mask, (target_width, target_height))

        resized_mask = (resized_mask > 0).astype(np.uint8)

        images.append(resized_img)
        masks.append(resized_mask)
        #print(img_id)
    return images,masks



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



model_Unet = UNET(in_channels=3, out_channels=1).to(device=DEVICE)
checkpoint_path_Unet = 'Unet checkpoint/checkpoint.pth.tar'

model_Segnet = SegNet().to(DEVICE)
checkpoint_path_Segnet = 'SegNet Checkpoint/checkpoint.pth.tar'

model_Deeplab = DeepLab(num_classes=1).to(device=DEVICE)
checkpoint_path_Deeplab = 'DeepLab/checkpoint.pth.tar'

model_UnetPlus = NestedUNet(input_channels=3, output_channels=1,num_classes=1).to(device=DEVICE)
checkpoint_path_UnetPLus = 'Unetplus/checkpoint.pth.tar'



checkpoint_unet = torch.load(checkpoint_path_Unet,map_location=DEVICE)
checkpoint_Segnet = torch.load(checkpoint_path_Segnet, map_location=DEVICE)
checkpoint_DeepLab = torch.load(checkpoint_path_Deeplab, map_location=DEVICE)
checkpoint_Unetplus = torch.load(checkpoint_path_UnetPLus, map_location=DEVICE)





model_Unet.load_state_dict(checkpoint_unet['state_dict'])
model_Segnet.load_state_dict(checkpoint_Segnet['state_dict'])
model_Deeplab.load_state_dict(checkpoint_DeepLab['state_dict'])
model_UnetPlus.load_state_dict(checkpoint_Unetplus['state_dict'])


test_images,Ground_truth = load_data(ANN_DIR,IMG_DIR)

test_images_np =np.array(test_images)
ground_truth_np = np.array(Ground_truth)


test_dl = DataLoader(TensorDataset(torch.tensor(np.rollaxis(test_images_np, 3, 1)), torch.tensor(ground_truth_np[:, np.newaxis])),
                     batch_size=1, shuffle=True)
print("\n Evaluations\n")

print("Number of test images:", len(test_dl))
print("\n")

print("U-Net")
print("==============")
save_predictions_as_imgs(
   test_dl, model_Unet, device=DEVICE
)
print("\n")

print("U-Net++")
print("==============")
save_predictions_as_imgs(
   test_dl, model_UnetPlus, device=DEVICE
)
print("\n")
print("SegNet")
print("==============")
save_predictions_as_imgs(
   test_dl, model_Segnet, device=DEVICE
)
print("\n")
print("DeepLabV3+")
print("==============")
save_predictions_as_imgs(
   test_dl, model_Deeplab, device=DEVICE
)