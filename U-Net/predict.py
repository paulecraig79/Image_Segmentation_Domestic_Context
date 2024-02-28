import argparse
import logging
import os
import torch
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from torch.utils.data import TensorDataset, DataLoader
from utils import save_predictions_as_imgs
from model import UNET
from Seg import SegNet
import cv2

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
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
def preprocess(image,scale_factor):
    w,h = image.size
    newW,newH = int(w*scale_factor), int(h*scale_factor)
    image = image.resize((newW, newH),Image.BICUBIC)
    img = np.asarray(image)

    if img.ndim == 2:
        img = img[np.newaxis,...]
    else:
        img = img.transpose((2,0,1))
    if(img>1).any():
        img = img/255.0

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

def predict_and_plot(model, image_path, device, threshold=0.5):
    # Set the model to evaluation mode
    model.eval()

    # Define image transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert NumPy array to PIL Image
        transforms.Resize((450, 450)),  # Resize image to the required input size
        transforms.ToTensor(),  # Convert PIL Image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])
    original_image = Image.open(image_path)
    # Load and preprocess the image
    image = np.asarray(Image.open(image_path).convert("RGB"))  # Load image as NumPy array
    image = transform(image).unsqueeze(0)  # Add batch dimension  # Add batch dimension

      # Add batch dimension

    # Move the image to the device
    image = image.to(device)

    # Forward pass through the model
    with torch.no_grad():
        preds = model(image)
        preds = F.interpolate(preds, size=(original_image.size[1], original_image.size[0]), mode='bilinear', align_corners=False)
        mask = torch.sigmoid(preds) > threshold

    # Convert predictions to numpy array and squeeze to remove batch dimension
    mask = mask.cpu().squeeze().numpy().astype(np.uint8)

    # Plot the original image and the predicted mask
    plot_img_and_mask(original_image, mask)


def plot_img_and_mask(img, mask):
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title('Mask (Foreground)')
    ax[1].imshow(mask == 1, cmap='binary')  # Plot only the foreground class (class 1)
    plt.xticks([]), plt.yticks([])
    plt.show()



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Input = input("[1]U-NET [2]SegNet\n")

if Input=="1":
    model = UNET(in_channels=3, out_channels=1).to(device=DEVICE)
    checkpoint_path = 'Unet checkpoint/checkpoint.pth.tar'
elif Input=="2":
    model = SegNet().to(DEVICE)
    checkpoint_path = 'SegNet Checkpoint/checkpoint.pth.tar'


try:
    checkpoint = torch.load(checkpoint_path,map_location=DEVICE)
    if checkpoint is None:
        raise ValueError("Checkpoint is None.")
    print("Checkpoint loaded successfully.")
except FileNotFoundError:
    print(f"Checkpoint file '{checkpoint_path}' not found.")
except Exception as e:
    print(f"Error loading checkpoint: {e}")

print(checkpoint.keys())


model.load_state_dict(checkpoint['state_dict'])


test_images,Ground_truth = load_data(ANN_DIR,IMG_DIR)

test_images_np =np.array(test_images)
ground_truth_np = np.array(Ground_truth)


test_dl = DataLoader(TensorDataset(torch.tensor(np.rollaxis(test_images_np, 3, 1)), torch.tensor(ground_truth_np[:, np.newaxis])),
                     batch_size=1, shuffle=True)

save_predictions_as_imgs(
   test_dl, model, device=DEVICE
)