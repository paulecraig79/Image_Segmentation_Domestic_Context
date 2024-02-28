import numpy as np
import json
import gc
from torchvision import transforms
import torch
from pycocotools.coco import COCO
import cv2
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset,Dataset

IMAGE_HEIGHT, IMAGE_WIDTH = 1280, 768


class CustomDataset(Dataset):
    def __init__(self, images, masks, categories_per_mask, transform=None):
        self.images = images
        self.masks = masks
        self.categories_per_mask = categories_per_mask
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        category_id = self.categories_per_mask[idx]

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask, category_id


def extend_image(img, channels=None):
    height, width = img.shape[0], img.shape[1]
    delta = IMAGE_WIDTH - width
    if channels:
        padding = np.zeros((height, int(delta / 2), channels), np.uint8)
    else:
        padding = np.zeros((height, int(delta / 2)), np.uint8)
    img = np.concatenate((padding, img, padding), axis=1)
    return img


ann_dir = "Ingredients-11/train/_annotations.coco.json"
images_folder_path = "Ingredients-11/train/"

val_ann_dir = "Ingredients-11/valid/_annotations.coco.json"
val_images_folder_path = "Ingredients-11/valid/"






def load_data(ann_dir,images_folder_path):

    coco = COCO(ann_dir)
    categories = coco.getCatIds(catNms=[''])
    image_ids = coco.getImgIds(catIds=categories)

    images = []
    masks = []
    categories_per_mask = []

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
        categories_per_mask.append(ann['category_id'])
        #print(img_id)
    return images, masks, categories_per_mask


#training images and masks
tr_images,tr_masks,tr_categories = load_data(ann_dir,images_folder_path)

#Validation images and maks
val_images,val_masks,val_categories = load_data(val_ann_dir,val_images_folder_path)

#plt.figure(figsize=(18, 6))
# for i in range(5):
#     plt.subplot(2, 10, i + 1)
#     plt.axis("off")
#     plt.imshow(val_images[i])
#     plt.title('Image')
#
#     plt.subplot(2, 10, i + 6)  # Adjust the subplot position for category ID
#     plt.axis("off")
#     plt.text(0.5, 0.5, 'Category ID: ' + str(val_categories_name[i]), ha='center', va='center')
#
#     plt.subplot(2, 10, i + 11)
#     plt.axis("off")
#     plt.imshow(val_masks[i], cmap='gray')
#     plt.title('Mask')
#
#     plt.subplot(2, 10, i + 16)  # Adjust the subplot position for category ID
#     plt.axis("off")
#     plt.text(0.5, 0.5, 'Category ID: ' + str(val_categories_name[i]), ha='center', va='center')
#
# plt.show()


Train_dataset = CustomDataset(tr_images, tr_masks, tr_categories, transform=transforms.ToTensor())
val_dataset = CustomDataset(val_images, val_masks,val_categories,transform=transforms.ToTensor())


#Convert images and masks into numpy array
images_tr_np = np.array(tr_images)
masks_tr_np = np.array(tr_masks)
images_val_np = np.array(val_images)
masks_val_np = np.array(val_masks)


batch_size = 5


data_tr = torch.utils.data.DataLoader(Train_dataset, batch_size=batch_size, shuffle=True)
data_val = torch.utils.data.DataLoader(Train_dataset, batch_size=batch_size, shuffle=False)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
