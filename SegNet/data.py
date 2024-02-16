import numpy as np
import json
from pycocotools.coco import COCO
import cv2
from matplotlib import pyplot as plt

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


ann_dir = "Ingredients.v11i.coco/train/_annotations.coco.json"
images_folder_path = "Ingredients.v11i.coco/train"


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
    print(img_id)
    images.append(image)
    masks.append(mask)

plt.figure(figsize=(18, 6))
for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.axis("off")
    plt.imshow(images[i])

    plt.subplot(2, 10, i + 11)
    plt.axis("off")
    plt.imshow(masks[i], cmap='gray')

plt.show()



