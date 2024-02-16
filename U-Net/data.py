from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from model import UNET
import torchvision.transforms.functional as TF
# Specify path to annotations file
ann_file = 'Ingredients.v11i.coco/train/_annotations.coco.json'
img_dir ="Ingredients.v11i.coco/train/"

# Initialize COCO object
coco = COCO(ann_file)

# Display COCO categories
cats = coco.loadCats(coco.getCatIds())

class CocoDataset(Dataset):
    def __init__(self,coco,transform=None,target_transform=None, resize_shape=(256, 256)):
        self.coco = coco
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        if isinstance(resize_shape, int):
            self.resize_shape = (resize_shape, resize_shape)
        elif len(resize_shape) == 2:
            self.resize_shape = resize_shape
        else:
            raise ValueError("resize_shape must be an int or a tuple/list with two elements")

    def __len__(self):
        return len(self.coco.getImgIds())

    def __getitem__(self, idx):
        img_info = self.coco.loadImgs(idx)[0]
        img_id = img_info['id']
        img_path = img_dir + img_info['file_name']
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)  # Apply transformations

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        masks = []
        for ann_id in ann_ids:
            ann = self.coco.loadAnns(ann_id)[0]
            mask = self.coco.annToMask(ann)
            mask_pil = TF.to_tensor(mask)  # Convert mask to PIL Image
            mask_pil = TF.resize(mask_pil, self.resize_shape[::-1])  # Resize mask
             # Convert mask to tensor
            masks.append(mask_pil)

        # Resize masks to the smallest shape
        min_shape = min(mask.shape for mask in masks)
        masks = [TF.resize(mask, min_shape[::-1]) for mask in masks]

        masks = torch.stack(masks)  # Stack masks into a tensor

        if self.target_transform:
            masks = self.target_transform(masks)

        return img, masks



transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])


dataset = CocoDataset(coco, transform=transform)
data_loader = DataLoader(dataset, batch_size=8,shuffle=True)

model = UNET()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),0.001)
for epoch in range(10):
    model.train()
    for images,masks in data_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,masks)
        loss.backward()
        optimizer.step()
