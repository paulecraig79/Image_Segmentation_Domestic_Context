from pycocotools.coco import COCO
import numpy as np
from PIL import Image
from data import CocoDataset
import torch
from torch.utils.data import Dataset, DataLoader
from model import UNET
import torchvision.transforms as transforms

ann_file = 'Ingredients.v11i.coco/train/_annotations.coco.json'
img_dir ="Ingredients.v11i.coco/train/"

coco = COCO(ann_file)

# Display COCO categories
cats = coco.loadCats(coco.getCatIds())



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
