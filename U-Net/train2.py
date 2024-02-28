from tqdm import tqdm
from pycocotools.coco import COCO
import numpy as np
from PIL import Image



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from data2 import data_tr, data_val
from model import UNET

from utils2 import *
import torchvision.transforms as transforms



#Parameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 10
NUM_CLASSES = 32
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False


def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (images, masks, category_ids) in enumerate(loader):
        # Transfer data to device if using GPU
        images, masks, category_ids = images.to(device=DEVICE), masks.to(device=DEVICE), category_ids.to(device=DEVICE)

        #targets = torch.squeeze(targets, dim=1)  # Assuming the singleton dimension is at index 1

        # targets = torch.squeeze(targets, dim=1)
        #forward
        with torch.cuda.amp.autocast():
            predictions = model(images)
            loss = loss_fn(predictions, category_ids)

        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #update tqdm loop
        loop.set_postfix(loss=loss.item())

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

def main():
    print(DEVICE)
    model = UNET(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader = data_tr
    val_loader = data_val
    #if first run set Load model to false
    #If checkpoint has been saved or in directory then set load model to true
    if LOAD_MODEL:
        load_checkpoint(torch.load("Unet checkpoint/checkpoint.pth.tar"), model)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train(train_loader,model,optimizer,loss_fn,scaler)


        #save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()

        }
        save_checkpoint(checkpoint)
        # #check accuracy
        # check_accuracy(val_loader,model,device=DEVICE)
        #print examples
        save_predictions_as_imgs(
            val_loader,model,folder="saved_images/", device=DEVICE
        )




if __name__ == '__main__':
    main()