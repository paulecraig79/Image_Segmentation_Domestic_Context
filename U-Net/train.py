from tqdm import tqdm
from pycocotools.coco import COCO
import numpy as np
from PIL import Image



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from data import data_tr, data_val
from model import UNET

from utils import *
import torchvision.transforms as transforms



#Parameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 3
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False


def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        data = data.to(torch.float16)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        targets = torch.squeeze(targets, dim=1)  # Assuming the singleton dimension is at index 1

        # targets = torch.squeeze(targets, dim=1)
        #forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #update tqdm loop
        loop.set_postfix(loss=loss.item())



def main():
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader = data_tr
    val_loader = data_val
    #if first run set Load model to false
    #If checkpoint has been saved or in directory then set load model to true
    if LOAD_MODEL:
        load_checkpoint(torch.load("checkpoint.pth.tar"),model)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train(train_loader,model,optimizer,loss_fn,scaler)


        #save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()

        }
        save_checkpoint(checkpoint)
        #check accuracy
        check_accuracy(val_loader,model,device=DEVICE)
        #print examples
        save_predictions_as_imgs(
            val_loader,model,folder="saved_images/", device=DEVICE
        )




if __name__ == '__main__':
    main()