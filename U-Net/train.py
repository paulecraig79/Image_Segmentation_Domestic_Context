from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from data import data_tr, data_val
from utils import *


#Models
from Models.UNet import UNET
from Models.Seg import SegNet
from Models.Unetplus import NestedUNet
from Models.DeepLabV3plus import DeepLab



#Parameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 10
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False




#Train function
def train(loader, model,optimizer, loss_fn, scaler):

    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        data = data.to(torch.float16)
        targets = targets.float().to(device=DEVICE)




        #forward
        with torch.cuda.amp.autocast():
            predictions = model(data)

            target_resized = F.interpolate(targets, size=predictions.size()[2:], mode='nearest')
            loss = loss_fn(predictions, target_resized)

        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #update tqdm loop
        loop.set_postfix(loss=loss.item())



def main():
    print(DEVICE)
    model = UNET(in_channels=3,out_channels=1).to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    #Import training and validation data
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


if __name__ == '__main__':
    main()