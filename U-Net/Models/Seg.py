
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch
import numpy as np
import torch.nn  as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
from time import time
import seaborn as sns
import gc
from IPython.display import clear_output


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
class SegNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        # Each enc_conv/dec_conv block should look like this:
        # nn.Sequential(
        #     nn.Conv2d(...),
        #     ... (2 or 3 conv layers with relu and batchnorm),
        # )
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # 256 -> 128
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.pool1 =  nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.pool2 =  nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.bottleneck_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.boottleneck_upsample = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # decoder (upsampling)
        self.upsample0 =  nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv0 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.upsample1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.upsample2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.upsample3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # encoder
        x = x.float()
        e0 = self.enc_conv0(x)
        x, indices0 = self.pool0(e0)
        e1 = self.enc_conv1(x)
        x, indices1 = self.pool1(e1)
        e2 = self.enc_conv2(x)
        x, indices2 = self.pool2(e2)
        e3 = self.enc_conv3(x)
        x, indices3 = self.pool3(e3)

        gc.collect()

        # bottleneck
        b = self.bottleneck_conv(x)
        x, bottleneck_indices = self.bottleneck_pool(b)
        x = self.boottleneck_upsample(x, bottleneck_indices)
        dec_b = self.dec_bottleneck(x)

        gc.collect()

        # decoder
        x = self.upsample0(dec_b, indices3)
        d0 = self.dec_conv0(x)
        x = self.upsample1(d0, indices2)
        d1 = self.dec_conv1(x)
        x = self.upsample2(d1, indices1)
        d2 = self.dec_conv2(x)
        x = self.upsample3(d2, indices0)
        d3 = self.dec_conv3(x)

        return d3

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, limit=0.5):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = torch.sigmoid(outputs.squeeze(1)) > limit  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1).byte()
    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresholds

    return thresholded
def bce(y_pred, y_real):
    return y_pred - y_real * y_pred + torch.log(1 + torch.exp(-y_pred))

def bce_loss(y_pred, y_real):
    return torch.mean(bce(y_pred, y_real))

def score_model(model, metric, data, threshold=0.5):
    model.eval()  # testing mode
    scores = 0
    total_size = 0
    with torch.set_grad_enabled(False):
        for X_batch, Y_label in data:
            total_size += len(X_batch)
            Y_pred = model(X_batch.to(device))
            scores += metric(Y_pred, Y_label.to(device), limit=threshold).mean().item() * len(X_batch)

    return scores / total_size



def analyze_history(loss_history, score_history):
    sns.set(style='whitegrid', font_scale=1.5)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.figure(figsize=(12, 6))

    train_loss = loss_history[:, 0]
    val_loss = loss_history[:, 1]
    ax1.plot(train_loss, marker='o', label='train')
    ax1.plot(val_loss, marker='o', label='val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # train_score = score_history[:, 0]
    # val_score = score_history[:, 1]
    val_score = score_history
    # ax2.plot(train_score, marker='o', label='train')
    ax2.plot(val_score, marker='o', label='val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()

    plt.show()

def fit_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    processed_data = 0

    tic = time()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        processed_data += inputs.size(0)

        del inputs
        del outputs
        del labels
        torch.cuda.empty_cache()
        gc.collect()

    toc = time()

    print("Fit epoch time: ", toc - tic)
    train_loss = running_loss / processed_data
    return train_loss

def eval_epoch(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    processed_size = 0
    X_val, Y_val  = next(iter(val_loader))

    tic = time()
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        processed_size += inputs.size(0)

        del inputs
        del outputs
        del labels
        torch.cuda.empty_cache()
        gc.collect()

    toc = time()

    print("Eval epoch time: ", toc - tic)

    Y_hat = model(X_val.to(device))
    Y_hat = Y_hat.to('cpu').detach()

    plt.figure(figsize=(8, 10))
    # Visualize tools
    for k in range(3):
        plt.subplot(3, 3, k+1)
        plt.imshow(np.rollaxis(X_val[k].numpy(), 0, 3))
        plt.title('Real image')
        plt.axis('off')

        plt.subplot(3, 3, k+4)
        plt.imshow(torch.sigmoid(Y_hat[k, 0]) > 0.5, cmap='gray')
        plt.title('Output')
        plt.axis('off')

        plt.subplot(3, 3, k+7)
        plt.imshow(torch.sigmoid(Y_val[k, 0]) > 0.5, cmap='gray')
        plt.title('Real mask')
        plt.axis('off')

    plt.show()

    del Y_hat
    torch.cuda.empty_cache()

    val_loss = running_loss / processed_size
    return val_loss

total = 0

def train(model, opt, loss_fn, epochs, data_tr, data_val, scheduler=None):
    torch.autograd.set_detect_anomaly(True)
    loss_history = []
    score_history = []

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        gc.collect()

        print('* Epoch %d/%d' % (epoch+1, epochs))

        train_loss = fit_epoch(model, data_tr, loss_fn, opt)
        val_loss = eval_epoch(model, data_val, loss_fn)
        if scheduler:
            scheduler.step(val_loss)

        loss_history.append((train_loss, val_loss))

        # train_score = score_model(model, iou_pytorch, data_tr)
        tic = time()
        val_score = score_model(model, iou_pytorch, data_val)
        toc = time()
        print("Score on eval time: ", toc - tic)
        score_history.append(val_score)

        global total
        total += 1
        torch.save(model.state_dict(), f"segnet_bce_1125_{total}_epoch.pth")

        print(f'\n\tTrain loss: {train_loss};'
              f'\n\tVal loss: {val_loss};'
              f' \n\tVal score: {val_score};')

    return loss_history, score_history
