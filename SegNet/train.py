import os

import torch
from torch import optim

from Model import SegNet
import torch.nn as nn
import torch.nn.functional as F

class Train():

    @staticmethod
    def save_checkpoint(state, path):
        torch.save(state, path)
        print("Checkpoint saved at {}".format(path))


    @staticmethod
    def Train(trainloader, path=None):

        model = SegNet()
        optimizer = optim.SGD(model.parameters(),lr=hyperparam.lr, momentum=hyperparam.momentum)
        loss_fn = nn.CrossEntropyLoss()
        run_epoch = hyperparam.epochs

        if path == None:
            epoch = 0
            path = os.path.join(os.getcwd(), 'segnet_weights.pth.tar')
            print("Creating new checkpoint '{}'".format(path))
        else:
            if os.path.isfile(path):
                print("Loading checkpoint '{}'".format(path))
                checkpoint = torch.load(path)
                epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("Loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch']))
            else:
                print("No checkpoint found at '{}'".format(path))

        for i in range(1, run_epoch + 1):
            print('Epoch {}:'.format(i))
            sum_loss = 0.0

            for j, data in enumerate(trainloader, 1):
                images, labels = data
                optimizer.zero_grad()
                output = model(images)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()

                sum_loss += loss.item()

                print('Loss at {} mini-batch: {}'.format(j, loss.item()/trainloader.batch_size))

            print('Average loss @ epoch: {}'.format((sum_loss/j*trainloader.batch_size)))

        print("Training complete. Saving checkpoint...")
        Train.save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, path)