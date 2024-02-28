import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from matplotlib import pyplot as plt


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print('Saving checkpoint..')
    torch.save(state,filename)

def load_checkpoint(checkpoint, model):
    print('Loading checkpoint..')
    model.load_state_dict(checkpoint['state_dict'])

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.float().to(device)
            y= y.to(device).unsqueeze(1)
            preds = model(x)
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum().item()
            num_pixels += y.size(0) * y.size(1) *y.size(2) *y.size(3)

            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )


    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")



    model.train()

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    with torch.no_grad():
        for idx, (x, y,cat_id) in enumerate(loader):
            x = x.float().to(device=device)
            preds = torch.sigmoid(model(x))
            preds_softmax = F.softmax(preds, dim=1)
            probability_map = preds_softmax[0, 0]  # Assuming there is only one class/category

            # Predicted class ID is the index of the maximum probability
            predicted_index = torch.argmax(probability_map)
            predicted_class_id = predicted_index.item()

            print("Predicted Index:", predicted_index)
            print("Probability Map Shape:", probability_map.shape)
            print("Probability Map Values:", probability_map)

            fig, axes = plt.subplots(1, 1, figsize=(8, 8))
            axes.imshow(probability_map.cpu().numpy(), cmap='gray')
            axes.axis('off')
            axes.set_title(f'Predicted Class ID: {predicted_class_id}')

            plt.show()

    model.train()