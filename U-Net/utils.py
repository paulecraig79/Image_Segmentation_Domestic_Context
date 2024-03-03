import time

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print('Saving checkpoint..')
    torch.save(state,filename)

def load_checkpoint(checkpoint, model):
    print('Loading checkpoint..')
    model.load_state_dict(checkpoint['state_dict'])


def save_predictions_as_imgs(
    loader, model, device="cuda"
):
    Start_time = time.time()
    model.eval()
    num_correct = 0
    num_pixels = 0
    global_accuracy = 0
    global_prec = 0
    global_recall = 0
    global_f1_score =0
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.float().to(device=device)
            y = y.to(device=device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum().item()
            num_pixels += y.numel()







            #Plot predictions
            #plt.figure(figsize=(12, 6))
            for i in range(len(x)):
                #plt.subplot(1, len(x), i + 1)
                #plt.imshow(preds[i].cpu().numpy().squeeze(), cmap='gray')




                precision = precision_score(y[i].cpu().numpy(), preds[i].cpu().numpy())
                recall = recall_score_(y[i].cpu().numpy(), preds[i].cpu().numpy())
                f1 = dice_coef(y[i].cpu().numpy(), preds[i].cpu().numpy())
                accuracy = num_correct / num_pixels


                global_accuracy += accuracy
                global_prec += precision
                global_recall += recall
                global_f1_score += f1


            #
                #plt.title(f'Prediction {idx*len(x) + i}: acc: {accuracy*100:.4f}% + Precision: {precision*100:.4f}% + Recall: {recall*100:.4f}% + F1: {f1*100:.4f}%')
                #plt.axis('off')
            #plt.show()


        print(f"Average Accuracy: {global_accuracy / len(loader) * 100:.4f}%")
        print(f"Average Precision: {global_prec / len(loader) * 100:.4f}%")
        print(f"Average Recall: {global_recall / len(loader) * 100:.4f}%")
        print(f"Average F1-Score: {global_f1_score / len(loader) * 100:.4f}%")
        print(f"Number of images: {len(loader)}")
        End_Time = time.time()
        Total_Time = End_Time - Start_time
        print(f"Total time: {Total_Time}")
        print(f"Average time per image: {Total_Time/len(loader):.4f}s")
    model.train()


def precision_score(ground_truth, predictions):
    intersection = np.sum(ground_truth * predictions)
    total_pixel_pred = np.sum(predictions)
    precision = np.mean(intersection/total_pixel_pred)
    return precision


def recall_score_(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_pixel_truth = np.sum(groundtruth_mask)
    recall = np.mean(intersect/total_pixel_truth)
    return recall

def dice_coef(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_sum = np.sum(pred_mask) + np.sum(groundtruth_mask)
    dice = np.mean(2*intersect/total_sum)
    return dice


