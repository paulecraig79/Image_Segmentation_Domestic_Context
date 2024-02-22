import torch
import torchvision


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
        for x, y in loader:
            x = x.float().to(device)
            y = y.to(device).squeeze(1)  # Squeeze the channel dimension
            preds_sigmoid = torch.sigmoid(model(x))
            preds_threshold = (preds_sigmoid > 0.5).float()

            # Visualize a sample of preds_thresholded and y for comparison
            print("Sample of preds_thresholded:")
            print(preds_threshold[0, 0, :10, :10])  # Print the first 10x10 values of the first prediction
            print("\nSample of y:")
            print(y[0, :10, :10])  # Print the first 10x10 values of the first ground truth label

            num_correct_manual = torch.sum((preds_threshold == y).float()).item()
            print("\nNumber of correct predictions (manual calculation):", num_correct_manual)


            num_correct += (preds_threshold == y).sum().item()
            num_pixels += torch.numel(y)
            dice_score += (2 * (preds_threshold * y).sum()) / ((preds_threshold + y).sum() + 1e-8)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print("Dice score:", dice_score.item())
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.float().to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )


    model.train()