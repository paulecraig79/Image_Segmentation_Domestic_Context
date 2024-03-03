from torchvision.models.segmentation import deeplabv3_resnet101
import torch.nn as nn



class DeepLab(nn.Module):
    def __init__(self, num_classes=1):
        super(DeepLab, self).__init__()
        self.model = deeplabv3_resnet101(pretrained=False, progress=True)
        self.model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)['out']