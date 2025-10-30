import albumentations as A
import config
import torch.nn as nn
from torchvision.models import vgg19



both_transform = A.Compose([
    A.RandomCrop(config.highres, config.highres),
    A.HorizontalFlip(),
    A.RandomRotate90()
])

highres_transform = A.Compose([
    A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    A.ToTensorV2()
])

lowres_transform = A.Compose([
    A.Resize(config.highres/4, config.highres/4),
    A.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    A.ToTensorV2()
])


def save_checkpoint(model, optimizer, step):
    pass

def load_checkpoint(model, optimizer):
    pass

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg19 = vgg19(pretrained=True).features[:36]
        self.mse = nn.MSELoss()
    
    def forward(self, input_img, target_img):
        input_img = self.vgg19(input_img)
        target_img = self.vgg19(target_img)
        return self.mse(input_img, target_img)
