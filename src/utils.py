from pathlib import Path
import torch
import albumentations as A
import config
import torch.nn as nn
from torchvision.models import vgg19
from torchvision.utils import save_image, make_grid



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

@torch.no_grad()
def visualize_sample(generator, samples, step, path, device):
    generator.eval()
    high_res_img, low_res_img = samples[0].to(device), samples[1].to(device) 
    high_res_perd = generator(low_res_img)
    high_res_perd.detach_()
    
    # import code; code.interact(local=locals())
    
    grid = make_grid(
        [high_res_img[0].cpu(), high_res_perd[0].cpu()],
        nrow=1,
        normalize=True
    )
    
    save_image(grid, fp=path / f'comparsion_org_vs_pred_epoch_{step}.png', normalize=True)
    
    generator.train()
    

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg19 = vgg19(pretrained=True).features[:36]
        self.mse = nn.MSELoss()
    
    def forward(self, input_img, target_img):
        input_img = self.vgg19(input_img)
        target_img = self.vgg19(target_img)
        return self.mse(input_img, target_img)
