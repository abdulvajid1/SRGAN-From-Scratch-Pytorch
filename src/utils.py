import os
from pathlib import Path
import torch
import albumentations as A
import config
import torch.nn as nn
from torchvision.models import vgg19
from torchvision.utils import save_image, make_grid

import descriminator



both_transform = A.Compose([
    A.CenterCrop(config.highres, config.highres),
    A.HorizontalFlip(),
])

highres_transform = A.Compose([
    A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    A.ToTensorV2(),
])

lowres_transform = A.Compose([
    A.Resize(config.highres/4, config.highres/4),
    A.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    A.ToTensorV2()
])


def save_checkpoint(generator: nn.Module, descriminator: nn.Module, gen_optimizer, desc_optimizer, global_step):
    path = os.path.join('checkpoints')
    save_path = Path(path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    generator_state = generator.state_dict()
    descriminator_state = descriminator.state_dict()
    gen_optimizer_state = gen_optimizer.state_dict()
    desc_optimizer_state = desc_optimizer.state_dict()
    full_state = dict(gen=generator_state, desc=descriminator_state, gen_optim=gen_optimizer_state, desc_optim=desc_optimizer_state)
    
    torch.save(full_state, os.path.join(save_path, f"ckpt_{global_step}.ckpt"))

def load_checkpoint(generator, descriminator, gen_optimizer, desc_optimizer):
    path = Path('checkpoints')
    last_model_path = sorted(list(path.glob("*.ckpt")))[-1]
    
    load_dict = torch.load(last_model_path)
    generator.load_state_dict(load_dict['gen'])
    descriminator.load_state_dict(load_dict['desc'])
    gen_optimizer.load_state_dict(load_dict['gen_optim'])
    desc_optimizer.load_state_dict(load_dict['desc_optim'])
    return last_model_path

@torch.inference_mode()
def visualize_sample(generator, samples, step, path, device):
    generator.eval()
    high_res_img, low_res_img = samples[0].to(device), samples[1].to(device) 
    high_res_perd = generator(low_res_img)
    high_res_perd.detach_()
    
    # save_image(high_res_img[0].cpu(), fp=path / f'real_img{step}.png', normalize=True)
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
        for param in self.vgg19.parameters():
            param.requires_grad = False
        self.mse = nn.MSELoss()
    
    def forward(self, input_img, target_img):
        input_img = self.vgg19(input_img)
        target_img = self.vgg19(target_img)
        return self.mse(input_img, target_img)
