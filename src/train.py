from os import path
from pathlib import Path
from typing import Iterator
from numpy import iterable
import tqdm
import config
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import load_checkpoint, save_checkpoint, visualize_sample
from generator import Generator
from descriminator import Descriminator
from dataset import get_dataloader
from torch.utils.tensorboard import SummaryWriter
import logging
from rich.logging import RichHandler
from utils import VGGLoss



logging.basicConfig(
    # filename="training.log",
    level=logging.INFO,
    datefmt="[%X]",                # optional time format
    handlers=[RichHandler()]
    )

writer = SummaryWriter()

torch.set_float32_matmul_precision('high')

# evaluate
@torch.no_grad()
def evaluate(generator, descriminator, test_dataloader, device):
    generator.eval()
    progress_bar = tqdm.tqdm(test_dataloader, dynamic_ncols=True)
    gen_loss_list = [], desc_loss_list = []
    
    for step, (highres_real, lowres_img) in enumerate(progress_bar):
        highres_gen = generator(lowres_img)
        
        highres_real_pred = descriminator(highres_real)
        highres_gen_pred = descriminator(highres_gen.detach())
        
        highres_real_labels = torch.zeros(highres_real_pred.size()[0]).to(device)
        highres_gen_labels = torch.ones(highres_gen_pred.size()[0]).to(device)
        
        desc_highres_real_loss = F.binary_cross_entropy_with_logits(highres_real_pred.view(-1), highres_real_labels)
        desc_highres_gen_loss = F.binary_cross_entropy_with_logits(highres_gen_pred.view(-1), highres_gen_labels)
        
        desc_loss = desc_highres_real_loss + desc_highres_gen_loss
        
        highres_gen_pred = descriminator(highres_gen)
        gen_loss = F.binary_cross_entropy_with_logits(highres_real_pred, highres_real_labels)
        
        gen_loss_list.append(gen_loss)
        desc_loss_list.append(desc_loss)
        
    gen_loss_mean = torch.tensor(gen_loss_list).mean()
    desc_loss_mean = torch.tensor(desc_loss_list).mean()
    generator.train()
    
    return gen_loss_mean, desc_loss_mean
        

# train
def train(generator, descriminator, genr_optimizer, desc_optimizer, train_dataloader, vgg_loss, device='cuda', epoch=10, save_img_path=None, save_step=10, alpha=0.001):
    
    progress_bar = tqdm.tqdm(train_dataloader, dynamic_ncols=True)
    
    for step, (highres_real, lowres_img) in enumerate(progress_bar):
        highres_real, lowres_img = highres_real.to(device), lowres_img.to(device)
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            highres_gen = generator(lowres_img)
            highres_real_pred = descriminator(highres_real)
            highres_gen_pred = descriminator(highres_gen.detach())
            
            highres_real_labels = torch.ones(highres_real_pred.size()[0]).to(device)
            highres_gen_labels = torch.zeros(highres_gen_pred.size()[0]).to(device)
            
            desc_highres_real_loss = F.binary_cross_entropy_with_logits(highres_real_pred.flatten(), highres_real_labels)
            desc_highres_gen_loss = F.binary_cross_entropy_with_logits(highres_gen_pred.flatten(), highres_gen_labels)
        
            desc_loss = desc_highres_real_loss + desc_highres_gen_loss
        
        desc_optimizer.zero_grad()
        desc_loss.backward()
        desc_optimizer.step()
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            highres_gen_pred = descriminator(highres_gen)
            gen_loss = F.binary_cross_entropy_with_logits(highres_gen_pred.flatten(), highres_real_labels)
            # content_loss = F.mse_loss(highres_gen, highres_real)
            content_loss = vgg_loss(highres_gen, highres_real)
            gen_loss = alpha * gen_loss + (1.0 - alpha) * content_loss
            
            
        
        genr_optimizer.zero_grad()
        gen_loss.backward()
        genr_optimizer.step()
        
        if step % 10 == 0:
            writer.add_scalar("train/gen_loss", gen_loss, step)
            writer.add_scalar("train/desc_loss", desc_loss, step)
        
        progress_bar.set_postfix({
            "genr_loss": f"{gen_loss.item(): .5f}",
            "desc_loss": f"{desc_loss.item(): .5f}",
        })
        
        
        if step % save_step == 0:
            samples = next(iter(train_dataloader))
            visualize_sample(generator, samples, step, path=save_img_path, device=device)
        
        
    


# main
def main():
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    generator = Generator().to(device)
    descriminator = Descriminator().to(device)
    vgg_loss = VGGLoss().eval().to(device)
    
    generator = torch.compile(generator)
    descriminator = torch.compile(descriminator)
    vgg_loss = torch.compile(vgg_loss)
    
    logging.info("Model loaded & Compiled")
    
    desc_lr = config.desc_lr
    genr_lr = config.genr_lr
    
    genr_optimizer = optim.AdamW(generator.parameters(), lr=genr_lr)
    desc_optimizer = optim.AdamW(descriminator.parameters(), lr=desc_lr, weight_decay=0.2)
    
    if config.is_load_checkpoint:
        generator, descriminator, optimizer = load_checkpoint(generator, descriminator, optimizer=optimizer)
    
    
    train_dataloader = get_dataloader(img_root_dir='data', batch_size=config.batch_size, device=device, num_workers=2, pin_memory=True)
    logging.info(f"Loaded data loader with batch size of {config.batch_size}")
    # test_dataloader = get_dataloader(img_root_dir='./test', shuffle=False, batch_size=1, pin_memory=False, num_workers=2)
    
    num_epoches = config.num_epoches
    eval_step = config.eval_step
    sample_save_path = Path("sample_images")
    sample_save_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Training Started")
    for epoch in range(1, num_epoches+1):
        generator.train()
        descriminator.train()
        train(generator, descriminator, genr_optimizer, desc_optimizer, train_dataloader, vgg_loss, epoch=epoch, device=device, save_img_path=sample_save_path)
        
        # samples = next(iter(train_dataloader))
        # visualize_sample(generator, samples, epoch, path=sample_save_path)
        # logging.info(f"Saved sample images for epoch {epoch}")
        # if epoch % eval_step == 0:
        #     evaluate(generator, descriminator, test_dataloader, epoch=epoch)
        
        
if __name__ == "__main__":
    main()
            
        