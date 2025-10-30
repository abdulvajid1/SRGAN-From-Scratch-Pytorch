import tqdm
import config
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import load_checkpoint, save_checkpoint
from generator import Generator
from descriminator import Descriminator
from dataset import get_dataloader


# Load data

# evaluate
@torch.no_grad()
def evaluate(generator, descriminator, test_dataloader, device):
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
    
    
    return gen_loss_mean, desc_loss_mean
        

# train
def train(generator, descriminator, genr_optimizer, desc_optimizer, train_dataloader, device):
    
    progress_bar = tqdm.tqdm(train_dataloader, dynamic_ncols=True)
    
    for step, (highres_real, lowres_img) in enumerate(progress_bar):
        highres_gen = generator(lowres_img)
        
        highres_real_pred = descriminator(highres_real)
        highres_gen_pred = descriminator(highres_gen.detach())
        
        highres_real_labels = torch.zeros(highres_real_pred.size()[0]).to(device)
        highres_gen_labels = torch.ones(highres_gen_pred.size()[0]).to(device)
        
        desc_highres_real_loss = F.binary_cross_entropy_with_logits(highres_real_pred.view(-1), highres_real_labels)
        desc_highres_gen_loss = F.binary_cross_entropy_with_logits(highres_gen_pred.view(-1), highres_gen_labels)
        
        desc_loss = desc_highres_real_loss + desc_highres_gen_loss
        
        desc_optimizer.zero_grad()
        desc_loss.backward()
        desc_optimizer.step()
        
        highres_gen_pred = descriminator(highres_gen)
        gen_loss = F.binary_cross_entropy_with_logits(highres_real_pred, highres_real_labels)
        
        genr_optimizer.zero_grad()
        gen_loss.backward()
        genr_optimizer.step()
        
        
        
    


# main
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    descriminator = Descriminator().to(device)
    
    desc_lr = config.desc_lr
    gen_lr = config.gen_lr
    
    genr_optimizer = optim.AdamW(generator.parameters(), lr=gen_lr, weight_decay=0.1)
    desc_optimizer = optim.AdamW(descriminator.parameters(), lr=desc_lr)
    
    if load_checkpoint:
        generator, descriminator, optimizer = load_checkpoint(generator, descriminator, optimizer=optimizer)
    
    train_dataloader = get_dataloader(img_root_dir='./train', device=device)
    test_dataloader = get_dataloader(img_root_dir='./test', shuffle=False, batch_size=1, pin_memory=False, num_workers=2)
    
    num_epoches = config.epoches
    eval_step = config.eval_step
    
    for epoch in range(1, num_epoches+1):
        train(generator, descriminator, genr_optimizer, desc_optimizer, train_dataloader, epoch=epoch)
        if epoch % eval_step == 0:
            train(generator, descriminator, test_dataloader, epoch=epoch)
        