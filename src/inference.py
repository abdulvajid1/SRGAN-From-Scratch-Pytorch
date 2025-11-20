from utils import lowres_transform, load_checkpoint
from generator import Generator
import pathlib
import albumentations as A
import cv2
import numpy as np
import torch
from torchvision.utils import save_image
import argparse
from pathlib import Path
from PIL import Image

initial_transform = A.Compose([
    A.Resize(128, 128, interpolation=cv2.INTER_CUBIC),
    A.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0],),
    A.ToTensorV2()
])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
generator = Generator().to(device=device)
generator = torch.compile(generator)
load_checkpoint(generator, is_infer=True)
generator.eval()


# read image
@torch.inference_mode()
def inference(lowres_image):
    img = np.array(Image.open(lowres_image))
    img = initial_transform(image=img)['image'].unsqueeze(0)
    highres = generator(img.to(device))
    highres.detach_()
    highres = highres.cpu()
    save_image(highres, fp="baboon_pred.png", normalize=True)


def main(args):
    inference(args.img_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default='baboon.png')
    args = parser.parse_args()
    if Path(args.img_path).exists():
        main(args)
    else:
        raise FileNotFoundError("Could not find the files")