from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from utils import (both_transform, 
                   highres_transform, 
                   lowres_transform)


class CustomImgDataset(Dataset):
    def __init__(self, root_dir, device):
        super().__init__()
        self.images = Path(root_dir).glob('*')
        self.device = device
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # read img
        image = Image.open(self.images[index])
        # both transform
        image = both_transform(image=image)['image']
        # highres & low res
        high_res_img = highres_transform(image=image).to(self.device)
        low_res_img = lowres_transform(image=image).to(self.device)
        
        return (highres_transform, low_res_img)
    
def get_dataloader(img_root_dir, batch_size=8, shuffle=True, num_workers=2, pin_memory=False, **kwargs):
    dataset = CustomImgDataset(root_dir=img_root_dir)
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            num_workers=num_workers, 
                            pin_memory=pin_memory,
                            **kwargs)
    
    return dataloader
        
        