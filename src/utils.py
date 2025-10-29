import albumentations as A
import config


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
    A.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    A.ToTensorV2()
])


def save_checkpoint(model, optimizer, step):
    pass

def load_checkpoint(model, optimizer):
    pass
