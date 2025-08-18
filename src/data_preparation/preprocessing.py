import torch
from torchvision.transforms import v2


def get_transforms():
    return v2.Compose([
        v2.Resize(size=(256,256)),
        v2.CenterCrop(size=(224,224)),
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
