from glob import glob
import torch
from PIL import Image
from torchvision import transforms
import os

V = 32
F = 16
PATH_ANOMALOUS = "/workspace/datasets/Skull"
PATH_NORMAL = "/workspace/datasets/Upvote"
FILE_NAME_TEMPLATE = "image_%04d.jpg"

class DataSet(torch.utils.data.Dataset):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.length = min((
            len(glob(os.path.join(PATH_ANOMALOUS, "*.jpg"))),
            len(glob(os.path.join(PATH_NORMAL, "*.jpg"))),
        )) - F

    def __len__(self):
        # return self.length
        return int(6000 / F)

    def __getitem__(self, idx):
        return torch.stack([
            torch.stack([self.transform(self._open_image(path)) for path in self._get_pathlist(PATH_ANOMALOUS, n, F)])
            for n in range(idx * F, (idx + V) * F, F)            
        ]), torch.stack([
            torch.stack([self.transform(self._open_image(path)) for path in self._get_pathlist(PATH_NORMAL, n, F)])
            for n in range(idx * F, (idx + V) * F, F)
        ])
    
    def _open_image(self, path):
        return Image.open(path).convert("RGB")

    def _get_pathlist(self, dirpath, idx, f):
        return [
            os.path.join(
                dirpath, "image_%04d.jpg" % num
            )
            for num in range(idx, idx + f)
        ]