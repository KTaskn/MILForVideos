import torch
from PIL import Image
from torchvision import transforms
import os

F = 16
PATH_ANOMALOUS = "/workspace/datasets/Skull"
PATH_NORMAL = "/workspace/datasets/Upvote"
FILE_NAME_TEMPLATE = "image_%04d.jpg"

class DataSet(torch.utils.data.Dataset):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        return torch.stack([
            self.transform(self._open_image(path))
            for path in self._get_pathlist(PATH_ANOMALOUS, idx, F)
        ]), torch.stack([
            self.transform(self._open_image(path))
            for path in self._get_pathlist(PATH_ANOMALOUS, idx, F)
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