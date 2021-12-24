from glob import glob
import torch
from PIL import Image
from torchvision import transforms
import os
import math

class DataSet(torch.utils.data.Dataset):
    def __init__(self, images, F=16, func_extract=None):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.images = images
        self.F = F
        if func_extract is not None:
            self.func_extract = func_extract
        else:
            self.func_extract = lambda x: self.transform(self._open_image(x))

    def __len__(self):
        return math.ceil(self.images.__len__() / self.F)

    def __getitem__(self, idx):
        # 小分けにする
        start = idx * self.F
        end = idx * self.F + self.F
        
        if start <= self.images.__len__() - self.F:
            return [
                self.func_extract(path)
                for path in self.images[start:end]
            ]
        else:
            return [
                self.func_extract(path)
                for path in self.images[self.images.__len__() - 5:self.images.__len__()]
            ]
    
    def _open_image(self, path):
        return Image.open(path).convert("RGB")
