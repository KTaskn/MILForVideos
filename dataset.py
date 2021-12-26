import logging
from glob import glob
import torch
from PIL import Image
from torchvision import transforms
import os
import math

class DataSet(torch.utils.data.Dataset):
    def __init__(self, paths_image, F=16, func_extract=None):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.paths_image = paths_image
        self.F = F
        if func_extract is not None:
            self.func_extract = func_extract
        else:
            self.func_extract = self._func_extract

    def __len__(self):
        return math.ceil(self.paths_image.__len__() / self.F)

    def __getitem__(self, idx):
        # 小分けにする
        start = idx * self.F
        end = idx * self.F + self.F
        
        if start <= self.paths_image.__len__() - self.F:
            sub = self.paths_image[start:end]
        else:
            sub = self.paths_image[self.paths_image.__len__() - 5:self.paths_image.__len__()]
        return self.func_extract(sub)
    
    def _func_extract(self, sub_paths_image):
        return torch.stack([
            self.transform(self._open_image(x))
            for x in sub_paths_image
        ])
    
    def _open_image(self, path):
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            logging.error(path)
            raise e
