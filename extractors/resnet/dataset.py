import logging
import torch
from PIL import Image
from torchvision import transforms
import math

class DataSet(torch.utils.data.Dataset):
    def __init__(self, paths_image, func_extract=None):
        self.paths_image = paths_image
        
        # Function to transform images into features
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Function to extract features
        if func_extract is not None:
            # if you want to use your own extract function
            self.func_extract = func_extract
        else:
            self.func_extract = self._func_extract

    def __len__(self):
        return self.paths_image.__len__()

    def __getitem__(self, idx):
        return self.func_extract(self.paths_image[idx])
    
    def _func_extract(self, path_image):
        return self.transform(self._open_image(path_image))
    
    def _open_image(self, path):
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            logging.error(path)
            raise e
