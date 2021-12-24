import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from mil import MIL
from i3d import InceptionI3d
from dataset import DataSet, F, V

PRETRAINED_PATH = "./rgb_i3d_pretrained.pt"

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.i3d = InceptionI3d()
        self.i3d.load_state_dict(torch.load(PRETRAINED_PATH))

        # パラメータ固定
        for param in self.i3d.parameters():
            param.requires_grad = False

    def forward(self, batch):
        n_batch = batch.size()[0]
        return torch.stack([
            self._forward(batch[idx_batch])
            for idx_batch in range(n_batch)
        ])
    
    def _forward(self, videos):
        videos.transpose_(1, 2)
        x = self.i3d(videos)
        x = x.squeeze(2)
        return self.affine(x).squeeze(1)