import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from mil import MIL
from i3d import InceptionI3d

PRETRAINED_PATH = "./rgb_i3d_pretrained.pt"

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.i3d = InceptionI3d()
        self.i3d.load_state_dict(torch.load(PRETRAINED_PATH))

    def forward(self, video):
        video = video.transpose_(1, 2)
        return self.i3d(video).squeeze(2)

def _get_net(net):    
    if net:
        return net
    else:
        return MyNet()
    
N_BATCHES = 5
N_WORKERS = 5
    
def extract(dataset, net = None, n_batches=N_BATCHES, n_workers=N_WORKERS):
    net = _get_net(net)    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=n_batches,
        shuffle=False,
        num_workers=n_workers)

    ret = []
    with tqdm(total=len(loader), unit="batch") as pbar:
        for videos in loader:
            outputs = net(videos)
            pbar.update(1)
            ret.append(outputs)
    return torch.cat(ret)