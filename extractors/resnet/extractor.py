import torch
import torch.nn as nn
from resnet import resnet152

RESNET_PRETRAINED = "./model_best.pth.tar"

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet152()
        self.resnet = torch.nn.DataParallel(self.resnet)
        self.resnet.load_state_dict(torch.load(RESNET_PRETRAINED)["state_dict"])

    def forward(self, images):
        return self.resnet(images)
