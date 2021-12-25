import torch
from glob import glob
from extractor import MyNet
from dataset import DataSet

def test_extractor():
    paths_image = sorted(glob("/datasets/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Train/Train002/*"))
    ds = DataSet(paths_image)
    videos = torch.stack([ds.__getitem__(idx) for idx in range(5)])
    
    net = MyNet()
    net(videos)