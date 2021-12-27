import torch
from glob import glob
from extractor import MyNet, extract
from dataset import DataSet

def test_MyNet():
    paths_image = sorted(glob("/datasets/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Train/Train002/*"))
    ds = DataSet(paths_image)
    videos = torch.stack([ds.__getitem__(idx) for idx in range(5)])
    
    net = MyNet()
    assert net(videos).size() == torch.Size([5, 1024])


def test_extract():    
    paths_image = sorted(glob("/datasets/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Train/Train002/*"))[:32]
    ds = DataSet(paths_image)
    net = MyNet()
    assert extract(ds, net, n_batches=1).size() == torch.Size([2, 1024])
    assert extract(ds, net, n_batches=2).size() == torch.Size([2, 1024])
    
    paths_image = sorted(glob("/datasets/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Train/Train002/*"))[:64]
    ds = DataSet(paths_image)
    assert extract(ds, net, n_batches=1).size() == torch.Size([4, 1024])
    assert extract(ds, net, n_batches=2).size() == torch.Size([4, 1024])
    