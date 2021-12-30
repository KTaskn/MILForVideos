from dataset import DataSet
from glob import glob
import torch

def test_getitem():
    images = list(range(100))
    ds = DataSet(images, func_extract=lambda x: x)
    assert ds.__getitem__(0) == 0
    assert ds.__getitem__(1) == 1    
    assert ds.__getitem__(2) == 2
    
    ds = DataSet(images, func_extract=lambda x: 2 * x)
    assert ds.__getitem__(0) == 0
    assert ds.__getitem__(1) == 2
    assert ds.__getitem__(2) == 4


def test_deffunction():
    paths_image = sorted(glob("/datasets/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Train/Train002/*.tif"))
    ds = DataSet(paths_image)
    assert ds.__getitem__(0).size() == torch.Size([3, 224, 224])