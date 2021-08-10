import torch
from dataset import DataSet, V, F

class TestDataSet:
    def test_get_pathlist(self):
        ds = DataSet()
        assert ds._get_pathlist(
            "/workspace/datasets/Skull",
            9998, 3
        ) == [
            "/workspace/datasets/Skull/image_9998.jpg",
            "/workspace/datasets/Skull/image_9999.jpg",
            "/workspace/datasets/Skull/image_10000.jpg",
        ]

    def test_init(self):
        ds = DataSet()
        W, H, C = 224, 224, 3
        a_data, n_data = next(iter(ds))
        assert a_data.shape == torch.Size([V, F, C, W, H])
        assert n_data.shape == torch.Size([V, F, C, W, H])