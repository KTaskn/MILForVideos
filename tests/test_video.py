from video import Extractor, VideoFeature
import torch
import pytest

class TestVideoFeature:
    def test_init(self):
        path_list = []
        labels = []
        features = torch.tensor([])
        video_feature = VideoFeature(path_list, labels, features)
        assert video_feature.path_list == path_list
        assert type(video_feature.labels) is torch.Tensor
        assert video_feature.labels.tolist() == labels
        assert type(video_feature.features) is torch.Tensor
        assert all(video_feature.features == features)
        
        path_list = [1, 2, 3]
        labels = [1, 2, 3]
        features = torch.rand([3, 5])
        VideoFeature(path_list, labels, features)
    
    
    def test_pathlistとfeaturesは同じ長さの必要がある(self):
        path_list = [1]
        features = torch.tensor([])
        with pytest.raises(Exception):
            VideoFeature(path_list, features)
            
        path_list = [1, 2, 3]
        features = torch.rand([5, 5])
        with pytest.raises(Exception):
            VideoFeature(path_list, features)
        
    
    
    def test_pathlistとlabelsは同じ長さの必要がある(self):
        path_list = [1]
        labels = [1, 2]
        features = torch.tensor([1])
        with pytest.raises(Exception):
            VideoFeature(path_list, features)
            
        path_list = [1, 2, 3]
        labels = [2, 3]
        features = torch.rand([3, 5])
        with pytest.raises(Exception):
            VideoFeature(path_list, features)

class TmpModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x + 1
       

class TestExtractor:
    def test_extractor(self):
        path_list = ["1.jpg", "2.jpg", "3.jpg"]
        labels = [1, 2, 3]
        model = TmpModel()
        
        def parser(path):
            if path == "1.jpg":
                return torch.tensor([1])
            elif path == "2.jpg":
                return torch.tensor([2])
            else:                
                return torch.tensor([3])                
        
        extractor = Extractor(path_list, labels, model, parser)
        video_feature = extractor.extract()
        assert type(video_feature) is VideoFeature
        assert video_feature.path_list == path_list
        assert video_feature.labels.tolist() == labels
        assert video_feature.features.size() == torch.Size([3, 1])
        assert all(video_feature.features == torch.tensor([[2], [3], [4]]))
        
        
        extractor = Extractor(path_list, labels, model, parser, n_batches=10, n_workers=10)
        assert extractor.n_batches == 10
        assert extractor.n_workers == 10
        
    
    def test_extractor_2(self):
        path_list = ["1.jpg", "2.jpg", "3.jpg"]
        labels = [1, 2, 3]
        model = TmpModel()
        
        def parser(path):
            if path == "1.jpg":
                return torch.tensor([1, 1])
            elif path == "2.jpg":
                return torch.tensor([2, 2])
            else:                
                return torch.tensor([3, 3])                
        
        extractor = Extractor(path_list, labels, model, parser)
        video_feature = extractor.extract()
        assert type(video_feature) is VideoFeature
        assert video_feature.path_list == path_list
        assert video_feature.labels.tolist() == labels
        assert video_feature.features.size() == torch.Size([3, 2])
        assert (video_feature.features == torch.tensor([[2, 2], [3, 3], [4, 4]])).all()
        
        
    def test_extractor_cuda(self):
        path_list = ["1.jpg", "2.jpg", "3.jpg"]
        labels = [1, 2, 3]
        model = TmpModel()
        
        def parser(path):
            if path == "1.jpg":
                return torch.tensor([1, 1])
            elif path == "2.jpg":
                return torch.tensor([2, 2])
            else:                
                return torch.tensor([3, 3])                
        
        extractor = Extractor(path_list, labels, model, parser, cuda=True)
        
        video_feature = extractor.extract()
        assert video_feature.path_list == path_list
        assert video_feature.labels.tolist() == labels
        assert video_feature.features.size() == torch.Size([3, 2])
        assert (video_feature.features == torch.tensor([[2, 2], [3, 3], [4, 4]])).all()