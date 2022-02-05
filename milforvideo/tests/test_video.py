from video import Extractor, VideoFeature
import torch
import pytest

class TestVideoFeature:
    def test_concat(self):        
        a_path_list = [1, 2, 3]
        a_labels = [1, 2, 3]
        a_features = torch.rand([3, 5])
        a = VideoFeature(a_path_list, a_labels, a_features)
        
        b_path_list = [4, 5, 6]
        b_labels = [4, 5, 6]
        b_features = torch.rand([3, 5])
        b = VideoFeature(b_path_list, b_labels, b_features)
        c = VideoFeature.concat(a, b)
        
        assert c.path_list == a_path_list + b_path_list
        assert type(c.labels) is torch.Tensor
        assert c.labels.tolist() == a_labels + b_labels
        assert type(c.features) is torch.Tensor
        assert (c.features == torch.cat([a_features, b_features])).all().all()
        
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
    
    
    def test_compute_instances_default_Vで分割する(self):
        V = 32
        TMP = 10
        N = TMP * V
        path_list = list(range(N))
        labels = list(range(N))
        features = torch.rand([N, 10, 1000])
        vf = VideoFeature(path_list, labels, features)
        # defaultは32分割
        actual = vf.compute_instances()
        assert actual.size() == torch.Size([V, 10, 1000])
        for idx in range(V):
            assert (features[idx * TMP:(idx + 1) * TMP].mean(dim=0) == actual[idx]).all()
        
        N = TMP * (V - 1) + 5
        path_list = list(range(N))
        labels = list(range(N))
        features = torch.rand([N, 10, 1000])
        vf = VideoFeature(path_list, labels, features)
        # defaultは32分割
        actual = vf.compute_instances()
        assert actual.size() == torch.Size([V, 10, 1000])
        assert (features[-5:].mean(dim=0) == actual[-1]).all()
        
        
        N = TMP * (V - 1) + 30
        path_list = list(range(N))
        labels = list(range(N))
        features = torch.rand([N, 10, 1024])
        vf = VideoFeature(path_list, labels, features)
        # defaultは32分割
        actual = vf.compute_instances()
        assert actual.size() == torch.Size([V, 10, 1024])
        
    
    def test_compute_instances_Vを変更しで分割する(self):
        TMP = 10
        V = 5
        N = TMP * V
        path_list = list(range(N))
        labels = list(range(N))
        features = torch.rand([N, 10, 1000])
        vf = VideoFeature(path_list, labels, features)
        actual = vf.compute_instances(V=V)
        assert actual.size() == torch.Size([V, 10, 1000])
        for idx in range(5):
            assert (features[idx * TMP:(idx + 1) * TMP].mean(dim=0) == actual[idx]).all()
        
        N = TMP * (V - 1) + 2
        path_list = list(range(N))
        labels = list(range(N))
        features = torch.rand([N, 10, 1000])
        vf = VideoFeature(path_list, labels, features)
        # defaultは32分割
        actual = vf.compute_instances(V=5)
        assert actual.size() == torch.Size([V, 10, 1000])
        assert (features[-2:].mean(dim=0) == actual[-1]).all()
        
        N = 1001
        path_list = list(range(N))
        labels = list(range(N))
        features = torch.rand([N, 10, 1024])
        vf = VideoFeature(path_list, labels, features)
        # defaultは32分割
        actual = vf.compute_instances(V=V)
        assert actual.size() == torch.Size([V, 10, 1024])    
    
    def test_compute_instances_Vより数が少ない(self):
        N = 99
        V = 100
        path_list = list(range(N))
        labels = list(range(N))
        features = torch.rand([N, 10, 1000])
        vf = VideoFeature(path_list, labels, features)
        with pytest.raises(Exception):
            vf.compute_instances(V=V)
        
        N = 100
        V = 100
        path_list = list(range(N))
        labels = list(range(N))
        features = torch.rand([N, 10, 1000])
        vf = VideoFeature(path_list, labels, features)
        vf.compute_instances(V=V)
        


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
        
    
    def test_extractor_multicol(self):
        path_list = [
            ["1.jpg", "2.jpg", "3.jpg"], 
            ["4.jpg", "5.jpg", "6.jpg"], 
            ["7.jpg", "8.jpg", "9.jpg"],
        ]
        labels = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        model = TmpModel()
        
        def parser(path):
            if path == ["1.jpg", "2.jpg", "3.jpg"]:
                return torch.tensor([1])
            elif path == ["4.jpg", "5.jpg", "6.jpg"]:
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
        
    def test_fold(self):
        l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        expect = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],            
            [8, 9, 10]
        ]
        actual = Extractor.fold(l, 3)
        assert expect == actual
                
        l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        expect = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10]
        ]
        actual = Extractor.fold(l, 5)
        assert expect == actual
                
        l = []
        expect = []
        actual = Extractor.fold(l, 5)
        assert expect == actual
        
    
    def test_extractor_fold(self):
        path_list = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg", "8.jpg", "9.jpg", "10.jpg"]
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        model = TmpModel()
        
        def parser(row):
            if row == ["1.jpg", "2.jpg", "3.jpg"]:
                return torch.tensor([1])
            elif row == ["4.jpg", "5.jpg", "6.jpg"]:
                return torch.tensor([2])
            elif row == ["7.jpg", "8.jpg", "9.jpg"]:
                return torch.tensor([3])
            elif row == ["8.jpg", "9.jpg", "10.jpg"]:
                return torch.tensor([4])    
            else:                
                return torch.tensor([5])                
        
        F = 3
        extractor = Extractor(path_list, labels, model, parser, F=F)
        video_feature = extractor.extract()
        assert video_feature.path_list == Extractor.fold(path_list, F)
        assert video_feature.labels.tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [8, 9, 10]]
        assert all(video_feature.features == torch.tensor([[2], [3], [4], [5]]))
        
        extractor = Extractor(path_list, labels, model, parser, F=F, aggregate=max)
        video_feature = extractor.extract()
        assert video_feature.path_list == Extractor.fold(path_list, F)
        assert video_feature.labels.tolist() == [3, 6, 9, 10]
        assert all(video_feature.features == torch.tensor([[2], [3], [4], [5]]))