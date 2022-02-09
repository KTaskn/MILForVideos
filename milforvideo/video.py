import torch
import numpy as np
from typing import List, Callable

class VideoFeature:
    def __init__(self, path_list: List[str], labels: List[int], features: torch.Tensor):
        if len(path_list) != len(labels):
            raise Exception(f"path_list and labels must be same length, but {len(path_list)} and {len(labels)}")
        
        if len(path_list) != features.size(0):
            raise Exception(f"path_list and features must be same length, but {len(path_list)} and {features.size(0)}")
        
        self.path_list = path_list
        self.features = features
        self.labels = torch.tensor(labels)
    
    def compute_instances(self, V=32):        
        if V > self.features.size(0):
            raise ValueError(f"number of instances must be less than {self.features.size(0)}, but {V}")
        if self.features.size(0) % V == 0:
            n_heads = self.features.size(0) // V
            # label は max をとって、-1 なら（すべて-1なら）、後段の処理で無視する
            return torch.stack([
                self.features[num:num+n_heads].mean(dim=0)
                for num in range(0, self.features.size(0), n_heads)
            ]), torch.cat([
                self.labels[num:num+n_heads].max(dim=0)[0]
                for num in range(0, self.labels.size(0), n_heads)
            ])
        else:
            n_heads = self.features.size(0) // (V - 1)
            n_tail = self.features.size(0) % (V - 1)
            
            feature_heads = torch.stack([
                self.features[num:num+n_heads].mean(dim=0)
                for num in range(0, self.features.size(0) - n_tail, n_heads)
                if self.features[num:num+n_heads].size(0) == n_heads
            ])
            feature_tail = self.features[-n_tail:].mean(dim=0).unsqueeze(0)
                        
            label_heads = torch.cat([
                self.labels[num:num+n_heads].max(dim=0)[0]
                for num in range(0, self.labels.size(0) - n_tail, n_heads)
                if self.features[num:num+n_heads].size(0) == n_heads
            ])
            label_tail = self.labels[-n_tail:].max(dim=0)[0].unsqueeze(0)
            return torch.cat([
                feature_heads, feature_tail
            ]), torch.cat([
                label_heads, label_tail
            ])
    
    @classmethod
    def concat(cls, a, b):
        return cls(a.path_list + b.path_list, a.labels.tolist() + b.labels.tolist(), torch.cat([a.features, b.features]))

class _DataSetForParsing(torch.utils.data.Dataset):
    def __init__(self, path_list: List[str], parser: Callable[[str], torch.Tensor]):
        self.path_list = path_list
        self.parser = parser
        
    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        return self.parser(self.path_list[idx])

class Extractor:
    def __init__(self, path_list: List[str], labels: List[int], model: torch.nn.Module, parser: Callable[[str], torch.Tensor],
                 F=None, aggregate=None,
                 n_batches=5, n_workers=5, cuda=False):        
        if F is None:
            self.path_list = path_list
            self.labels = [
                aggregate(row) if aggregate else row
                for row in labels
            ]
        else:
            self.path_list = self.fold(path_list, F)
            self.labels = [
                aggregate(row) if aggregate else row
                for row in self.fold(labels, F)
            ]            
        self.cuda = cuda
        self.model = model.eval().cuda() if self.cuda else model.eval()
        self.parser = parser
        self.n_batches = n_batches
        self.n_workers = n_workers
    
    def extract(self):
        with torch.no_grad():
            dataset = _DataSetForParsing(self.path_list, self.parser)        
            loader = torch.utils.data.DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.n_batches,
                num_workers=self.n_workers)
            
            Y = torch.cat([
                torch.cat([
                    self.model(batch[:, idx].cuda() if self.cuda else batch[:, idx])
                    for idx in range(batch.size(1))
                ], axis=1)
                for batch in loader
            ])
            Y = Y.cpu() if self.cuda else Y
            return VideoFeature(self.path_list, self.labels, Y)
    
    def images(self):
        return _DataSetForParsing(self.path_list, self.parser)
    
    @classmethod
    def fold(cls, l_path, F):    
        return [
            l_path[idx:idx+F] if idx + F < len(l_path) else l_path[-F:]
            for idx in range(0, len(l_path), F)
        ]

class _VideoFeaturesDataSet(torch.utils.data.Dataset):
    def __init__(self, video_features_normal: List[VideoFeature], video_features_anomalous: List[VideoFeature], V=32):
        self.video_features_normal = video_features_normal
        self.video_features_anomalous = video_features_anomalous
        self.V = V

    def __len__(self):
        return len(self.video_features_anomalous)

    def __getitem__(self, idx):
        normal_idx = np.random.randint(0, len(self.video_features_normal))
        video_normal = self.video_features_normal[normal_idx]
        video_anomalous = self.video_features_anomalous[idx]
        
        features_normal, labels_normal = video_normal.compute_instances(self.V)
        features_anonmalous, labels_anomalous = video_anomalous.compute_instances(self.V)
        
        return features_normal, features_anonmalous, labels_normal, labels_anomalous

def generate_dataloader(
    video_features_normal: List[VideoFeature], 
    video_features_anomalous: List[VideoFeature],
    V=32,
    batch_size=1,
    num_workers=0,
    shuffle=True):
    dataset = _VideoFeaturesDataSet(video_features_normal, video_features_anomalous, V=V)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle)