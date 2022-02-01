import torch
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


class _DataSet(torch.utils.data.Dataset):
    def __init__(self, path_list: List[str], parser: Callable[[str], torch.Tensor]):
        self.path_list = path_list
        self.parser = parser
        
    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        return self.parser(self.path_list[idx])

class Extractor:
    def __init__(self, path_list: List[str], labels: List[int], model: torch.nn.Module, parser: Callable[[str], torch.Tensor],
                 n_batches=5, n_workers=5, cuda=False):
        self.path_list = path_list
        self.labels = labels
        self.cuda = cuda
        self.model = model.eval().cuda() if self.cuda else model.eval()
        self.parser = parser
        self.n_batches = n_batches
        self.n_workers = n_workers
    
    def extract(self):
        with torch.no_grad():
            dataset = _DataSet(self.path_list, self.parser)        
            loader = torch.utils.data.DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.n_batches,
                num_workers=self.n_workers)
            Y = torch.cat([
                self.model(batch.cuda() if self.cuda else batch)
                for batch in loader
            ])
            Y = Y.cpu() if self.cuda else Y
            return VideoFeature(self.path_list, self.labels, Y)
        