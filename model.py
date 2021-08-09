import torch
import torch.nn as nn

class MIL(nn.Module):
    def __init__(self, lambda1=0.01, lambda2=0.01, lambda3=0.01):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

    def __call__(self, anomalous, normal, weights):
        return (
            self._term0(anomalous, normal)
            + self._term1(anomalous)
            + self._term2(anomalous)
            + torch.norm(weights))

    def _term0(self, anomalous, normal):
        return torch.tensor([
                0.0,
                1.0 - anomalous.max() + normal.max()
            ]).max()
        
    def _term1(self, anomalous):
        return torch.pow(anomalous.diff(), 2.0).sum()

    def _term2(self, anomalous):
        return anomalous.sum()