import torch
import torch.nn as nn

class MIL(nn.Module):
    def __init__(self, lambda1=8e-5, lambda2=8e-5, lambda3=0.01):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

    def __call__(self, anomalous, normal, model):
        weights_norm = torch.mean(torch.tensor([
            w.data.norm()
            for w in model.parameters()
        ]))
        batch_size = float(anomalous.size()[0])
        term0 = self._term0(anomalous, normal)
        term1 = self._term1(anomalous)
        term2 = self._term2(anomalous)
        return (
            term0
            + self.lambda1 * term1 / batch_size
            + self.lambda2 * term2 / batch_size
            + self.lambda3 * weights_norm)

    def _term0(self, anomalous, normal):
        a_max = anomalous.max()
        n_max = normal.max()
        print("a_max:", a_max)
        print("n_max:", n_max)
        return torch.tensor([
                0.0,
                1.0 - a_max + n_max
            ]).max()
        
    def _term1(self, anomalous):
        return torch.pow(anomalous.diff(), 2.0).sum()

    def _term2(self, anomalous):
        return anomalous.sum()