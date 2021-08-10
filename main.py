import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from mil import MIL
from i3d import InceptionI3d
from dataset import DataSet, F, V

PRETRAINED_PATH = "./rgb_i3d_pretrained.pt"
MODEL_PATH = "./model.pt"

N_BATCH = 5
N_WORKER = 5
N_EPOCH = 100

class MyAffine(nn.Module):
    def __init__(self):
        super().__init__()
        # 学習モデル
        self.activation = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.layer1 = nn.Linear(1024, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 1)

        # self.norm1 = nn.LayerNorm(256)
        # self.norm2 = nn.LayerNorm(128)
        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.6)

    def forward(self, x):
        x = self.dropout1(self.activation(self.layer1(x)))
        x = self.dropout2(self.activation(self.layer2(x)))
        return self.sigmoid(self.layer3(x))


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.i3d = InceptionI3d()
        self.i3d.load_state_dict(torch.load(PRETRAINED_PATH))

        # パラメータ固定
        for param in self.i3d.parameters():
            param.requires_grad = False

        self.affine = MyAffine()

    def forward(self, batch, n_batch):
        return torch.stack([
            self._forward(batch[idx_batch])
            for idx_batch in range(n_batch)
        ])
    
    def _forward(self, videos):
        videos.transpose_(1, 2)
        x = self.i3d(videos)
        x = x.squeeze(2)
        return self.affine(x).squeeze(1)

def train(model, loader):
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = optim.Adadelta(model.parameters(), lr=0.01, eps=1e-8)
    criterion = MIL()

    loss_sum = 0
    total = 0
    with tqdm(total=len(loader), unit="batch") as pbar:
        for anomaly_data, normal_data in loader:
            anomaly_data = anomaly_data.cuda()
            normal_data = normal_data.cuda()

            predicts_anomaly = model(anomaly_data, N_BATCH)
            predicts_nomal = model(normal_data, N_BATCH)

            loss = criterion(
                predicts_anomaly,
                predicts_nomal,
                model.affine)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * anomaly_data.size(0) 
            total += anomaly_data.size(0)
            running_loss = loss_sum / total

            pbar.set_postfix({"loss":running_loss})
            pbar.update(1)

def predict(model, loader):
    model.eval()
    with torch.no_grad():
        for idx, (anomaly_data, _) in enumerate(loader):
            anomaly_data = anomaly_data.cuda()
            predict = model(anomaly_data, 1)
            print(f"{idx}: {predict.mean().item()}")


if __name__ == "__main__":
    model = MyNet()
    if MODEL_PATH:
        model.load_state_dict(torch.load(MODEL_PATH))
    model.cuda()

    dataset = DataSet()
    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=N_BATCH,
        shuffle=True,
        num_workers=N_WORKER)
    testloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=N_WORKER)

    for epoch in range(N_EPOCH):
        train(model, trainloader)
        torch.save(model.state_dict(), f'model.pt')
        predict(model, testloader)