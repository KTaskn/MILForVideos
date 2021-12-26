import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from mil import MIL
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse

N_BATCH = 5
N_WORKER = 5
N_EPOCH = 100

V = 32


class MyAffine(nn.Module):
    def __init__(self):
        super().__init__()
        # 学習モデル
        self.activation = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.layer1 = nn.Linear(1024, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 1)

        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.6)

    def forward(self, x):
        x = self.dropout1(self.activation(self.layer1(x)))
        x = self.dropout2(self.activation(self.layer2(x)))
        return self.sigmoid(self.layer3(x))
    
class DataSet(torch.utils.data.Dataset):
    def __init__(self, feature_normal, feature_anomalous, label_normal, label_anomalous):
        self.feature_normal = feature_normal
        self.feature_anomalous = feature_anomalous
        self.label_normal = label_normal
        self.label_anomalous = label_anomalous

    def __len__(self):
        length = self.feature_normal.size(0) if self.feature_anomalous.size(0) > self.feature_normal.size(0) else self.feature_anomalous.size(0)
        return length - V + 1

    def __getitem__(self, idx):
        return (            
            torch.stack([self.feature_normal[idx + num] for num in range(V)]),
            torch.stack([self.feature_anomalous[idx + num] for num in range(V)]),
            torch.stack([self.label_normal[idx + num] for num in range(V)]),
            torch.stack([self.label_anomalous[idx + num] for num in range(V)]),
        )
    

def train(model, loader):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = MIL()

    loss_sum = 0
    total = 0
    model.train()
    with tqdm(total=len(loader), unit="batch") as pbar:
        for features_normal, features_anomalous, _, _ in loader:
            features_anomalous = features_anomalous.cuda()
            features_normal = features_normal.cuda()

            predicts_anomalous = model(features_anomalous)
            predicts_normal = model(features_normal)

            loss = criterion(
                predicts_anomalous,
                predicts_normal,
                model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * features_anomalous.size(0) 
            total += features_anomalous.size(0)
            running_loss = loss_sum / total

            pbar.set_postfix({"loss":running_loss})
            pbar.update(1)
    return model
    
def predict(model, features, labels):
    model.eval()
    with torch.no_grad():
        features = features.cuda()
        predicts = model(features).cpu().numpy()
        labels = labels.cpu().numpy()
    
    print(f"roc: {roc_auc_score(labels, predicts)}")


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("normal_path", help="", type=str)
    parser.add_argument("anomalous_path", help="", type=str)
    
    args = parser.parse_args()
    
    model = MyAffine()
    model = model.cuda()
    
    dict_normal = torch.load(args.normal_path)
    dict_anomalous = torch.load(args.anomalous_path)

    dataset = DataSet(dict_normal["features"], dict_anomalous["features"], dict_normal["labels"], dict_anomalous["labels"])
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
        predict(model, dict_anomalous["features"], dict_anomalous["labels"])
        model = train(model, trainloader)