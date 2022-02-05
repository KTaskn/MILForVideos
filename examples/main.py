
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from milforvideo.mil import MIL
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse
from milforvideo.video import VideoFeature
from typing import List

N_BATCH = 30
N_WORKER = 5
N_EPOCH = 500

# 分割数
V = 1
torch.manual_seed(3407)

class MyAffine(nn.Module):
    def __init__(self, input_size):
        print(f"input_size: {input_size}")
        super().__init__()
        # 学習モデル
        self.activation = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 1)

        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.6)

    def forward(self, x):
        x = self.dropout1(self.activation(self.layer1(x)))
        x = self.dropout2(self.activation(self.layer2(x)))
        return self.sigmoid(self.layer3(x))
    
class DataSet(torch.utils.data.Dataset):
    def __init__(self, video_features_normal: List[VideoFeature], video_features_anomalous: List[VideoFeature]):
        self.video_features_normal = video_features_normal
        self.video_features_anomalous = video_features_anomalous

    def __len__(self):
        return len(self.video_features_anomalous)

    def __getitem__(self, idx):
        normal_idx = np.random.randint(0, len(self.video_features_normal))
        video_normal = self.video_features_normal[normal_idx]
        video_anomalous = self.video_features_anomalous[idx]
        
        n_normal = video_normal.features.size(0) // V
        n_anomalous = video_anomalous.features.size(0) // V
        return (
            torch.stack([
                video_normal.features[num:num+n_normal].mean(dim=0)
                for num in range(0, video_normal.features.size(0), n_normal)
            ]),
            torch.stack([
                video_anomalous.features[num:num+n_anomalous].mean(dim=0)
                for num in range(0, video_anomalous.features.size(0), n_anomalous)
            ]),
        )
    

def train(model, loader, gpu=True, lambda1=8e-5, lambda2=8e-5, lambda3=0.01):
    model = model.cuda() if gpu else model
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = MIL(lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)

    loss_sum = 0
    total = 0
    model.train()
    with tqdm(total=len(loader), unit="batch") as pbar:
        for features_normal, features_anomalous in loader:
            features_anomalous = features_anomalous.cuda() if gpu else features_anomalous
            features_normal = features_normal.cuda() if gpu else features_normal
            
            loss = 0.0
            for idx in range(features_anomalous.size(2)):
                predicts_anomalous = model(features_anomalous[:, :, idx])
                predicts_normal = model(features_normal[:, :, idx])

                loss += criterion(
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
    return model.cpu() if gpu else model
    
def evaluate(model, features, labels, gpu=True):
    model = model.cuda() if gpu else model
    model.eval()
    with torch.no_grad():
        features = features.cuda() if gpu else features
        predicts = torch.stack([model(features[:, idx]) for idx in range(features.size(1))]).squeeze(2)
        predicts = predicts.cpu().numpy() if gpu else predicts.numpy()
        labels = labels.cpu().numpy() if gpu else labels.numpy()
    predicts = predicts.max(axis=0)
    
    return roc_auc_score(labels, predicts)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("normal_path", help="Normal data path", type=str)
    parser.add_argument("anomalous_path", help="Anomaly data path", type=str)
    parser.add_argument("lambda1", type=float)
    parser.add_argument("lambda2", type=float)
    parser.add_argument("lambda3", type=float)
    parser.add_argument("--gpu", action='store_true', help="Use GPU")
    
    args = parser.parse_args()    
    print(f"normal_path: {args.normal_path}")
    print(f"anomalous_path: {args.anomalous_path}")
    print(f"lambda1: {args.lambda1}")
    print(f"lambda2: {args.lambda2}")
    print(f"lambda3: {args.lambda3}")
    print(f"epoch: {N_EPOCH}")
    print(f"gpu: {args.gpu}")
    
    
    list_videos_normal: List[VideoFeature] = torch.load(args.normal_path)
    list_videos_anomalous: List[VideoFeature] = torch.load(args.anomalous_path)
    
    # Fit size of feature
    model = MyAffine(input_size=list_videos_normal[0].features[0].size(-1))

    dataset = DataSet(list_videos_normal, list_videos_anomalous)
    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=N_BATCH,
        shuffle=True,
        num_workers=N_WORKER)

    for epoch in range(N_EPOCH):
        model = train(model, trainloader, gpu=args.gpu, lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3)
    anom_auc = evaluate(model, 
                        torch.cat([
                            video.features
                            for video in list_videos_anomalous
                        ]),
                        torch.cat([
                            video.labels
                            for video in list_videos_anomalous
                        ]),
                        gpu=args.gpu)  
    anom_auc = anom_auc if anom_auc > 0.5 else 1.0 - anom_auc     
    
    all_auc = evaluate(model, torch.cat([
        torch.cat([
            video.features
            for video in list_videos_normal
        ]),           
        torch.cat([
            video.features
            for video in list_videos_anomalous
        ]),
    ]), torch.cat([
        torch.cat([
            video.labels
            for video in list_videos_normal
        ]),           
        torch.cat([
            video.labels
            for video in list_videos_anomalous
        ]),
    ]), gpu=args.gpu)
    all_auc = all_auc if all_auc > 0.5 else 1.0 - all_auc     
    print(f"AUC score (anomalous): {anom_auc}")
    print(f"AUC score (normal & anomalous): {all_auc}") 
    print("")