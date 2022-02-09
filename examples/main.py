
import torch
import torch.optim as optim
from tqdm import tqdm
from milforvideo.model import FCL
from milforvideo.mil import MIL
from milforvideo.video import generate_dataloader
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse
from milforvideo.video import VideoFeature
from typing import List

N_BATCH = 30
N_WORKER = 5
N_EPOCH = 1

torch.manual_seed(3407)

def train(model, loader, gpu=True, lambda1=8e-5, lambda2=8e-5, lambda3=0.01):
    model = model.cuda() if gpu else model
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = MIL(lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)

    loss_sum = 0
    total = 0
    model.train()
    with tqdm(total=len(loader), unit="batch") as pbar:
        for features_normal, features_anomalous, labels_normal, labels_anomalous in loader:
            features_normal = features_normal.cuda() if gpu else features_normal
            features_anomalous = features_anomalous.cuda() if gpu else features_anomalous
            labels_normal = labels_normal.cuda() if gpu else labels_normal
            labels_anomalous = labels_anomalous.cuda() if gpu else labels_anomalous
            
            print(features_normal.size(), labels_normal.size())
            
            loss = 0.0
            for idx in range(features_anomalous.size(2)):
                filter_ = labels_anomalous[:, :, idx] > -1
                # analysing on each cluster
                predicts_anomalous = model(features_anomalous[filter_, :, idx])
                predicts_normal = model(features_normal[filter_, :, idx])

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
    return model.cpu() if gpu else model, running_loss
    
def evaluate(model, features, labels, gpu=True):
    model = model.cuda() if gpu else model
    model.eval()
    with torch.no_grad():
        features = features.cuda() if gpu else features
        predicts = torch.stack([model(features[:, idx]) for idx in range(features.size(1))]).squeeze(2)
        predicts = predicts.cpu().numpy() if gpu else predicts.numpy()
        labels = labels.cpu().numpy() if gpu else labels.numpy()
    predicts = predicts.max(axis=0)
    
    return roc_auc_score(labels[:, 0], predicts)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("normal_path", help="Normal data path", type=str)
    parser.add_argument("anomalous_path", help="Anomaly data path", type=str)
    parser.add_argument("lambda1", type=float)
    parser.add_argument("lambda2", type=float)
    parser.add_argument("lambda3", type=float)
    parser.add_argument("V", type=int)
    parser.add_argument("--gpu", action='store_true', help="Use GPU")
    
    args = parser.parse_args()    
    print(f"normal_path: {args.normal_path}")
    print(f"anomalous_path: {args.anomalous_path}")
    print(f"lambda1: {args.lambda1}")
    print(f"lambda2: {args.lambda2}")
    print(f"lambda3: {args.lambda3}")
    print(f"V: {args.V}")
    print(f"epoch: {N_EPOCH}")
    print(f"gpu: {args.gpu}")
    
    
    video_features_normal: List[VideoFeature] = torch.load(args.normal_path)
    video_features_anomalous: List[VideoFeature] = torch.load(args.anomalous_path)
        
    # Fit size of feature
    model = FCL(input_size=video_features_normal[0].features[0].size(-1))

    trainloader = generate_dataloader(
        video_features_normal,
        video_features_anomalous,
        batch_size=N_BATCH,
        shuffle=True,
        num_workers=N_WORKER,
        V=args.V)

    for epoch in range(N_EPOCH):
        model, running_loss = train(model, trainloader, gpu=args.gpu, lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3)
        anom_auc = evaluate(model, 
                            torch.cat([
                                video.features
                                for video in video_features_anomalous
                            ]),
                            torch.cat([
                                video.labels
                                for video in video_features_anomalous
                            ]),
                            gpu=args.gpu)  
        anom_auc = anom_auc if anom_auc > 0.5 else 1.0 - anom_auc     
        
        all_auc = evaluate(model, torch.cat([
            torch.cat([
                video.features
                for video in video_features_normal
            ]),           
            torch.cat([
                video.features
                for video in video_features_anomalous
            ]),
        ]), torch.cat([
            torch.cat([
                video.labels
                for video in video_features_normal
            ]),           
            torch.cat([
                video.labels
                for video in video_features_anomalous
            ]),
        ]), gpu=args.gpu)
        all_auc = all_auc if all_auc > 0.5 else 1.0 - all_auc     
        print(f"{running_loss},{anom_auc},{all_auc}")