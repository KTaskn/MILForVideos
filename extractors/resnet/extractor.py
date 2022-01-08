import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import DataSet
import argparse
import pandas as pd

RESNET_PRETRAINED = "./model_best.pth.tar"
N_BATCHES = 1500
N_WORKERS = 10

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)

    def forward(self, images):
        return self.resnet(images)

def extract(dataset, net, n_batches=N_BATCHES, n_workers=N_WORKERS, cuda=False):
    net = net.cuda() if cuda else net
    net.eval()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=n_batches,
        shuffle=False,
        num_workers=n_workers)

    outputs, labels = [], []
    with torch.no_grad():
        with tqdm(total=len(loader), unit="batch") as pbar:
            for images, b_labels in loader:
                images = images.cuda() if cuda else images
                predict = net(images)
                predict = predict.cpu() if cuda else predict
                outputs.append(predict)
                labels.append(b_labels)
                pbar.update(1)
    return torch.cat(outputs), torch.cat(labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pathlist", help="Path to the dataset", type=str)
    parser.add_argument("output_path", help="Path to the output file", type=str)
    parser.add_argument("--gpu", action='store_true', help="Use GPU")
    
    args = parser.parse_args()
    print(f"pathlist: {args.pathlist}")
    print(f"output_path: {args.output_path}")
    print(f"gpu: {args.gpu}")
    
    # Get the image path and label from a input file
    with open(args.pathlist) as f:
        grp_path_and_label = [
            row.split(" ")
            for row in f.read().split("\n")
            if row
        ]        
        df = pd.DataFrame({
            "grp": [int(grp) for grp, _, _ in grp_path_and_label],
            "path": [path for _, path, _ in grp_path_and_label],
            "label": [int(label) for _, _, label in grp_path_and_label],
        })
    
    # You can change the model here
    net = MyNet()
    outputs, labels = [], []
    for grp, df_grp in df.groupby("grp"):
        ds = DataSet(
            df_grp["path"].tolist(),
            df_grp["label"].tolist())

        o, l = extract(ds, net, cuda=args.gpu)
    
        o = o.unsqueeze(1)
        outputs.append(o)
        labels.append(l)
                
    outputs = torch.cat(outputs)
    labels = torch.cat(labels)
        
    print(f"features_size: {outputs.size()}")
    print(f"labels_size: {labels.size()}")
    
    torch.save({
        "features": outputs,
        "labels": labels
    }, args.output_path)