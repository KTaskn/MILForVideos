import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import DataSet
import argparse

RESNET_PRETRAINED = "./model_best.pth.tar"
N_BATCHES = 5
N_WORKERS = 5

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)

    def forward(self, images):
        return self.resnet(images)

def extract(dataset, net, n_batches=N_BATCHES, n_workers=N_WORKERS, cuda=False):
    net = net.cuda() if cuda else net
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=n_batches,
        shuffle=False,
        num_workers=n_workers)

    outputs = []
    with tqdm(total=len(loader), unit="batch") as pbar:
        for images in loader:
            images = images.cuda() if cuda else images
            predict = net(images)
            predict = predict.cpu() if cuda else predict
            outputs.append(predict)
            pbar.update(1)
    return torch.cat(outputs)

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
        path_and_label = [
            row.split(" ")
            for row in f.read().split("\n")
            if row
        ]
        paths_image = [path for path, _ in path_and_label]
        labels = [int(label) for _, label in path_and_label]
    
    ds = DataSet(paths_image)

    # You can change the model here
    net = MyNet()
    with torch.no_grad():
        outputs = extract(ds, net, cuda=args.gpu)
        
    ds_labels = DataSet(labels, func_extract=lambda x: x)
    labels = torch.tensor([
        ds_labels.__getitem__(idx)
        for idx in range(ds_labels.__len__())
    ])
        
    print(f"features_size: {outputs.size()}")
    print(f"labels_size: {labels.size()}")
    
    torch.save({
        "features": outputs,
        "labels": labels
    }, args.output_path)