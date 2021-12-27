import torch
import torch.nn as nn
from tqdm import tqdm
from i3d import InceptionI3d
from dataset import DataSet
import argparse

PRETRAINED_PATH = "./rgb_i3d_pretrained.pt"
N_BATCHES = 5
N_WORKERS = 5

# Number of frames per feature
F = 16

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.i3d = InceptionI3d()
        self.i3d.load_state_dict(torch.load(PRETRAINED_PATH))

    def forward(self, video):
        video = video.transpose_(1, 2)
        return self.i3d(video).squeeze(2)
    
def extract(dataset, net, n_batches=N_BATCHES, n_workers=N_WORKERS, cuda=False):
    net = net.cuda() if cuda else net
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=n_batches,
        shuffle=False,
        num_workers=n_workers)

    outputs = []
    with tqdm(total=len(loader), unit="batch") as pbar:
        for videos in loader:
            videos = videos.cuda() if cuda else videos
            predict = net(videos)
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
    
    ds = DataSet(paths_image, F=F)

    # You can change the model here
    net = MyNet()
    with torch.no_grad():
        outputs = extract(ds, net, cuda=args.cuda)
        
    ds_labels = DataSet(labels, F=F, func_extract=lambda X: 1 if sum(X) > 0 else 0)
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