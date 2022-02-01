import sys
import torch
import torch.nn as nn
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

sys.path.append("../..")

from video import Extractor

RESNET_PRETRAINED = "./model_best.pth.tar"

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)

    def forward(self, images):
        return self.resnet(images).unsqueeze(1)

def img2tensor(path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(path).convert("RGB")
    return transform(image)

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
    outputs = []
    for grp, df_grp in tqdm(df.groupby("grp")):
        extractor = Extractor(df_grp["path"].tolist(), df_grp["label"].tolist(), net, img2tensor, cuda=args.gpu)
        features = extractor.extract()
        outputs.append(features)
    
    print("faetures_size: ", outputs[0].features.size())
    torch.save(outputs, args.output_path)