import torch.nn as nn

class FCL(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # FC Layer
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
    