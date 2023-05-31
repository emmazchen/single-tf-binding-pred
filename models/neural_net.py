import torch
from torch import nn

class NeuralNet(nn.Module):
    def __init__(self, model_kwargs):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(**model_kwargs['l1_kwargs'])
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(**model_kwargs['l2_kwargs'])
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(**model_kwargs['l3_kwargs']) #outputs logits -> BCEWithLogitsLoss applies sigmoid

    def forward(self, x):
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        x = self.relu2(x)
        out = self.l3(x)
        return out
