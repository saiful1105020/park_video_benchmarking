import torch
from torch import nn

class SingleViewLinearProbe(nn.Module):
    def __init__(self, n_features, hidden_dim, drop_prob):
        super(SingleViewLinearProbe,self).__init__()
        self.fc1 = nn.Linear(in_features=n_features, out_features=hidden_dim, bias=True)
        self.drop = nn.Dropout(p = drop_prob)
        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=1, bias=True)
        self.hidden_activation = nn.ReLU()
        self.sig = nn.Sigmoid()
 
    def forward(self, x):
        x1 = self.hidden_activation(self.fc1(x))
        x1 = self.drop(x1)
        y = self.fc2(x1)
        y = self.sig(y)
        return y
 