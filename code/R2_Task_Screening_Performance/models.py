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
        x = x.float()
        x1 = self.hidden_activation(self.fc1(x))
        x1 = self.drop(x1)
        y = self.fc2(x1)
        y = self.sig(y)
        return y
    
 
class MultiViewLinearProbe(nn.Module):
    def __init__(self, n_views, n_features, hidden_dim, drop_prob):
        super(MultiViewLinearProbe,self).__init__()
        self.n_views = n_views
        self.fc1_modules = nn.ModuleList([nn.Linear(in_features=n_features, out_features=hidden_dim, bias=True) for _ in range(n_views)])
        self.drop = nn.Dropout(p = drop_prob)
        self.fc2_modules = nn.ModuleList([nn.Linear(in_features=self.fc1_modules[0].out_features, out_features=1, bias=True) for _ in range(n_views)])
        self.hidden_activation = nn.ReLU()
        self.fc3 = nn.Linear(in_features=n_views, out_features=1, bias=True)
        self.sig = nn.Sigmoid()
 
    def forward(self, x):
        # print(x.shape)  # (batch_size, n_views, n_features)

        # x is supposed to be of shape (batch_size, n_views, n_features)
        x = x.float()

        for i in range(self.n_views):
            x_view = x[:, i, :]  # Extract the i-th view (batch_size, n_features)
            # print(x_view.shape) # (256, 768)

            # Get a candidate prediction for this view
            x1_view = self.hidden_activation(self.fc1_modules[i](x_view))
            # print(x1_view.shape) # (256, 512)
            x1_view = self.drop(x1_view)
            y_view = self.fc2_modules[i](x1_view)
            # print(y_view.shape)
            y_view = self.sig(y_view) # (256, 1)

            # Merge the predictions from different views
            if i == 0:
                y = y_view
            else:
                y = torch.cat((y, y_view), dim=1)  # Concatenate along feature dimension

        # Neural late fusion
        # print(y.shape) # (256, n_views)
        y = self.fc3(y)
        # print(y.shape)    # (256, 1)
        y = self.sig(y)
        return y