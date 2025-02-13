from torch import nn
import numpy as np
import torch

class Classification_net(nn.Module):
    def __init__(self, in_features,hidden_size):
        super(Classification_net, self).__init__()
        self.layer1=nn.Linear(in_features,hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)
        self.act=torch.nn.LeakyReLU()
        #self.layer1 = nn.Linear(in_features,1)

    def forward(self, x):
        print("\033[0;31;40mclass net output\033[0m")
        x=self.layer1(x)
        x=self.act(x)
        x=self.layer2(x)


        #print(x)
        return x

# class Classification_net_multi(nn.Module):
#     def __init__(self, in_features,hidden_size,layers=None):
#         super(Classification_net_multi, self).__init__()
#         self.layer1_1 = nn.Linear(in_features,hidden_size)
#         self.layer1_2 = nn.Linear(hidden_size, hidden_size)
#
#         self.layer_diag_1 = nn.Linear(hidden_size, hidden_size)
#         self.layer_diag_2 = nn.Linear(hidden_size, 1)
#
#         self.layer_sa_1 = nn.Linear(hidden_size, hidden_size)
#         self.layer_sa_2 = nn.Linear(hidden_size, 1)
#
#         self.layer_rbb_1 = nn.Linear(hidden_size, hidden_size)
#         self.layer_rbb_2 = nn.Linear(hidden_size, 1)
#
#         self.act=torch.nn.LeakyReLU()
#         #self.layer1 = nn.Linear(in_features,1)
#
#     def forward(self, x):
#         print("\033[0;31;40mclass net output\033[0m")
#         x = self.layer1_1(x)
#         x = self.act(x)
#         x = self.layer1_2(x)
#         x = self.act(x)
#
#         x_diag = self.layer_diag_1(x)
#         x_diag = self.act(x_diag)
#         x_diag = self.layer_diag_2(x_diag)
#
#         x_sa = self.layer_sa_1(x)
#         x_sa = self.act(x_sa)
#         x_sa = self.layer_sa_2(x_sa)
#
#         x_rbb = self.layer_rbb_1(x)
#         x_rbb = self.act(x_rbb)
#         x_rbb = self.layer_rbb_2(x_rbb)
#
#         return x_diag, x_sa, x_rbb
#



class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, m):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        in_dim = input_size
        for _ in range(m):
            self.layers.append(nn.Linear(in_dim, hidden_size))
            self.layers.append(nn.ReLU())
            in_dim = hidden_size
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# class Classification_net_multi(nn.Module):
#     def __init__(self, in_features):
#         super(Classification_net_multi, self).__init__()
#         mid_size=256
#         hidden_size=512
#         self.rMLP=MLP(in_features,hidden_size,mid_size,m=4)
#         self.layer_diag_1 = nn.Linear(mid_size, mid_size)
#         self.layer_diag_2 = nn.Linear(mid_size, 1)
#
#         self.layer_sa_1 = nn.Linear(mid_size, mid_size)
#         self.layer_sa_2 = nn.Linear(mid_size, mid_size)
#         self.layer_sa_3 = nn.Linear(mid_size, 1)
#
#         self.layer_rbb_1 = nn.Linear(mid_size, mid_size)
#         self.layer_rbb_2 = nn.Linear(mid_size, mid_size)
#         self.layer_rbb_3 = nn.Linear(mid_size, 1)
#
#         self.act=torch.nn.LeakyReLU()
#         #self.layer1 = nn.Linear(in_features,1)
#
#     def forward(self, x):
#         x=self.rMLP(x)
#         x_diag = self.layer_diag_1(x)
#         x_diag = self.act(x_diag)
#         x_diag = self.layer_diag_2(x_diag)
#
#         x_sa = self.layer_sa_1(x)
#         x_sa = self.act(x_sa)
#         x_sa = self.layer_sa_2(x_sa)
#         x_sa = self.act(x_sa)
#         x_sa = self.layer_sa_3(x_sa)
#
#         x_rbb = self.layer_rbb_1(x)
#         x_rbb = self.act(x_rbb)
#         x_rbb = self.layer_rbb_2(x_rbb)
#         x_rbb = self.act(x_rbb)
#         x_rbb = self.layer_rbb_3(x_rbb)
#
#         return x_diag, x_sa, x_rbb
#
class Classification_net_multi(nn.Module):
    def __init__(self, in_features):
        super(Classification_net_multi, self).__init__()
        mid_size=256
        hidden_size=512
        self.rMLP=MLP(in_features,hidden_size,mid_size,m=4)
        self.layer_diag_1 = nn.Linear(mid_size, mid_size)
        self.layer_diag_2 = nn.Linear(mid_size, 1)

        self.saMLP = MLP(mid_size, mid_size, 1, m=4)
        self.rbbMLP = MLP(mid_size, mid_size, 1, m=4)

        self.act=torch.nn.LeakyReLU()
        #self.layer1 = nn.Linear(in_features,1)

    def forward(self, x):
        x=self.rMLP(x)
        x_diag = self.layer_diag_1(x)
        x_diag = self.act(x_diag)
        x_diag = self.layer_diag_2(x_diag)

        x_sa= self.saMLP(x)
        x_rbb = self.rbbMLP(x)

        return x_diag, x_sa, x_rbb