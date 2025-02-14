import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.linear import Linear
from torch_geometric.nn import GCNConv, RGCNConv
from torch_geometric.data import Data


class Projection(nn.Module):

    def __init__(self, input_dim, hid_dim):
        super(Projection, self).__init__()
        self.fc1 = Linear(input_dim, hid_dim)
        self.fc2 = Linear(hid_dim, hid_dim)
        self.act_fn = nn.LeakyReLU()
        self.layernorm = nn.LayerNorm(hid_dim, eps=1e-6)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.layernorm(x)
        x = self.fc2(x)
        return x


class GCN(nn.Module):

    def  __init__(self, nfeat, nhid):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid, bias=True)
        self.conv2 = GCNConv(nhid, nhid, bias=True)

    def forward(self, data):
        try:
            x = self.conv1(data.x, data.edge_index, data.edge_weight)
            x = F.leaky_relu(x)
            x = self.conv2(x, data.edge_index, data.edge_weight)
        except:
            x = self.conv1(data.x, data.edge_index)
            x = F.leaky_relu(x)
            x = self.conv2(x, data.edge_index)
        return x


class RGCN(nn.Module):

    def __init__(self, num_nodes, nhid, num_rels, num_bases=None):
        super(RGCN, self).__init__()
        self.rconv1 = RGCNConv(num_nodes, nhid, num_rels, bias=True, num_bases=num_bases)
        self.rconv2 = RGCNConv(nhid, nhid, num_rels, bias=True, num_bases=num_bases)

    def forward(self, data):
        x = self.rconv1(None, data.edge_index, data.edge_type)
        x = F.leaky_relu(x)
        x = self.rconv2(x, data.edge_index, data.edge_type)
        return x
