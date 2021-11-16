#import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as PyG
#from torch_geometric.transforms import Distance
#from torch_geometric.data import Data as PyGData
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, nFeats):
        super(GCN, self).__init__()
      
        self.conv1 = GCNConv(nFeats, 32)
        self.conv2 = GCNConv(32, 64)

        self.fc = nn.Sequential(
            nn.Linear( 64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.5),
            nn.Linear( 32, 1), nn.Softplus(),
        )
        
    def forward(self, data):
        #data.edge_index = torch.LongTensor([]).view(2,-1)
        x = self.conv1(data.x, data.edge_index)
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, data.edge_index)
        x = scatter_mean(x, data.batch, dim=0)
        out = self.fc(x)
        return out
