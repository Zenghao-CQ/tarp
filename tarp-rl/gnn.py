from dgl.nn import GraphConv
import torch.nn as nn
import torch.nn.functional as F
import torch as th

class GNN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GNN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats[0],allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats[0], h_feats[1],allow_zero_in_degree=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(h_feats[1],h_feats[2])
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        h = self.fc(h)
        return h