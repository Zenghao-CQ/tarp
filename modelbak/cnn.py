from dgl.nn import GraphConv
import torch.nn as nn
import torch.nn.functional as F
import torch as th

class CNN(nn.Module):
    def __init__(self, in_feats, h_feats, kenels, num_class):#in [batch,features, 1]
        super(CNN, self).__init__()
        self.fcin = nn.Linear(in_feats, h_feats[0])
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=0)
        
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=h_feats[1], kernel_size=kenels[0]),
                          nn.LeakyReLU(),
                          nn.MaxPool1d(kernel_size=4),
                          ) #in [batch,100, ?]
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=h_feats[1], out_channels=h_feats[2], kernel_size=kenels[1]),
                          nn.LeakyReLU(),
                          nn.MaxPool1d(kernel_size=4),
                          )#in [batch,50, ?]
        # 全连接层
        self.fc = nn.Linear(336, num_class)

    def forward(self,in_feat): #in [batch,1,features]
        # print("000",in_feat.shape)
        h= self.fcin(in_feat)
        # print("111",h.shape)
        h = self.conv1(h)
        h = F.relu(h)
        h = h.view(h.size(0),-1)
        # print("222",h.shape)
        h = self.fc(h)
        # print("333",h.shape)
        h = self.softmax(h)
        return h