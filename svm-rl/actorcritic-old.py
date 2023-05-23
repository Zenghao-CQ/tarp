import torch 
import torch.nn as nn
import dgl
from dgl.nn import GraphConv
import numpy as np
from torch.autograd import Variable
from gnn import GNN

def fanin_init(size, fanin=None):
    """Utility function for initializing actor and critic"""
    fanin = fanin or size[0]
    w = 1./ np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-w, w)

HID_LAYER1 = 40
HID_LAYER2 = 24
HID_LAYER3 = 24
GCONV1=24
GCONV2=24
WFINAL = 0.003

class Actor(nn.Module):    
    def __init__(self, stateDim,resDim, actionDim): #state Dim = Node feature size

        super(Actor, self).__init__()
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.resDim = resDim

        self.norm0 = nn.BatchNorm1d(self.stateDim)
                
        self.fc1 = nn.Linear(self.stateDim, HID_LAYER1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())   

        self.conv1 = GraphConv(HID_LAYER1, GCONV1,allow_zero_in_degree=True)
        self.conv1.weight.data = fanin_init(self.conv1.weight.data.size())

        self.conv2 = GraphConv(GCONV1, GCONV2,allow_zero_in_degree=True)
        self.conv2.weight.data = fanin_init(self.conv2.weight.data.size())

        self.bn1 = nn.BatchNorm1d(GCONV2)
                                    
        self.fc2 = nn.Linear(GCONV2, HID_LAYER2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
                     
        
        self.normres = nn.BatchNorm1d(self.resDim)
        self.fcres = nn.Linear(self.resDim, HID_LAYER3)
        self.fcres.weight.data.uniform_(-WFINAL, WFINAL)
                                               
        self.bn2 = nn.BatchNorm1d(HID_LAYER2+HID_LAYER3)

        self.fc3 = nn.Linear(HID_LAYER2+HID_LAYER3, self.actionDim)
        self.fc3.weight.data.uniform_(-WFINAL, WFINAL)

        self.ReLU = nn.ReLU(inplace=False)
        self.Tanh = nn.Tanh()
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, g): #ip delay+N, ipres-> resource ie. N
        ip = g.ndata["N_DELAY"]
        ipres = g.ndata["N"]
        ip = torch.cat([ip, ipres],dim=1)

        ip_norm = self.norm0(ip)    
        #print("@@@@")
        #print(ip_norm.shape)                        
        h1 = self.ReLU(self.fc1(ip_norm))
        #print(h1.shape)                        
        g1 = self.conv1(g, h1)
        #print(g1.shape)                        
        g2 = self.conv2(g, g1)
        #print(g2.shape)                        
        h1_norm = self.bn1(g2)
        hres = self.fcres(self.normres(ipres))
        ht=self.ReLU(self.fc2(h1_norm))
        h2 = torch.cat([ht, hres],dim=1)
        h2_norm = self.bn2(h2)
        #print(h2.shape)                        
        # action = self.Tanh((self.fc3(h2_norm)))
        action = self.Softmax(self.fc3(h2_norm))
        #print(action.shape)                        
        return action
        
class Critic(nn.Module):
    def __init__(self, stateDim, actionDim):
        super(Critic, self).__init__()
        self.stateDim = stateDim
        self.actionDim = actionDim
        
        self.fc1 = nn.Linear(self.stateDim, HID_LAYER1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.bn1 = nn.BatchNorm1d(HID_LAYER1)
        
        self.conv1 = GraphConv(HID_LAYER1, GCONV1,allow_zero_in_degree=True)
        self.conv1.weight.data = fanin_init(self.conv1.weight.data.size())
        
        self.fc2 = nn.Linear(GCONV1 + self.actionDim, HID_LAYER2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        
        self.conv2 = GraphConv(HID_LAYER2, GCONV2,allow_zero_in_degree=True)
        self.conv2.weight.data = fanin_init(self.conv2.weight.data.size())

        self.fc3 = nn.Linear(GCONV2, 1)
        self.fc3.weight.data.uniform_(-WFINAL, WFINAL)
        
        self.ReLU = nn.ReLU(inplace=False)
        
    def forward(self, g, action): # add N add N_delay
        ip = g.ndata["N_DELAY"]
        ipres = g.ndata["N"]
        ip = torch.cat([ip, ipres],dim=1)

        h1 = self.ReLU(self.fc1(ip))
        h1_norm = self.bn1(h1)
        #print("#####")
        #print(h1_norm.shape)
        g1 = self.conv1(g, h1_norm)
        #print(g1.shape)
        h2 = self.ReLU(self.fc2(torch.cat([g1, action], dim=1)))
        #print(h2.shape)
        g2 = self.conv2(g, h2)
        #print(g2.shape)
        g.ndata["TEMP"]=g2
        gall = dgl.mean_nodes(g,'TEMP') # mean all nodes features for every subgraph
        Qval = self.fc3(gall) 
        return Qval.squeeze()