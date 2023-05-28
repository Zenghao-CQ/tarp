import dgl
from dgl.data.utils import load_graphs
from gcn import GCN
import torch.nn.functional as F
import torch.nn as nn
import torch as th
glist,l = load_graphs('../dgl_graph/40.g')
for g in glist:
    print(g.ndata["label"])