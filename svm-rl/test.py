import dgl
import torch
from dgl.nn import GraphConv
import torch.nn as nn
from dgl.data.utils import load_graphs
import random
from actorcritic import Actor, Critic
from client import Environment,LoadPattern
import torch.optim as optim
from gen_data.build_dgl_graph import build_dgl_graph,get_sub
prof={
    "ts-travel-service":2.0,
    "ts-seat-service":2.0,
    "ts-basic-service":2.0,
    "ts-station-service":2.0,
    "ts-train-service":2.0,
}

lg=LoadPattern("wave",{"max":80,"min":20,"period":40,"prop":0.07})
env=Environment(prof,loadPattern=lg)
g,d,SLO_score,res_use=env.get_state()
print(g.ndata["N"])
print(g.ndata["N_DELAY"])
x = g.ndata["N_DELAY"]
y = g.ndata["N"]
x=torch.cat([x,y],dim=1)
print(x)
g1,reward,d,SLO_score,res_use=env.perform_action(prof)
x2 = g.ndata["N_DELAY"]
y = g.ndata["N"]
x2=torch.cat([x2,y],dim=1)
# g2,reward,d,SLO_score,res_use=env.perform_action(prof)
x=torch.stack((x,x2),dim=0)
# allg=dgl.batch([g,g1,g2])

actor = Actor(5,12)
critic = Critic(5,12)
a = actor(x)
print(a)

c = critic(x,a)
print(c)

# op= optim.Adam(critic.parameters(), lr=0.001)
# loss= nn.MSELoss()
# ll=loss(c,d)
# ll.backward(retain_graph=True)
# op.step()