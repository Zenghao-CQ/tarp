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
g,d,tp,SLO_score,res_use=env.get_state()
print(d,tp)
# g,reward,d,SLO_score,res_use=env.perform_action(prof)
# g1,reward,d,SLO_score,res_use=env.perform_action(prof)
# g2,reward,d,SLO_score,res_use=env.perform_action(prof)

# allg=dgl.batch([g,g1,g2])

# actor = Actor(5,2,13)
# critic = Critic(5,13)
# a = actor(allg)
# print(a)
# c = critic(allg,a)
# d=torch.FloatTensor([1,2,3])
# print(c)
# op= optim.Adam(critic.parameters(), lr=0.001)
# loss= nn.MSELoss()
# ll=loss(c,d)
# ll.backward(retain_graph=True)
# op.step()