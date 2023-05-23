from build_dgl_graph import build_dgl_graph
import torch as th
import dgl
from dgl.data.utils import save_graphs
import copy
import random


glist=[]
def dfs(i,ks,vs,load):
    if i==ln:
        d=dict(zip(ks,vs))
        # print(d)
        g,d=build_dgl_graph(load,d)# fine graph
        glist.append(g)
        # print(g.ndata['label'])
        return
    vsp=copy.deepcopy(vs)
    v=vs[i]
    while v>=0.5:
        vsp[i]=v
        dfs(i+1,ks,vsp,load)
        v-=0.5

if __name__=="__main__":
    prof_bd={
        "ts-travel-service":3.5,
		"ts-seat-service":1.5,
		"ts-basic-service":3.5,
		"ts-station-service":3.5,
		"ts-train-service":1.5,
    }
    load=40
    # prof_bd={
    #     "ts-travel-service":2,
	# 	"ts-seat-service":1.5,
	# 	"ts-basic-service":2,
	# 	"ts-station-service":2,
	# 	"ts-train-service":1,
    # }
    # load=10
    g,d=build_dgl_graph(load,prof_bd)# fine graph
    for i in range(int(len(glist)/10)):
        glist.append(g)
    random.shuffle(glist)
    
    ks=list(prof_bd.keys())
    vs=list(prof_bd.values())
    ln=len(prof_bd)
    dfs(0,ks,vs,load)
    save_graphs('../dgl_graph/40.g',glist)
