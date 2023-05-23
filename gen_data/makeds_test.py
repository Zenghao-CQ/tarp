from build_dgl_graph import build_dgl_graph
import torch as th
import dgl
from dgl.data.utils import save_graphs
import copy
import random


glist=[]

if __name__=="__main__":
    prof_bd={
        "ts-travel-service":3.5,
		"ts-seat-service":1.5,
		"ts-basic-service":3.5,
		"ts-station-service":3.5,
		"ts-train-service":1.5,
    }
    load=40
    for k,v in prof_bd.items():
        v-=0.5
        while v>0.5:
            for i in range(30):
                prof=copy.deepcopy(prof_bd)
                prof[k]=v
                g,d=build_dgl_graph(load,prof)# fine graph
                glist.append(g)
            v-=0.5
    for i in range(200):
        g,d=build_dgl_graph(load,prof_bd)# fine graph
        glist.append(g)
    random.shuffle(glist)
    
    # ks=list(prof_bd.keys())
    # vs=list(prof_bd.values())
    # ln=len(prof_bd)
    # dfs(0,ks,vs,load)
    print(len(glist))
    save_graphs('../dgl_graph/40-test.g',glist)
