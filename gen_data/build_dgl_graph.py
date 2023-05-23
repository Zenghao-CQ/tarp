import dgl
import networkx as nx
import matplotlib.pyplot as plt
import torch as th
import random
from .gen_graph import gen_graph
from .config import mslist, qalist, edge_feature, sublist

n_num=11

def build_dgl_graph(load:int, prof:dict):
    n,nd,e,d,lb = gen_graph(load,prof) #node edge delay
    n_f=[[.0,.0] for i in range (n_num)] # use cpu_alloc and cpu_used
    nd_f = [[.0,.0] for i in range (n_num)] # delay for every node
    for i in range(n_num):
        if n[i][0]==-1:
            # n_f[i][0]=n[i][0] # if not limited, times 4 will be enough
            n_f[i][0]=n[i][2]*4 # if not limited, times 4 will be enough
        else:
            n_f[i][0]=n[i][0]
        n_f[i][1]=n[i][2]
    for k,v in nd.items():
        nd_f[k]=v
    for i, vec in enumerate(nd_f):
        for j, v in enumerate(vec):
            nd_f[i][j]+=random.uniform(0,v/100)
    n_f=th.tensor(n_f)
    nd_f=th.tensor(nd_f)
    n_mask=th.ones(n_num, dtype=th.bool)
    for idx,t in enumerate(mslist):
        if t not in qalist:
            n_mask[idx]=False
    e_s,e_d,e_f=[],[],[]
    for t in e:
        e_s.append(t['from'])
        e_d.append(t['to'])
        # e_f.append([t['rps'],t['throughtPut'],t['AVG'],t['P95'],t['P99']])
        e_f.append([float(t[x]) for x in edge_feature])
    e_s=th.tensor(e_s)
    e_d=th.tensor(e_d)
    e_f=th.tensor(e_f)
    # print(n_f)
    # print(n_mask)
    # print(e_s)
    # print(e_d)
    # print(e_f)
    g = dgl.graph((e_s,e_d))
    g.ndata['N']=n_f
    g.ndata['N_DELAY']=nd_f
    g.ndata['label']=th.tensor(lb)
    g.ndata['mask']=n_mask
    g.edata['E']=e_f
    # print(g)
    # nx_G = g.to_networkx()
    # pos = nx.spring_layout(nx_G,k = 0.004, iterations = 500, scale = 0.6)
    # nx.draw(nx_G, pos, with_labels=True, node_color=[[.6, .7, .7]], font_size = 6, alpha = 0.5)
    # plt.savefig("./a.jpg")
    # dgl.save_graphs()
    # return g,n_f,n_mask,e_s,e_f,e_d
    return g,d

sub_num = 5
def get_sub(load:int, prof:dict):
    n,nd,e,d,lb,tp = gen_graph(load,prof) #nodes,nodes_delay, edges, delay, labels
    n_f=[[.0,.0] for i in range (sub_num)] # use cpu_alloc and cpu_used
    nd_f = [[.0,.0] for i in range (n_num)] # delay for every node
    nd_f2 = [[.0,.0] for i in range (sub_num)] # delay for every node
    for i in range(sub_num):
        idx=sublist[i]
        n_f[i][0]=n[idx][0]
        n_f[i][1]=n[idx][2]
    for k,v in nd.items():
        if k in sublist:
            nd_f[k]=v
    for i, vec in enumerate(nd_f):
        for j, v in enumerate(vec):
            nd_f[i][j]+=random.uniform(0,(v/200)+1)
    for i in range(len(nd_f2)):
        nd_f2[i]=nd_f[sublist[i]]
    n_f=th.tensor(n_f)
    nd_f2=th.tensor(nd_f2)
    # n_mask=th.ones(sub_num, dtype=th.bool)
    # for idx,t in enumerate(mslist):
    #     if t not in qalist:
    #         n_mask[idx]=False
    e_s,e_d,e_f=[],[],[]
    old2sub={}
    for idx,i in enumerate(sublist):
        old2sub[i]=idx
    for t in e:
        if (t["from"]==9) and (t["to"]==7): #travel to ticketinfo
            e_s.append(4)
            e_d.append(0)
            e_f.append([float(t[x]) for x in edge_feature])
        if (t['from'] not in sublist) or (t['to'] not in sublist):
            continue
        e_s.append(old2sub[t['from']])
        e_d.append(old2sub[t['to']])
        # e_f.append([t['rps'],t['throughtPut'],t['AVG'],t['P95'],t['P99']])
        e_f.append([float(t[x]) for x in edge_feature])
    e_s=th.tensor(e_s)
    e_d=th.tensor(e_d)
    e_f=th.tensor(e_f)
    g = dgl.graph((e_s,e_d))
    g.ndata['N']=n_f
    g.ndata['N_DELAY']=nd_f2
    # g.ndata['label']=th.tensor(lb)
    # g.ndata['mask']=n_mask
    g.edata['E']=e_f
    return g,d,tp


if __name__ == '__main__':
    prof={
        "ts-travel-service":3,
		"ts-seat-service":1.5,
		"ts-basic-service":3.5,
		"ts-station-service":3.5,
		"ts-train-service":2,
    }
    # prof={}
    # for ms in qalist:
    #     prof[ms]=0.5
    load=40
    # g,n_f,n_mask,e_s,e_f,e_d = build_dgl_graph(load,prof)
    g,d = build_dgl_graph(load,prof)
    print(g.ndata["label"])
    print(g.ndata["N"])
    print(g.ndata["N_DELAY"])
    print(d)
