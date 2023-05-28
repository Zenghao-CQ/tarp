from torch.functional import F
from sklearn import svm
# import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from dgl.data.utils import load_graphs
from sklearn.metrics import f1_score,accuracy_score

if __name__ == '__main__':
    # glist,l = load_graphs('../dgl_graph/ss/no.g')
    # glist=glist[200:]
    # gl2,l = load_graphs('../dgl_graph/ss/1.g')
    # glist.extend(gl2[50:])
    glist=[]
    gl2,l = load_graphs('../dgl_graph/ms/1.g')
    glist.extend(gl2[100:])
    gl2,l = load_graphs('../dgl_graph/ms/2.g')
    glist.extend(gl2[100:])
    gl2,l = load_graphs('../dgl_graph/ms/3.g')
    glist.extend(gl2[100:])
    gl2,l = load_graphs('../dgl_graph/ms/9.g')
    glist.extend(gl2[100:])
    gl2,l = load_graphs('../dgl_graph/ms/11.g')
    glist.extend(gl2[100:])   
    mask=np.zeros([25])
    mask[1]=1
    mask[2]=1
    mask[3]=1
    mask[9]=1
    mask[11]=1
    # mask=glist[0].ndata["mask"].bool()
    print(mask)
    # random.shuffle(glist)
    accall=0
    for g in glist:
        x=[]
        y=[]
        n=g.ndata["N_DELAY"]
        l=g.ndata["label"]
        for i in range(len(n)):
            if mask[i]:
                x.append(n[i])
                y.append(l[i])
        y=torch.tensor(y)
        x=np.array([item.detach().numpy() for item in x])
        y=np.array(y)
        # print(y)
        
        # ln=int(len(glist)*0.8)
        # xt=x[ln:]
        # yt=y[ln:]

        # x=x[:ln]
        # y=y[:ln]
        # x=x.numpy()
        # y=y.numpy()

        predictor = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
        # 进行训练
        predictor.fit(x, y)
        # 预测结果
        result = predictor.predict(x)
        accall+=accuracy_score(result,y)
        # 进行评估
    
    # print("F-score: {0:.3f}".format(f1_score(result,yt,average='micro')))
    print("F-score: {0:.3f}".format(accall/len(glist)))