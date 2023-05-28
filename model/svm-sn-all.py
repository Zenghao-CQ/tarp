from torch.functional import F
from sklearn import svm
# import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from dgl.data.utils import load_graphs
from sklearn.metrics import f1_score,accuracy_score

if __name__ == '__main__':
    glist,l = load_graphs('../dgl_graph/ss/no.g')
    gl2,l = load_graphs('../dgl_graph/ss/1.g')
    glist.extend(gl2)
    gl2,l = load_graphs('../dgl_graph/ss/2.g')
    glist.extend(gl2)
    gl2,l = load_graphs('../dgl_graph/ss/3.g')
    glist.extend(gl2)
    gl2,l = load_graphs('../dgl_graph/ss/4.g')
    glist.extend(gl2)
    gl2,l = load_graphs('../dgl_graph/ss/5.g')
    glist.extend(gl2)
    gl2,l = load_graphs('../dgl_graph/ss/6.g')
    glist.extend(gl2)    
    mask=np.zeros([40])
    mask[0]=1
    mask[1]=1
    mask[2]=1
    mask[3]=1
    mask[4]=1
    mask[5]=1
    # mask=glist[0].ndata["mask"].bool()
    # random.shuffle(glist)
    x=[]
    y=[]
    for g in glist:
        n=g.ndata["N"]
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
    accall =accuracy_score(result,y)
        # 进行评估
    
    # print("F-score: {0:.3f}".format(f1_score(result,yt,average='micro')))
    print("F-score: {0:.3f}".format(accall))