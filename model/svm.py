from torch.functional import F
from sklearn import svm
# import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from dgl.data.utils import load_graphs

if __name__ == '__main__':
    glist,l = load_graphs('../dgl_graph/40.g')
    # gl2,l = load_graphs('../dgl_graph/10.g')
    # glist.extend(gl2)
    mask=glist[0].ndata["mask"].bool()
    random.shuffle(glist)
    x=[]
    y=[]
    for g in glist:
        n=g.ndata["N_DELAY"]
        l=g.ndata["label"]
        for i in range(len(n)):
            if mask[i]:
                x.append(n[i])
                y.append(l[i])
    y=torch.tensor(y)
    x=np.array([item.detach().numpy() for item in x])
    y=np.array(y)
    
    ln=int(len(glist)*0.8)
    xt=x[ln:]
    yt=y[ln:]

    x=x[:ln]
    y=y[:ln]
    # x=x.numpy()
    # y=y.numpy()

    predictor = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
    # 进行训练
    predictor.fit(x, y)
    # 预测结果
    result = predictor.predict(xt)
    # 进行评估
    from sklearn.metrics import f1_score,accuracy_score
    
    # print("F-score: {0:.2f}".format(f1_score(result,y,average='micro')))
    print("F-score: {0:.2f}".format(accuracy_score(result,yt)))