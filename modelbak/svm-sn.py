from torch.functional import F
from sklearn import svm
# import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from dgl.data.utils import load_graphs
from sklearn.metrics import f1_score,accuracy_score
mask_num=0
batch_size=20

def valid(g_test,predictor):
    acc=.0
    for g in g_test:
        n=g.ndata["N_DELAY"]
        l=g.ndata["label"]
        x=[]
        y=[]
        flag=0
        for i in range(len(n)):
            if mask[i]:
                x.append(n[i])
                y.append(l[i])
                if l[i]==torch.tensor([1]):
                    flag+=1
        if flag==mask_num or flag==0:
            continue
        y=torch.tensor(y)
        x=np.array([item.detach().numpy() for item in x])
        y=np.array(y)
        # x=x.numpy()
        # y=y.numpy()
        # 预测结果
        result = predictor.predict(x)
        acc+=accuracy_score(result,y)
    print("Valid ACC: {0:.2f}".format(acc/len(g_test)))


if __name__ == '__main__':
    glist=[]
    # glist,l = load_graphs('../dgl_graph/ss/no.g')
    # glist=glist[200:]
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
    for i in mask:
        if i==True:
            mask_num+=1
    random.shuffle(glist)
    
    ln=int(len(glist)*0.8)
    g_train=glist[:ln]
    g_test=glist[ln:]
    predictor = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')

    for i in range(100):
        random.shuffle(g_train)
        print("Epoch",i)
        gs=[g_train[i:i+batch_size] for i in range(0,len(g_train),batch_size)]
        for batch in gs:
            x=[]
            y=[]
            for idx,g in enumerate(batch):
                n=g.ndata["N_DELAY"]
                l=g.ndata["label"]
                flag=0
                for i in range(len(n)):
                    if mask[i]:
                        x.append(n[i])
                        y.append(l[i])
                        if l[i]==torch.tensor([1]):
                            flag+=1
            # if flag==mask_num*len(batch) or flag==0:
            #     continue
            y=torch.tensor(y)
            x=np.array([item.detach().numpy() for item in x])
            y=np.array(y)
            # x=x.numpy()
            # y=y.numpy()
            # 进行训练
            predictor.fit(x, y)
            # 预测结果
            result = predictor.predict(x)
            # 进行评估    
            # print("F-score: {0:.2f}".format(f1_score(result,y,average='micro')))
    valid(g_test,predictor)