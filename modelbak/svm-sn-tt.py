from torch.functional import F
from sklearn import svm
# import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from dgl.data.utils import load_graphs
from sklearn.metrics import f1_score,accuracy_score
import torch.nn.functional as F
mask_num=0
batch_size=40

def valid(g_test,predictor):
    acc=.0
    mask=g_test[0].ndata["mask"].bool()
    mask_num=0
    for i in mask:
        if i==True:
            mask_num+=1
    for g in g_test:
        n=g.ndata["N_DELAY"]
        F.normalize(n.float(),p=1,dim=-1)
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
    print("Valid ACC: {0:.4f}".format(acc/len(g_test)))


if __name__ == '__main__':
    glist,l = load_graphs('../dgl_graph/40.g')
    g_train = glist

    val_list=[]
    gl2,l = load_graphs('../dgl_graph/ms/1.g')
    val_list.extend(gl2[:100])
    gl2,l = load_graphs('../dgl_graph/ms/2.g')
    val_list.extend(gl2[:100])
    gl2,l = load_graphs('../dgl_graph/ms/3.g')
    val_list.extend(gl2[:100])
    gl2,l = load_graphs('../dgl_graph/ms/9.g')
    val_list.extend(gl2[:100])
    gl2,l = load_graphs('../dgl_graph/ms/11.g')
    val_list.extend(gl2[:100])
    random.shuffle(val_list)
    
    g_train,val_list=val_list,g_train
    mask=g_train[0].ndata["mask"].bool()
    for i in mask:
        if i==True:
            mask_num+=1
    random.shuffle(glist)
    

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
                F.normalize(n.float(),p=1,dim=-1)
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
    valid(val_list,predictor)
        