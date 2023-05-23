import dgl
from dgl.data.utils import load_graphs
from cnn import CNN
import torch.nn.functional as F
import torch.nn as nn
import torch as th
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt

def collate(graphs):
    # graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph

def eval(glist, model):
    val_acc=0
    for g in glist:
        features = g.ndata["N_DELAY"].to(th.float32)
        features = th.cat([features,g.ndata["N"]],dim=1)
        # normalize
        features = features.unsqueeze(1)
        F.normalize(features.float(),p=1,dim=-1)
        # features = th.cat((g.ndata["N"],g.ndata["N_DELAY"]),1)
        labels = g.ndata["label"].long()
        train_mask = g.ndata["mask"].bool()
        test_mask = g.ndata["mask"].bool()
        val_mask = g.ndata["mask"].bool()
        # test_mask = g.ndata["test_mask"]
        # val_mask = g.ndata["val_mask"]
        result = model(features)
        pred = result.argmax(1)
        val_acc  += (pred[val_mask]==labels[val_mask]).float().mean()
    return val_acc/len(glist)
        

def train(glist, val_list, model, learning_rate=0.01, num_epoch=20):
    optimizer = th.optim.Adam(model.parameters(), lr = learning_rate)
    best_val_acc = 0
    best_test_acc = 0
    
    epoch_losses=[]
    epoch_acc=[]
    epoch_val=[]
    for epoch in range(num_epoch):
        epoch_loss=0
        train_acc=0
        val_acc=0
        # test_acc=0
        for iter, g in enumerate(data_loader):
            # features = g.ndata["N"]
            features = g.ndata["N_DELAY"].to(th.float32)
            features = th.cat([features,g.ndata["N"]],dim=1)
            # normalize
            features = features.unsqueeze(1)
            F.normalize(features.float(),p=1,dim=-1)
            # features = th.cat((g.ndata["N"],g.ndata["N_DELAY"]),1)
            labels = g.ndata["label"].long()
            # print("shape",features.shape)
            train_mask = g.ndata["mask"].bool()
            test_mask = g.ndata["mask"].bool()
            val_mask = g.ndata["mask"].bool()
            # test_mask = g.ndata["test_mask"]
            # val_mask = g.ndata["val_mask"]
            result = model(features)
            pred = result.argmax(1)
            
            loss = F.cross_entropy(result[train_mask], labels[train_mask])
            optimizer.zero_grad()            
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.detach().item()
            train_acc += (pred[train_mask]==labels[train_mask]).float().mean()
            # val_acc  += (pred[val_mask]==labels[val_mask]).float().mean()        
            # test_acc  += (pred[test_mask]==labels[test_mask]).float().mean()
            
            # if best_val_acc < val_acc:
            #     best_val_acc, best_test_acc = val_acc, test_acc
        epoch_loss /=(iter+1)
        train_acc /=(iter+1)
        val_acc =eval(val_list,model)
        test_acc =val_acc
        if epoch % 1 == 0:
            print('In epoch {}, loss: {},train acc: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                epoch, epoch_loss,train_acc, val_acc, best_val_acc, test_acc, best_test_acc))
        epoch_losses.append(epoch_loss)
        epoch_acc.append(train_acc)
        epoch_val.append(val_acc)


glist=[]
gl2,l = load_graphs('../dgl_graph/ms/1.g')
glist.extend(gl2)
gl2,l = load_graphs('../dgl_graph/ms/2.g')
glist.extend(gl2)
gl2,l = load_graphs('../dgl_graph/ms/3.g')
glist.extend(gl2)
gl2,l = load_graphs('../dgl_graph/ms/9.g')
glist.extend(gl2)
gl2,l = load_graphs('../dgl_graph/ms/11.g')
glist.extend(gl2)
random.shuffle(glist)

val_list=[]
gl2,l = load_graphs('../dgl_graph/ss/2.g')
val_list.extend(gl2[:50])
gl2,l = load_graphs('../dgl_graph/ss/3.g')
val_list.extend(gl2[:50])
gl2,l = load_graphs('../dgl_graph/ss/4.g')
val_list.extend(gl2[:50])
gl2,l = load_graphs('../dgl_graph/ss/5.g')
val_list.extend(gl2[:50])
gl2,l = load_graphs('../dgl_graph/ss/6.g')
val_list.extend(gl2[:50])
random.shuffle(val_list)


data_loader = DataLoader(glist, batch_size=32, shuffle=True,
                         collate_fn=collate)

#设置参数
in_feats = glist[0].ndata["N_DELAY"].shape[1]+glist[0].ndata["N"].shape[1]
# in_feats = glist[0].ndata["N"].shape[1]+glist[0].ndata["N_DELAY"].shape[1]
h_feats = [16,16,8]
kernels = [4,4]
# num_class = (th.max(glist[0].ndata["label"]) + 1).item() #或者 num_class = dataset.num_classes
num_class = 2 #或者 num_class = dataset.num_classes
# 创建模型
model = CNN(in_feats, h_feats, kernels, num_class)


if __name__ == "__main__":
    # th.manual_seed(3407)
    # print(len(glist))
    train(glist,val_list, model, num_epoch=150, learning_rate=0.0005)
    # g=glist[100]
    # print(glist[0].ndata["N_DELAY"])
    # print(g.ndata["label"])
    # print(g.edata["E"])