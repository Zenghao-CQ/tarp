import dgl
from dgl.data.utils import load_graphs
from gcn import GCN
import torch.nn.functional as F
import torch.nn as nn
import torch as th
from torch.utils.data import DataLoader


def collate(graphs):
    # graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph


def train(glist, model, learning_rate=0.01, num_epoch=100):
    optimizer = th.optim.Adam(model.parameters(), lr = learning_rate)
    best_val_acc = 0
    best_test_acc = 0
    g=glist[0]
    features = g.ndata["N"]
    labels = g.ndata["label"]
    train_mask = g.ndata["mask"].bool()
    test_mask = g.ndata["mask"].bool()
    val_mask = g.ndata["mask"].bool()
    # test_mask = g.ndata["test_mask"]
    # val_mask = g.ndata["val_mask"]
    
    epoch_losses=[]
    for epoch in range(num_epoch):
        epoch_loss=0
        train_acc=0
        val_acc=0
        test_acc=0

        for iter, g in enumerate(data_loader):
            result = model(g, features)
            pred = result.argmax(1)
            
            loss = F.cross_entropy(result[train_mask], labels[train_mask])
            optimizer.zero_grad()            
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.detach.item
            train_acc += (pred[train_mask]==labels[train_mask]).float().mean()
            val_acc  += (pred[val_mask]==labels[val_mask]).float().mean()        
            test_acc += (pred[test_mask]==labels[test_mask]).float().mean() 
            
            if best_val_acc < val_acc:
                best_val_acc, best_test_acc = val_acc, test_acc
        epoch_loss /=(iter+1)
        train_acc /=(iter+1)
        val_acc /=(iter+1)
        test_acc /=(iter+1)
        if epoch % 5 == 0:
            print('In epoch {}, loss: {}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                epoch, epoch_loss, val_acc, best_val_acc, test_acc, best_test_acc))
        epoch_losses.append(epoch_loss)


glist,l = load_graphs('../dgl_graph/40.g')

data_loader = DataLoader(glist, batch_size=32, shuffle=True,
                         collate_fn=collate)

#设置参数
in_feats = glist[0].ndata["N"].shape[1]
h_feats = 16
num_class = (th.max(glist[0].ndata["label"]) + 1).item() #或者 num_class = dataset.num_classes
# 创建模型
model = GCN(in_feats, h_feats, num_class)


if __name__ == "__main__":
    train(glist, model, num_epoch=20, learning_rate=0.002)
    # print(glist)