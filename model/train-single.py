import dgl
from dgl.data.utils import load_graphs
from gcn import GCN
import torch.nn.functional as F
import torch.nn as nn
import torch as th
glist,l = load_graphs('../dgl_graph/40.g')
print("len: ",len(glist))
g=glist[0]
#设置参数
in_feats = g.ndata["N"].shape[1]
h_feats = 16
num_class = (th.max(g.ndata["label"]) + 1).item() #或者 num_class = dataset.num_classes
# 创建模型
model = GCN(in_feats, h_feats, num_class)

def train(g, model, learning_rate=0.01, num_epoch=100):
    optimizer = th.optim.Adam(model.parameters(), lr = learning_rate)
    best_val_acc = 0
    best_test_acc = 0
    
    features = g.ndata["N"]
    labels = g.ndata["label"]
    train_mask = g.ndata["mask"].bool()
    test_mask = g.ndata["mask"].bool()
    val_mask = g.ndata["mask"].bool()
    # test_mask = g.ndata["test_mask"]
    # val_mask = g.ndata["val_mask"]
    
    for epoch in range(num_epoch):
        result = model(g, features)
        pred = result.argmax(1)
        
        loss = F.cross_entropy(result[train_mask], labels[train_mask])
        
        train_acc = (pred[train_mask]==labels[train_mask]).float().mean()
        val_acc  = (pred[val_mask]==labels[val_mask]).float().mean()        
        test_acc  = (pred[test_mask]==labels[test_mask]).float().mean()
        
        if best_val_acc < val_acc:
            best_val_acc, best_test_acc = val_acc, test_acc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print('In epoch {}, loss: {}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                epoch, loss, val_acc, best_val_acc, test_acc, best_test_acc))
        
    
if __name__ == "__main__":
    train(g, model, num_epoch=20, learning_rate=0.002)
    # print(glist)