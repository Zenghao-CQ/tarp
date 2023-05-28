import dgl
from dgl.data.utils import load_graphs
from gcn import GCN
import torch.nn.functional as F
import torch.nn as nn
import torch as th
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt

def draw_fig(list,name,epoch):
    # 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
    x1 = range(1, epoch+1)
    print(x1)
    y1 = list
    if name=="loss":
        plt.cla()
        plt.title('Train loss vs. epoch', fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train loss', fontsize=20)
        plt.grid()
        plt.savefig("../lossAndacc-ss/Train_loss.png")
        plt.show()
    elif name =="acc":
        plt.cla()
        plt.title('Train accuracy vs. epoch', fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train accuracy', fontsize=20)
        plt.grid()
        plt.savefig("../lossAndacc-ss/Train_accuracy.png")
        plt.show()
    elif name =="val":
        plt.cla()
        plt.title('Valid accuracy vs. epoch', fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Vlaid accuracy', fontsize=20)
        plt.grid()
        plt.savefig("../lossAndacc-ss/Vlid_accuracy.png")
        plt.show()
        
        
        

def collate(graphs):
    # graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph

def eval(glist, model):
    val_acc=0
    for g in glist:
        features = g.ndata["N_DELAY"].to(th.float32)
        # features = th.cat((g.ndata["N"],g.ndata["N_DELAY"]),1)
        labels = g.ndata["label"]
        train_mask = g.ndata["mask"].bool()
        test_mask = g.ndata["mask"].bool()
        val_mask = g.ndata["mask"].bool()
        # test_mask = g.ndata["test_mask"]
        # val_mask = g.ndata["val_mask"]
        result = model(g, features)
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
            # normalize
            F.normalize(features.float(),p=1,dim=1)
            # features = th.cat((g.ndata["N"],g.ndata["N_DELAY"]),1)
            labels = g.ndata["label"].long()
            train_mask = g.ndata["mask"].bool()
            test_mask = g.ndata["mask"].bool()
            val_mask = g.ndata["mask"].bool()
            # test_mask = g.ndata["test_mask"]
            # val_mask = g.ndata["val_mask"]
            result = model(g, features)
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
    draw_fig(epoch_losses,"loss",num_epoch)
    draw_fig(epoch_acc,"acc",num_epoch)
    draw_fig(epoch_val,"val",num_epoch)

glist=[]
# glist,l = load_graphs('../dgl_graph/ss/no.g')
# glist=glist[200:]
# gl2,l = load_graphs('../dgl_graph/ss/1.g')
# glist.extend(gl2[50:])
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

random.shuffle(glist)
ln=int(len(glist)*0.8)
val_list=glist[ln:]
glist=glist[:ln]
data_loader = DataLoader(glist, batch_size=32, shuffle=True,
                         collate_fn=collate)

#设置参数
in_feats = glist[0].ndata["N_DELAY"].shape[1]
# in_feats = glist[0].ndata["N"].shape[1]+glist[0].ndata["N_DELAY"].shape[1]
h_feats = [16,16]
# num_class = (th.max(glist[0].ndata["label"]) + 1).item() #或者 num_class = dataset.num_classes
num_class = 2 #或者 num_class = dataset.num_classes
# 创建模型
model = GCN(in_feats, h_feats, num_class)


if __name__ == "__main__":
    th.manual_seed(3407)
    train(glist,val_list, model, num_epoch=200, learning_rate=0.00005)
    # g=glist[100]
    # print(g.ndata["N"])
    # print(g.ndata["label"])
    # print(g.edata["E"])