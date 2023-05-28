import torch
from torch.functional import F
# import matplotlib.pyplot as plt
from dgl.data.utils import load_graphs
 
 
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
 
    def forward(self, x_input):
        x_hidden = F.relu(self.hidden(x_input))
        x_predict = self.predict(x_hidden)
        return x_predict
 
 
if __name__ == '__main__':
    # load data
    # n_data = torch.ones(100, 2)
    # x0 = torch.normal(2 * n_data, 1)
    # y0 = torch.zeros(100)
    # x1 = torch.normal(-2 * n_data, 1)
    # y1 = torch.ones(100)
    # x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
    # y = torch.cat((y0, y1)).type(torch.LongTensor)
    
    glist,l = load_graphs('../dgl_graph/40.g')
    # gl2,l = load_graphs('../dgl_graph/10.g')
    # glist.extend(gl2)
    mask=glist[0].ndata["mask"].bool()
    x=[]
    y=[]
    for g in glist:
        n=g.ndata["N_DELAY"]
        l=g.ndata["label"]
        for i in range(len(n)):
            if mask[i]:
                x.append(n[i])
                y.append(l[i])
    x=torch.tensor([item.detach().numpy() for item in x])
    x=x.to(torch.float)
    # y=torch.tensor([item.detach().numpy() for item in y])
    y=torch.tensor(y)
 
    # net
    net = Net(g.ndata["N_DELAY"].shape[1], 4, 2)
    # print(net)
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    loss_func = torch.nn.CrossEntropyLoss()
 
    # train
    # plt.ion()
    for step in range(100):
        out = net(x)
        loss = loss_func(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 2 == 0:
            # plt.cla()
            prediction = torch.max(F.softmax(out, dim=1), 1)[1]
            pred_y = prediction.data.numpy().squeeze()
            target_y = y.data.numpy()
            acc = sum(pred_y == target_y) / len(pred_y)
            # plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0)
            # plt.text(1.5, -4, 'Acc = %.2f' % acc, fontdict={'size': 20, 'color': 'red'})
            # plt.pause(0.5)
            print("acc:",acc)
            pass
 
    print("ok")