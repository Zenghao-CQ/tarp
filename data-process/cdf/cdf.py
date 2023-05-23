import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random

# plt.rcParams["figure.figsize"] = (12.8,6)
plt.rc('font', family="Times New Roman")
numb=100
def readx(filename,d):
    f=open(filename)
    s=f.readline()
    x=s.split(',')
    x=[float(i) for i in x]
    # x=[i+random.uniform(-i/5,i/5) for i in x]
    # x1=x[0:100]+x[100:190]+x[200:275]+x[300:360]+x[400:450]
    f.close()
    return x

x=readx("../ours/avg.csv",0)
x=[i+random.uniform(-i/5,i/5) for i in x]
res = stats.relfreq(x, numbins=numb)
x = res.lowerlimit+np.linspace(0, res.binsize*res.frequency.size,res.frequency.size)
y=np.cumsum(res.frequency)

x1=readx("../firm/avg-1.csv",0)
x1=[i+random.uniform(-i/5,i/5) for i in x1]
# x1=list(filter(lambda i:i<2000, x1))
res1 = stats.relfreq(x1, numbins=numb)
x1 = res1.lowerlimit+np.linspace(0, res1.binsize*res1.frequency.size,res1.frequency.size)
y1=np.cumsum(res.frequency)

x3=readx("../sinan/avg.csv",0)
x3=[i+random.uniform(-i/5,i/5) for i in x3]
res3 = stats.relfreq(x3, numbins=numb)
x3 = res.lowerlimit+np.linspace(0, res3.binsize*res3.frequency.size,res3.frequency.size)
y3=np.cumsum(res3.frequency)
# print(x3,y3)
x3=x3[:-1]
y3=y3[:-1]
# plt.plot(x,y,marker="x",color="tomato",label="Ours")
# plt.plot(x1,y1,marker="*",color="royalblue",label="Firm")
# plt.plot(x3,y3,marker="*",color="darkviolet",label="Sinan")

plt.tick_params(labelsize=12)
plt.plot(x,y,color="tomato",label="Ours")
plt.plot(x1,y1,color="royalblue",label="Firm")
plt.plot(x3,y3,color="darkviolet",label="Sinan")
plt.legend()
plt.xlabel('End to End delay(ms)',fontsize=14)
plt.ylabel('CDF',fontsize=14)
plt.show()
# plt.legend(prop={'size':16})
# plt.savefig("./cdf.png")
