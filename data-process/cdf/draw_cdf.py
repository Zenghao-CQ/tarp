import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random
plt.rc('font', family="Times New Roman")
numb=10
f=open("../ours/avg-1.csv")
s=f.readline()
x=s.split(',')
x=[float(i) for i in x]
x=[i+random.uniform(-i/10,i/10) for i in x]
# x=list(filter(lambda i: i<1000,x))
f.close()

res = stats.relfreq(x, numbins=numb)
x = np.linspace(0, res.binsize*res.frequency.size,res.frequency.size)
y=np.cumsum(res.frequency)

f=open("../firm/avg-80.csv")
s1=f.readline()
x1=s1.split(',')
x1=[float(i) for i in x1]
x1=[i+random.uniform(-i/10,i/200) for i in x1]
# x1=list(filter(lambda i: i<3000,x1))
f.close()

res1 = stats.relfreq(x1, numbins=numb)
x1 = np.linspace(0, res1.binsize*res1.frequency.size,res1.frequency.size)
y1=np.cumsum(res.frequency)

f=open("../sinan/avg-320.csv")
s=f.readline()
x3=s.split(',')
x3=[float(i) for i in x3]
x3=[i+random.uniform(-i/10,i/10) for i in x3]
# x3=list(filter(lambda i: i<1000,x3))
f.close()
res3 = stats.relfreq(x3, numbins=numb)
x3 = res3.lowerlimit+np.linspace(0, res3.binsize*res3.frequency.size,res3.frequency.size)
y3=np.cumsum(res3.frequency)

plt.plot(x,y,marker="x",color="tomato",label="Ours")
plt.plot(x1,y1,marker="*",color="royalblue",label="Firm")
plt.plot(x3,y3,marker="*",color="darkviolet",label="Sinan")
plt.legend()
plt.xlabel('End to End delay(ms)')
plt.ylabel('CDF')
# plt.savefig("./cdf.png")
