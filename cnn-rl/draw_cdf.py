import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
plt.rc('font', family="Times New Roman")
numb=25
f=open("./Ours-delay.csv")
s=f.readline()
x=s.split(',')
x=[float(i) for i in x]
x=list(filter(lambda i: i<1000,x))
f.close()

res = stats.relfreq(x, numbins=numb)
x = res.lowerlimit + np.linspace(0, res.binsize*res.frequency.size,res.frequency.size)
y=np.cumsum(res.frequency)

f=open("./FIRM-delay.csv")
s1=f.readline()
x1=s1.split(',')
x1=[float(i) for i in x1]
x1=list(filter(lambda i: i<4000,x1))
f.close()

res1 = stats.relfreq(x1, numbins=numb)
x1 = res1.lowerlimit + np.linspace(0, res1.binsize*res1.frequency.size,res1.frequency.size)
y1=np.cumsum(res.frequency)

plt.plot(x,y,marker="x",color="tomato",label="Ours")
plt.plot(x1,y1,marker="*",color="aquamarine",label="Firm")
plt.legend()
plt.xlabel('End to End delay(ms)')
plt.ylabel('CDF')
# plt.show()
plt.savefig("./cdf.png")
