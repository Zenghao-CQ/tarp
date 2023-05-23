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
    f.close()
    return x

x=readx("../ours/avg.csv",0)
x1=readx("../firm/avg-1.csv",0)
x2=readx("../sinan/avg.csv",0)

plt.tick_params(labelsize=12)
plt.plot(x,color="tomato",label="Ours")
plt.plot(x1,color="royalblue",label="Firm")
plt.plot(x2,color="darkviolet",label="Sinan")
plt.legend()
plt.xlabel('Time interval',fontsize=14)
plt.ylabel('End to end latency(ms)',fontsize=14)
# plt.show()
# plt.legend(prop={'size':16})
plt.savefig("./wave.png")
