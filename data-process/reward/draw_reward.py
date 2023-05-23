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
plt.plot(x,marker="x",color="tomato",label="Ours")
plt.plot(x1,marker="*",color="royalblue",label="Firm")
plt.plot(x2,marker="*",color="darkviolet",label="Sinan")

plt.tick_params(labelsize=12)
# plt.plot(x,y,color="tomato",label="Ours")
# plt.plot(x1,y1,color="royalblue",label="Firm")
# plt.plot(x3,y3,color="darkviolet",label="Sinan")
plt.legend()
plt.xlabel('Time inteval',fontsize=14)
plt.ylabel('CDF',fontsize=14)
plt.show()
# plt.legend(prop={'size':16})
# plt.savefig("./cdf.png")
