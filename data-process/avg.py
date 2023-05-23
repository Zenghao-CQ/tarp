import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random

plt.rcParams["figure.figsize"] = (12.8,6)
plt.rc('font', family="Times New Roman")
def readx(filename):
    f=open(filename)
    s=f.readline()
    x=s.split(',')
    x=[float(i) for i in x]
    f.close()
    print(filename)
    for i in range(5):
        t=sum(x[i*100:i*100+100])/100
        print(t)
    return x
x=readx("./ours/resource.csv")
x=readx("./firm/resource-1.csv")
x=readx("./sinan/resource.csv")
