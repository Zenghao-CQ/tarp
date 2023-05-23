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
    print(len(x))
    for i in range(5):
        t=sum(x[i*100:i*100+100])/100
        print(t)
    return x
    print("---------------------")
# x=readx("../data-process/firm/tp-100-1.csv")
# x=readx("../data-process/firm/tp-100-2.csv")
# x=readx("../data-process/sinan/avg.csv")
x=readx("../tp-cnn/logs/tp-80.csv")
x=readx("../tp-cnn/logs/tp-160.csv")
x=readx("../tp-cnn/logs/avg-160.csv")
