import random
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def x_update_scale_value(x,p):
    return str(int(x)*4)+'m'
plt.rc('font', family="Times New Roman")
plt.rcParams["figure.figsize"] = (12.8,6.5)
minn = 9
maxn = 81


stage1=4
stage2=14
stage3=25

buffer=[]

for i in range(40):
    load = random.uniform(15,25)
    buffer.append(load)

for i in range(stage1,stage1+2):
    load = random.uniform(67,83)
    buffer[i]=load
for i in range(stage2,stage2+2):
    load = random.uniform(45,53)
    buffer[i]=load
for i in range(stage3,stage3+2):
    load = random.uniform(75,85)
    buffer[i]=load
    

plt.cla()
plt.tick_params(labelsize=15)
plt.plot(buffer,color="black")  # total rewards in an iteration or episode
plt.xlabel('Time(minutes)',fontsize=18)
plt.ylabel('Requests per second',fontsize=18)
plt.gca().xaxis.set_major_formatter(FuncFormatter(x_update_scale_value))
plt.show()
# plt.savefig('load-b.png')
# f2=open("load-b.csv",mode='a')
# f2.write(",".join([str(x) for x in buffer]))
# f2.close()
