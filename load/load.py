import random
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def x_update_scale_value(x,p):
    return str(int(x)*2)+'m'
plt.rc('font', family="Times New Roman")
plt.rcParams["figure.figsize"] = (12.8,6.5)
minn = 9
max2 = 30
max3 = 81

stage1=14
stage2=36
stage3=66
stage4=80

buffer=[]

delta = (max3-minn)/(stage1-1)
for i in range(stage1):
    load=10+delta*i+random.uniform(-5,5)
    print(i,load)
    buffer.append(load)

for i in range(stage1,stage2):
    load=max3+random.uniform(-5,5)
    print(i,load)
    buffer.append(load)

delta = (max3-max2)/(stage3-stage2-1)
for i in range(stage2,stage3):
    load=max3-delta*(i-stage2)+random.uniform(-5,5)
    print(i,load)
    buffer.append(load)
delta = (max2-minn)/(stage4-stage3-1)
for i in range(stage3,stage4):
    load=max2-delta*(i-stage3)+random.uniform(-3,3)
    print(i,load)
    buffer.append(load)

plt.cla()
plt.tick_params(labelsize=15)
plt.plot(buffer,color="black")  # total rewards in an iteration or episode
plt.xlabel('Time(minutes)',fontsize=18)
plt.ylabel('Requests per second',fontsize=18)
plt.gca().xaxis.set_major_formatter(FuncFormatter(x_update_scale_value))
plt.show()
# plt.savefig('load.png')
# f2=open("load1.csv",mode='a')
# f2.write(",".join([str(x) for x in buffer]))
# f2.close()
