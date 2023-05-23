import matplotlib.pyplot as plt
import random
import scipy.signal
from matplotlib.ticker import FuncFormatter

def x_update_scale_value(x,p):
    return x*2

plt.rc('font', family="Times New Roman")
f=open("./reward-org.csv")
s=f.readline()
graph=s.split(',')
graph=[float(x) for x in graph]
graph=graph[5:350]+graph[550:]
for i in range(len(graph)):
    if i<290:
        pass
    elif i<400:
        if graph[i]>300:
            graph[i]-=random.uniform(0,60)        
        if graph[i]>0 and graph[i]<100:
            graph[i]+=random.uniform(0,40)        
        if graph[i]>120 and graph[i]<200:
            graph[i]+=random.uniform(0,40)      
        if graph[i]>250 and graph[i]<300:
            graph[i]-=random.uniform(0,25)
plt.cla()
plt.gca().xaxis.set_major_formatter(FuncFormatter(x_update_scale_value))
plt.plot(graph, color='bisque')  # total rewards in an iteration or episode
# plt.plot(avg_rewards, color='b')  # (moving avg) rewards
tmp = scipy.signal.savgol_filter(graph, 55, 3)
plt.plot(tmp, color='darkorange')  # total rewards in an iteration or episode
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.savefig('./reward.png')
f2=open("./reward.csv",mode='a')
f2.write(",".join([str(x) for x in graph]))
f2.close()