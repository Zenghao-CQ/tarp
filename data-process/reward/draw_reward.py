import matplotlib.pyplot as plt
import random
import scipy.signal
plt.rc('font', family="Times New Roman")

f=open("./reward.csv")
s=f.readline()
graph1=s.split(',')
graph1=[float(x) for x in graph1]
plt.cla()
plt.tick_params(labelsize=13)
plt.plot(graph1, color='bisque')  # total rewards in an iteration or episode
# plt.plot(avg_rewards, color='b')  # (moving avg) rewards
tmp = scipy.signal.savgol_filter(graph1, 55, 3)
plt.plot(tmp, color='darkorange')  # total rewards in an iteration or episode
plt.xlabel('Episodes',fontsize=15)
plt.ylabel('Reward',fontsize=15)
# plt.show()
plt.savefig('./reward.png')