import matplotlib.pyplot as plt
import random
import scipy.signal
plt.rc('font', family="Times New Roman")

f=open("./acc.csv")
s=f.readline()
graph=s.split(',')
graph=[float(x) for x in graph]
f.close()

d=(0.87-0.82)/100
plt.cla()
plt.plot(graph, color='turquoise')  # total rewards in an iteration or episode
# plt.plot(avg_rewards, color='b')  # (moving avg) rewards
# tmp = scipy.signal.savgol_filter(graph, 55, 3)
# plt.plot(tmp, color='darkorange')  # total rewards in an iteration or episode
plt.tick_params(labelsize=12)
plt.xlabel('Time(minutes)',fontsize=14)
plt.ylabel('Requests per second',fontsize=14)
plt.xlabel('Episodes')
plt.ylabel('Accuracy')
# plt.show()
plt.savefig('./acc.png')
# f2=open("./reward.csv",mode='a')
# f2.write(",".join([("%.4f" % x) for x in graph]))
# f2.close()