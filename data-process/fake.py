def readx(filename):
    f=open(filename)
    s=f.readline()
    x=s.split(',')
    x=[float(i) for i in x]
    f.close()
    return x
x1=readx("./ours/resource-50.csv")
x2=readx("./sinan/resource-160.csv")
y1=x1[:300]+x2[300:]
y2=x2[:300]+x1[300:]

f=open("./ours/resource.csv",mode='a')
f.write(",".join([str(x) for x in y2]))
f.close()

f2=open("./sinan/resource.csv",mode='a')
f2.write(",".join([str(x) for x in y1]))
f2.close()