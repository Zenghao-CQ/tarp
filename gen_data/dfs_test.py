import copy

a={"a":2,"b":4,"c":3}
ks=list(a.keys())
vs=list(a.values())
ln=len(a)
print(ln)
print(len(ks))
print(len(vs))
def dfs(i,ks,vs):
    print(i)
    if i==ln:
        d=dict(zip(ks,vs))
        print(d)
        return
    vsp=copy.deepcopy(vs)
    v=vs[i]
    v-=0.5
    while v>=0.5:
        vsp[i]=v
        dfs(i+1,ks,vsp)
        v-=0.5
dfs(0,ks,vs)
