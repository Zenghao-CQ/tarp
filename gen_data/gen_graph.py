from .parse_json import parse_json 
from .config import mslist,qalist,basepath,stdd

def gen_graph(load:int, prof:dict):
    ms2newid={}
    lb=[0 for  i in range(len(mslist))]
    for idx, ms in enumerate(mslist):
        ms2newid[ms]=idx
    path=basepath+'/'+str(load)
    filepath=path+'/'+'fine.json'
    fn,fnd,fe,fd,ftp=parse_json(filepath) #nodes,nodes_delay, edges, delay
    on=fn
    ond=fnd
    for k,v in prof.items():
        if k not in qalist:
            continue
        if v*1000 >= fn[ms2newid[k]][0]: #if ms cpu above fine
            continue
        lb[ms2newid[k]]=1 #resource not enough
        ms=k.split('-')[1]
        filepath=path+'/'+ms+'_'+str(float(v))+'.json'
        n,nd,e,d,tp=parse_json(filepath)
        score=0
        for a,b in zip(fd,d):
            if a<=b:
                score+=1
            else:
                score-=1
        if score > 0:
            on=n
            ond=nd
            fe=e
            fd=d
            ftp=tp
    for k,v in prof.items():
        on[ms2newid[k]][0]=v
    if (fd[0]<stdd[0]) and (fd[1]<stdd[1]) and (fd[2]<stdd[2]):
        lb=[0 for  i in range(len(mslist))]

    return on,ond,fe,fd,lb,ftp #nodes,nodes_delay, edges, delay, labels,throughput


if __name__ == '__main__':
    # prof = {}
    # for ms in qalist:
    #     prof[ms]=0.5
    prof={
        "ts-travel-service":3,
		"ts-seat-service":1.5,
		"ts-basic-service":3,
		"ts-station-service":3,
		"ts-train-service":2,
    }
    n,nd,e,d,lb = gen_graph(40, prof)
    # print(prof)
    print(n)
    print(nd)
    print(e)
    print(n)
    print(lb)