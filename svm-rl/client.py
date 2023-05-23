#['ts-basic-service', 'ts-config-service', 'ts-order-service', 'ts-price-service', 'ts-route-service', 'ts-seat-service', 'ts-station-service', 'ts-ticketinfo-service', 'ts-train-service', 'ts-travel-service', 'ts-ui-dashboard']
import sys
import random
import matplotlib.pyplot as plt
import torch
sys.path.append("..")
# from ..gen_data.build_dgl_graph import build_dgl_graph
from gen_data.build_dgl_graph import build_dgl_graph,get_sub

load_range=[10,20,40,60,80]
ALPHA=0.65

default_prof={
    'ts-basic-service': 2,
    'ts-seat-service': 2,
    'ts-station-service': 2,
    'ts-train-service': 2,
    'ts-travel-service': 2,
}

class LoadPattern():
    def __init__(self,type:str,prof:dict):
        self.prof=prof
        self.type=type
        if type=="guest":
            path=prof["path"]
            f=open(path)
            x=f.readline()
            x=x.split(',')
            x=[float(i) for i in x]
            self.buffer=x
            self.period=len(x)
            self.idx=0
            self.cnt=0
        if type=="gentle":
            if ("max" not in prof) or ("min" not in prof):
                return False
        if type=="burst":
            if ("max" not in prof) or ("min" not in prof) or ("period" not in prof) or ("prop" not in prof):
                return False
            max=self.prof["max"]
            min=self.prof["min"]
            period=self.prof["period"]
            prop=self.prof["prop"]
            if period < 10:
                return False
            if prop > 1:
                return False
            self.buffer=[]
            bst=int(period*(1-prop))
            for i in range(bst):
                self.buffer.append(min)
            for i in range(period-bst):
                self.buffer.append(max)
            self.idx=0
        if type=="wave":
            if ("max" not in prof) or ("min" not in prof) or ("period" not in prof):
                return False
            max=self.prof["max"]
            min=self.prof["min"]
            period=self.prof["period"]
            if period < 10:
                return False
            self.buffer=[]
            up=int(period/2)
            down=period-up
            delta=(max-min)/(period/2-1)
            for i in range(up):
                self.buffer.append(min+delta*i)
            for i in range(down):
                self.buffer.append(max-delta*(i+1))
            self.idx=0
            self.cnt=0

    def generate(self):
        if self.type=="gentle":
            return self.gentle()
        if self.type=="burst":
            return self.burst()
        if self.type=="wave":
            return self.wave()
        if self.type=="guest":
            return self.guest()

    def guest(self):
        load=self.buffer[self.idx]
        self.cnt+=1
        self.cnt%=100
        if self.cnt==0:
            self.idx+=1
            self.idx%=self.period
        return load

    def burst(self):
        max=self.prof["max"]
        period=self.prof["period"]
        load=self.buffer[self.idx]
        if max<40:
            bias=max/20
        else:
            bias=max/40
        load=load+random.uniform(-bias,bias)
        self.idx+=1
        self.idx%=period
        return load

    def gentle(self):
        max=self.prof["max"]
        min=self.prof["min"]
        return random.uniform(min,max)
    
    def wave(self):
        max=self.prof["max"]
        min=self.prof["min"]
        period=self.prof["period"]
        load=self.buffer[self.idx]
        if max-min<30:
            bias=2
        else:
            bias=6
        load=load+random.uniform(-bias,bias)
        self.cnt+=1
        self.cnt%=3
        if self.cnt==0:
            self.idx+=1
            self.idx%=period
        return load


class Environment():
    def __init__(self, prof:dict, loadPattern:LoadPattern): #prof for resource
        if prof=={}:
            self.prof=default_prof
        else:
            self.prof=prof #current resource profile
        g,d = get_sub(10, self.prof)
        self.g=g
        self.d=d
        self.lg=loadPattern
        self.load=20

    def get_state(self):
        g=self.g
        d=self.d
        SLO_score=self.cal_SLO(d)
        res_score,res_use=self.cal_res(g)
        return g,d,SLO_score,res_use

    def update_env(self):
        #new load
        load=self.lg.generate()
        if load<15:
            self.load=10
        elif load<30:
            self.load=20
        elif load<50:
            self.load=40
        elif load<70:
            self.load=60
        else:
            self.load=80
        g,d = get_sub(self.load, self.prof)
        self.g=g
        self.d=d

    def perform_action(self, prof):
        for k,v in prof.items():
            if k not in self.prof:
                print("Set resource failed: microservice %s not found" % k)
                return False
        self.prof=prof
        g,d = get_sub(self.load, self.prof)
        SLO_score=self.cal_SLO(d)
        res_score,res_use=self.cal_res(g)
        reward=SLO_score*ALPHA+res_score*(1-ALPHA)
        self.update_env()
        return g,reward,d,SLO_score,res_use #state_graph, reawrd, delay, SLO, resource usage

    def new_reset(self):
        self.prof=default_prof
        self.load=20 #rps

    def cal_SLO(self, d):
        AVG=d[0]
        P95=d[1]
        P99=d[2]
        AVG_score=0
        P95_score=0
        P99_score=0

        if AVG<260:
            AVG_score=1
        elif AVG<300:
            AVG_score=0.75
        elif AVG<320:
            AVG_score=0.4
        else:
            AVG_score=-1
        
        if P95<550:
            P95_score=1
        elif P95<610:
            P95_score=0.8
        elif P95<800:
            P95_score=0.4
        elif P95<1100:
            P95_score=0.1
        else:
            P95_score=-1
        
        if P99<750:
            P99_score=1
        elif P99<1000:
            P99_score=0.6
        elif P99<1400:
            P99_score=0.35
        else:
            P99_score=-1
        
        SLO_score=AVG_score*0.55+P95_score*0.35+P99_score*0.1
        return SLO_score

    def cal_res(self,g):
        res=torch.sum(g.ndata["N"],dim=0)
        res_use=res[1]/res[0]
        res_score=0
        if res_use>0.45:
            res_score=-1
        else:
            res_score=res_use-0.1/0.35
        return res_score,res_use

    def cal_res(self,g):
        res=torch.sum(g.ndata["N"],dim=0)
        res_use=res[1]/res[0]
        res_score=0
        if res_use>0.6:
            res_score=0
        else:
            res_score=res_use/0.6
        return res_score,res_use
            
        

if __name__ == "__main__":
    prof={
        "ts-travel-service":2.0,
		"ts-seat-service":2.0,
		"ts-basic-service":2.0,
		"ts-station-service":2.0,
		"ts-train-service":2.0,
    }
    # load=20
    # g,d = get_sub(load,prof)
    # print(g.ndata["N"])
    # print(g.ndata["N_DELAY"])
    # print(g.edata["E"])
    # print(d)


    lg=LoadPattern("guest",{"path":"../load/load3.csv"})
    for i in range(120):
        print(lg.generate())
    # lg=LoadPattern("burst",{"max":60,"min":20,"period":60,"prop":0.5})
    # env=Environment({},loadPattern=lg)
    # g,reward,d,SLO_score,res_use=env.perform_action(prof)
    # print(g)
    # print(res_use)
    