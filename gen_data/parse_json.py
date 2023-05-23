#['ts-basic-service', 'ts-config-service', 'ts-order-service', 'ts-price-service', 'ts-route-service', 'ts-seat-service', 'ts-station-service', 'ts-ticketinfo-service', 'ts-train-service', 'ts-travel-service', 'ts-ui-dashboard']
import json
import random
def parse_json(filepath:str):
    f = open(filepath)
    raw=f.readline() 
    raw=raw.split(';')[:-1]
    mslist = [] #used ms names
    delay=[0.,0.,0.]#end2end delay
    tp=0.0
    uiid,tvid=0,0 #id of ui and travel
    for t in raw:
        t=t.strip()
        t=t.split(" ")
        mslist.append(t[0])
    mslist.sort()
    ms2newid={} 
    nodes={}
    nodes_delay={}
    for i in range(len(mslist)):
        nodes_delay[i]=[0,0,0]
    
    for idx, ms in enumerate(mslist):
        ms2newid[ms]=idx
        if ms=='ts-ui-dashboard':
            uiid=idx
        if ms=='ts-travel-service':
            tvid=idx
    for t in raw:
        t=t.strip()
        t=t.split(" ")
        nodes[ms2newid[t[0]]]=[float(t[1])]

    raw=f.readline() #memorydd used
    mem_json = json.loads(raw)
    for t in mem_json: #for value, there is only one value, means mem(MB) avg use
        m,v =t["Metric"], t["Values"]
        if ("container" not in m) or (m["container"] not in mslist):
            continue
        nodes[ms2newid[m["container"]]].append(v[0]["value"])

    raw=f.readline() #memorydd used
    cpu_json = json.loads(raw)
    cpu_used={}
    for t in cpu_json: #for value, there is only one value, means mem(MB) avg use
        m,v =t["Metric"], t["Values"]
        if ("container" not in m) or (m["container"] not in mslist):
            continue
        nodes[ms2newid[m["container"]]].append(v[0]["value"])

    edges=[]
    
    raw=f.readline()
    raw = json.loads(raw)
    oldid2newid={} # new id is filltered, old id is from kiali 
    for t in raw["nodes"]:
        oldid=t["id"]
        name=t["name"]
        if name not in mslist:
            continue
        newid=ms2newid[name]
        oldid2newid[oldid]=newid
    for t in raw["edges"]:
        oldfrom=t["from"]
        oldto=t["to"]
        e={}
        if oldto in oldid2newid:
            newto = oldid2newid[oldto]
            nodes_delay[newto][0] =max(int(t["responsetime"]),nodes_delay[newto][0])
        if (oldfrom not in oldid2newid) or (oldto not in oldid2newid):
            continue
        newfrom = oldid2newid[oldfrom]
        newto = oldid2newid[oldto]
        rps=t["rps"]
        throughtPut=t["throughtPut"]
        responsetime=int(t["responsetime"])
        HttpPercentReq=t["HttpPercentReq"]
        e["from"]=newfrom
        e["to"]=newto
        e["rps"]=rps
        e["throughtPut"]=throughtPut
        e["HttpPercentReq"]=HttpPercentReq
        e["AVG"]=responsetime
        if newfrom==uiid and newto==tvid:
            delay[0]=int(responsetime)
            tp=float(throughtPut)
        edges.append(e)
    raw=f.readline()
    raw = json.loads(raw)
    oldid2newid={} # new id is filltered, old id is from kiali 
    for t in raw["nodes"]:
        oldid=t["id"]
        name=t["name"]
        if name not in mslist:
            continue
        newid=ms2newid[name]
        oldid2newid[oldid]=newid
    for t in raw["edges"]:
        oldfrom=t["from"]
        oldto=t["to"]
        if oldto in oldid2newid:
            newto = oldid2newid[oldto]
            nodes_delay[newto][1] =max(int(t["responsetime"]),nodes_delay[newto][1])
        if (oldfrom not in oldid2newid) or (oldto not in oldid2newid):
            continue
        newfrom = oldid2newid[oldfrom]
        newto = oldid2newid[oldto]
        responsetime=int(t["responsetime"])
        flag=True
        for idx,e in enumerate(edges):
            if e["from"] == newfrom and e["to"] == newto:
                edges[idx]["P95"]=responsetime
                flag=False
        if flag:
            print("edge %s to %s not found",oldfrom,oldto)
        if newfrom==uiid and newto==tvid:
            delay[1]=int(responsetime)
                
    raw=f.readline()
    raw = json.loads(raw)
    oldid2newid={} # new id is filltered, old id is from kiali 
    for t in raw["nodes"]:
        oldid=t["id"]
        name=t["name"]
        if name not in mslist:
            continue
        newid=ms2newid[name]
        oldid2newid[oldid]=newid
    for t in raw["edges"]:
        oldfrom=t["from"]
        oldto=t["to"]
        if oldto in oldid2newid:
            newto = oldid2newid[oldto]
            nodes_delay[newto][2] =max(int(t["responsetime"]),nodes_delay[newto][2])
        if (oldfrom not in oldid2newid) or (oldto not in oldid2newid):
            continue
        newfrom = oldid2newid[oldfrom]
        newto = oldid2newid[oldto]
        responsetime=int(t["responsetime"])
        flag=True
        for idx,e in enumerate(edges):
            if e["from"] == newfrom and e["to"] == newto:
                edges[idx]["P99"]=responsetime
                flag=False
        if flag:           
            print("edge %s to %s not found",oldfrom,oldto)
        if newfrom==uiid and newto==tvid:
            delay[2]=int(responsetime)
    # print(mslist)
    # print(nodes) # [cpu_alloc, mem_alloc, cpu_use]
    # print(edges) # [from, to, rps, throught, http, avg, p95, p99]
    for i in range(len(delay)):
        if delay[i]<500:
            d=delay[i]/200
            delay[i]+=random.uniform(-d,d)
        else:            
            d=delay[i]/50
            delay[i]+=random.uniform(-d,d)
    f.close()
    return nodes,nodes_delay, edges, delay, tp

# parse_json("./base/format.json")
if __name__ == '__main__':
    n,nd,e,d=parse_json("../base/std/60-60.json")
    print(n)
    print(nd)
    print(e)
    print(d)