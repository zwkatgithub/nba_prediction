import json
from write_data import PlayerData,OutputData
from math import sqrt
databasefile = './data/raw/data-{0}.json'
labebaselfile = './data/raw/label-{0}.json'
seasons = [str(i) for i in range(7,17)]

pos = ('F','C','G')

def processPlayer(p1,p2):
    n_p1 = {}
    n_p2 = {}
    for colName in PlayerData.colName():
        n_p1[colName] = p1[colName]
        n_p2[colName] = p2[colName]
        if colName[-3:] == 'pct':
            continue
            
        # mean = (p1[colName] + p2[colName])/2
        # std = sqrt(((p1[colName]-mean)**2 + (p2[colName]-mean)**2)/2)
        # p1[colName] = (p1[colName] - mean) / std
        # p2[colName] = (p2[colName] - mean) / std
        s = p1[colName] + p2[colName]
        if s == 0:
            s = 1
        n_p1[colName] /= s
        n_p2[colName] /= s
    return n_p1,n_p2
def processLabel(h,v):
    #print(h,v)
    h = h['three_pt']*3 + h['ft'] + h['in_pts']
    v = v['three_pt']*3 + v['ft'] + v['in_pts']
    return h > v
def processLabel2(h,v):
    r = []
    for labelname in OutputData.colName():
        r.append(h[labelname]-v[labelname])
    return r
def playerMinus(h,v):
    r = []
    for colName in PlayerData.colName():
        r.append(h[colName] - v[colName])
    return r
resdata = []
reslabel = []
for season in seasons:
    
    datafile = databasefile.format(season)
    labelfile = labebaselfile.format(season)
    with open(datafile,'r') as f:
        alldata = json.load(f)
    with open(labelfile,'r') as f:
        alllabel = json.load(f)
    for data, label in zip(alldata,alllabel):
        #one game
        home_data, visit_data = data[0], data[1]
        home_label, visit_label = label[0], label[1]
        r = []
        r2 = []
        for p in pos:
            for p1,p2 in zip(home_data[p],visit_data[p]):
                p1,p2 = processPlayer(p1,p2)
                r += playerMinus(p1,p2)
                r2 += playerMinus(p2,p1)
        resdata.append(r)
        resdata.append(r2)
        #print(home_label)
        #reslabel.append(home_label)
        reslabel.append(processLabel2(home_label,visit_label))
        reslabel.append(processLabel2(visit_label,home_label))
indexs = list(range(len(resdata)))
import numpy as np
np.random.shuffle(indexs)
resdata, reslabel = np.array(resdata), np.array(reslabel)
with open('./data/new_data1.json','w') as f:
    json.dump(resdata[indexs].tolist(),f)
with open('./data/new_label1.json','w') as f:
    json.dump(reslabel[indexs].tolist(),f)  

