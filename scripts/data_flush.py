import sys
sys.path.append('..')
import json
from write_data import PlayerData,OutputData
from math import sqrt
databasefile = 'data-{0}.json'
labebaselfile = 'label-{0}.json'
seasons = [str(i) for i in range(7,17)]

pos = ('F','C','G')

def processPlayer(p1,p2):
    for colName in PlayerData.colName():
        # mean = (p1[colName] + p2[colName])/2
        # std = sqrt(((p1[colName]-mean)**2 + (p2[colName]-mean)**2)/2)
        # p1[colName] = (p1[colName] - mean) / std
        # p2[colName] = (p2[colName] - mean) / std
        s = p1[colName] + p2[colName]
        p1[colName] /= s
        p2[colName] /= s
    return p1,p2
def processLabel(h,v):
    h = h['three_pt']*3 + h['ft'] + h['in_pts']
    v = v['three_pt']*3 + v['ft'] + v['in_pts']
    return h > v
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
        for p in pos:
            for p1,p2 in zip(home_data[p],visit_data[p]):
                p1,p2 = processPlayer(p1,p2)
                resdata.append(playerMinus(p1,p2))
                resdata.append(playerMinus(p2,p1)) 
        for hl, vl in zip(home_label, visit_label):
            reslabel.append(float(processLabel(hl,vl)))
            reslabel.append(float(processLabel(vl,hl)))
with open('./data/new_data.json','w') as f:
    json.dump(resdata,f)
with open('./data/new_label.json','w') as f:
    json.dump(reslabel,f)  

