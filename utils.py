import pandas as pd
import json
import numpy as np
from collections import defaultdict

from mxnet import gluon, nd
dataFolder = './data/processed/data.csv'


def normalize(data):
    #data += 1e-5
    print(data.shape)
    mean = data.mean(axis=0)
    return (data - mean) / data.std(axis=0)

def eigValPct(eigVals,percentage):
    sortArray=np.sort(eigVals) 
    sortArray=sortArray[-1::-1] 
    arraySum=np.sum(sortArray) 
    tempSum=0
    num=0
    for i in sortArray:
        tempSum+=i
        num+=1
        if tempSum>=arraySum*percentage:
            return num

def pca(dataMat,percentage=0.9):
    meanVals=np.mean(dataMat,axis=0) 
    meanRemoved=dataMat-meanVals
    covMat=np.cov(meanRemoved,rowvar=0) 
    eigVals,eigVects=np.linalg.eig(np.mat(covMat)) 
    k=eigValPct(eigVals,percentage)
    eigValInd=np.argsort(eigVals)  
    eigValInd=eigValInd[:-(k+1):-1]
    redEigVects=eigVects[:,eigValInd]  
    lowDDataMat=meanRemoved*redEigVects 
    reconMat=(lowDDataMat*redEigVects.T)+meanVals  
    return lowDDataMat,reconMat
def mse(output, label):
    return nd.sum((output-label)**2) / len(output)
def selectLoss(lossname):
    if lossname == 'l2loss':
        return gluon.loss.L2Loss()    
    elif lossname =='mse':
        return mse
    elif lossname =='log':
        return gluon.loss.LogisticLoss()
    elif lossname == 'softmax':
        return gluon.loss.SoftmaxCrossEntropyLoss()

def genDictIndex(label):
    d = defaultdict(lambda : 0)
    for l in label:
        d[l] += 1
    res = {}
    i = []
    for key,v in d.items():
        res[key]= len(i)
        i.append(key)
    return res, i
from data_analysis import OutputData
def read_data(dataFile):
    with open(dataFile,'r') as f:
        data = json.load(f)
    return np.array(data)
def loadDataLabel(labelName, rate = 0.7, alllabel=False, shuffle=False, CMLP=False):
    dataFile = './data/datalabel/{0}.txt'
    labelFile = './data/datalabel/label.txt'
    if CMLP:
        dataFile = './data/datalabel/CMLP/{0}.txt'
        labelFile = './data/datalabel/CMLP/label.txt'
    #data,_ = pca(normalize(read_data(dataFile.format(labelName))))
    #np.random.shuffle(data)
    data = normalize(read_data(dataFile.format(labelName)))
    label = read_data(labelFile)
    #label = label[:,OutputData.colName().index('three_pt')]*3+ label[:,OutputData.colName().index('ft')]+label[:,OutputData.colName().index('in_pts')]
    #label = label > 0
    #if all:
    #    rate = 1.0
    if alllabel==False:
        label = label[:,OutputData.colName().index(labelName)]
    train_num = int(len(data) * rate)
    indexs = list(range(0,len(data)))
    if shuffle:
        np.random.shuffle(indexs)
    train_index = indexs[:train_num]
    test_index = indexs[train_num:]
    train_data = data[train_index,:]
    train_label = label[train_index]
    test_data = data[test_index,:]
    test_label = label[test_index]
    return train_data, train_label, test_data, test_label
def loadDataLabel2(rate = 0.8):
    rawdatafile = './data/processed/data.csv'
    with open(rawdatafile,'r') as f:
        rawdata = np.array([[float(a) for a in r.split(',')] for r in f.read().split('\n')])
    data = normalize(rawdata[:,:55])
    label = rawdata[:,55:]
    label = label[:,OutputData.colName().index('three_pt')]*3+ label[:,OutputData.colName().index('ft')]+label[:,OutputData.colName().index('in_pts')]
    label = label > 0
    print(rate)
    #print(label.shape[0]*rate)
    train_num = int(rawdata.shape[0] * rate)
    #print(data[train_num:,:].shape)
    train_data, train_label = data[:train_num,:], label[:train_num]
    test_data, test_label = data[train_num:,:], label[train_num:]
    #train_data, _ = pca(normalize(train_data))
    #test_data, _ = pca(normalize(test_data))
    return train_data, train_label, test_data, test_label
def loadDataLabel3(labelname,rate=0.8, alllabel=False):
    datafile = './data/new_data.json'
    labelfile = './data/new_label.json'
    labelindex = OutputData.colName().index(labelname)
    with open(datafile,'r') as f:
        data = json.load(f)
    with open(labelfile,'r') as f:
        label = json.load(f)
    data = normalize(np.array(data))
    label = np.array(label)
    #data,_ = pca(normalize(np.array(data)))
    if alllabel == False:
        label = label[:,labelindex]
    #data = normalize(np.array(data))
    
    train_num = int(rate*len(data))
    train_data = data[:train_num,:]
    train_label = label[:train_num]
    test_data = data[train_num:,:]
    test_label = label[train_num:]
    return train_data,train_label, test_data, test_label
class DataLoader(object):
    
    def __init__(self,data,label):
        self.label = label
        self.data = data
    def dataIter(self, batch_size):
        indexs = list(range(self.data.shape[0]))
        np.random.shuffle(indexs)
        epochs = len(indexs) // batch_size 
        for epoch in range(epochs):
            cur_indexs = indexs[epoch*batch_size:min((epoch+1)*batch_size, len(indexs))]
            yield self.data[cur_indexs,:], self.label[cur_indexs]
   
        


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = loadDataLabel('three_pt')
    dataLoader = DataLoader(train_data, train_label)
    for data, label in dataLoader.dataIter(128):
        print (data ,label)
        break
    
        
