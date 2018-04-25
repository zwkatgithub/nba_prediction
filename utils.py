import pandas as pd
import json
import numpy as np
from data_analysis import OutputData
from mxnet import gluon, nd
dataFolder = './data/processed/data.csv'


def normalize(data):
    mean = data.mean(axis=0)
    return (data - mean) / data.std(axis=0)
def mse(output, label):
    return nd.sum((output-label)**2) / len(output)
def selectLoss(lossname):
    if lossname == 'l2loss':
        return gluon.loss.L2Loss()    
    elif lossname =='mse':
        return mse
    elif lossname =='log':
        return gluon.loss.LogisticLoss()
    
def read_data(dataFile):
    with open(dataFile,'r') as f:
        data = json.load(f)
    return np.array(data)
def loadDataLabel(labelName, rate = 0.7, all=False, shuffle=False):
    labelIndex = OutputData.colName().index(labelName)
    data = normalize(read_data('./data/datalabel/{0}.txt'.format(labelName)))
    label = read_data('./data/datalabel/label.txt')
    if all:
        rate = 1.0
    train_num = int(len(data) * rate)
    indexs = list(range(0,len(data)))
    if shuffle:
        np.random.shuffle(indexs)
    train_index = indexs[:train_num]
    test_index = indexs[train_num:]
    train_data = data[train_index,:]
    train_label = label[train_index,labelIndex]
    test_data = data[test_index,:]
    test_label = label[test_index,labelIndex]
    return train_data, train_label, test_data, test_label



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
    
        
