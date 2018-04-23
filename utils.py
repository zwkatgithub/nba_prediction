import pandas as pd
import json
import numpy as np
from data_analysis import OutputData
dataFolder = './data/processed/data.csv'


def normalize(data):
    mean = data.mean(axis=0)
    return (data - mean) / data.std(axis=0)


def data_iter(data,label,batch_size):
    '''
        :param index_label : [-9 -- -1]
    '''
    indexs = list(range(data.shape[0]))
    np.random.shuffle(indexs)
    epochs = len(indexs) // batch_size 
    for epoch in range(epochs):
        cur_indexs = indexs[epoch*batch_size:min((epoch+1)*batch_size, len(indexs))]
        yield data[cur_indexs,:],label[cur_indexs]
            
    
def read_data(dataFile):
    with open(dataFile,'r') as f:
        data = json.load(f)
    return np.array(data)
def splitData(data, label):
    train_num = len(data) * 0.7
    train_data = data[:train_num,:]
    train_label = label[:train_num,:]
    test_data = data[train_num:,:]
    test_label = data[train_num:,:]
    return train_data, train_label, test_data, test_label



class DataLoader(object):
    alldata = read_data('./data/datalabel/data.txt')
    alllabel = read_data('./data/datalabel/label.txt')
    def __init__(self,data,labelname):
        assert labelname in OutputData.colName()
        self.labelname = labelname
        self.labelindex = OutputData.colName().index(self.labelname)
        
        self.label = self.alllabel[:,self.labelindex]
        self.data = data
    def dataIter(self, batch_size):
        return data_iter(self.data,self.label, batch_size)


if __name__ == '__main__':
    dataLoader = DataLoader(DataLoader.alldata, 'three_pt')
    for data, label in dataLoader.dataIter(128):
        print (data ,label)
        break
    
        
