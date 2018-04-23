import pandas as pd
import json
import numpy as np
from data_analysis import OutputData
dataFolder = './data/processed/data.csv'


def normalize(data):
    mean = data.mean(axis=0)
    return (data - mean) / data.std(axis=0)
    
            
    
def read_data(dataFile):
    with open(dataFile,'r') as f:
        data = json.load(f)
    return np.array(data)
def loadDataLabel():
    data = normalize(read_data('./data/datalabel/data.txt'))
    label = read_data('./data/datalabel/label.txt')
    train_num = int(len(data) * 0.7)
    train_data = data[:train_num,:]
    train_label = label[:train_num,:]
    test_data = data[train_num:,:]
    test_label = data[train_num:,:]
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
    train_data, train_label, test_data, test_label = loadDataLabel()
    dataLoader = DataLoader(train_data, train_label)
    for data, label in dataLoader.dataIter(128):
        print (data ,label)
        break
    
        
