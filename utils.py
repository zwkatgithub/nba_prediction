import pandas as pd

dataFolder = './data/processed/data.csv'

def normalize():
    dl = pd.read_csv(dataFolder)
    data = dl.iloc[:,:-9].as_matrix()
    label = dl.iloc[:,-9:].as_matrix()
    norm_data = (data - data.mean(axis=0))/data.std(axis=0)
    return norm_data, label

def data_iter(index_label):
    '''
        :param index_label : [-9 -- -1]
    '''
    data, label = normalize()
    for line in range(data.shape[0]):
        yield data[line,:], label[line,index_label]
