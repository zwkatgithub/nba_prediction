import sys
sys.path.append('..')
from dev.data_analysis import DataLabel, OutputData
import os

class CreateDataLabel(object):
    def __init__(self,labelName, datafile='./data/datalabel/data.txt',labelfile='./data/datalabel/label.txt'):
        self.datafile = datafile
        self.labelfile = labelfile
        self.labelName = labelName
    def run(self):
        dataLabel = DataLabel('./data/processed/mask.txt','./data/processed/data.csv')
        dataLabel.createData(self.labelName)
        dataLabel.save(self.datafile)
        dataLabel.saveLabel(self.labelfile)

if __name__ == '__main__':
    for labelName in OutputData.colName():
        CreateDataLabel(labelName,datafile='./data/datalabel/'+labelName+'.txt').run()


