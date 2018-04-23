import sys
sys.path.append('..')
from dev.data_analysis import DataLabel
import os

class CreateDataLabel(object):
    def __init__(self,datafile='./data/datalabel/data.txt',labelfile='./data/datalabel/label.txt'):
        self.datafile = datafile
        self.labelfile = labelfile
    def run(self):
        dataLabel = DataLabel('./data/processed/mask.txt','./data/processed/data.csv')
        dataLabel.createData()
        dataLabel.save(self.datafile,self.labelfile)

if __name__ == '__main__':
    CreateDataLabel().run()


