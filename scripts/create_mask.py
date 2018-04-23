import sys
sys.path.append('..')
from dev.write_data import OutputData, PlayerData
import os
import json


dataFolder = './data/processed'

pathjoin = os.path.join

class Mask(object):
    def __init__(self, dataFolder,nSteps, stepSize, yz=0.01):
        self.dataFolder = dataFolder
        self.nSteps = nSteps
        self.stepSize = stepSize
        self.yz = yz
        self.pos = ('pf','sf','c','pg','sg')
    def processRow(self,row):
        if abs(row[-1]) >= self.yz:
            return 1
        return 0
    def createMask(self):
        self.mask = []
        for colName in OutputData.colName():
            dataFile = pathjoin(dataFolder,colName,str(self.nSteps)+'-'+str(self.stepSize),'params.txt')
            with open(dataFile,'r') as f:
                data = json.load(f)
            #data : 350*55
            self.mask.append([])
            for j in range(5):
                self.mask[-1].append([])
                for i in range(11):
                    row = [data[k][j*11+i] for k in range(self.nSteps)]
                    res = self.processRow(row)
                    self.mask[-1][-1].append(res)
    def save(self, file):
        with open(file,'w') as f:
            json.dump(self.mask,f)

if __name__ == '__main__':
    mask = Mask(dataFolder,350,0.004)
    mask.createMask()
    for row,colName in enumerate(OutputData.colName()):
        print(colName+' : ')
        for col,pos in enumerate(mask.pos):
            print(mask.mask[row][col])

    mask.save('./data/processed/mask.txt')        
