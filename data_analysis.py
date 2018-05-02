import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from math import sqrt
import json
from write_data import PlayerData,OutputData
import os
from utils import normalize, genDictIndex

def dataAnalysis(labelIndex,nSteps = 350, stepSize = 0.005,dataFile='./data/processed/data.csv',paramsFolder='./data/processed/'):
    paramsFolder = paramsFolder +str(OutputData.colName()[labelIndex])+'/'+str(nSteps)+'-'+str(stepSize)
    if not os.path.exists(paramsFolder):
        os.makedirs(paramsFolder)
    paramsFile = os.path.join(paramsFolder,'params.txt')
    with open(dataFile,'r') as f:
        data = [[float(v) for v in line.strip().split(',')] for line in f.read().strip().split('\n')]

    xList = []
    labels = []
    
    for row in data:
        xList.append(row[:-9])
        labels.append(row[labelIndex])
    nrows = len(xList)
    ncols = len(xList[0])

    xMeans = []
    xSD = []

    for i in range(ncols):
        col = [xList[j][i] for j in range(nrows)]
        mean = sum(col) / nrows
        xMeans.append(mean)
        colDiff = [(xList[j][i] - mean) for j in range(nrows)]
        sumSq = sum([colDiff[i] * colDiff[i] for i in range(nrows)])
        stdDev = sqrt(sumSq/nrows)
        xSD.append(stdDev)
    xNormalized = []
    for i in range(nrows):
        rowNormalized = [(xList[i][j] - xMeans[j])/xSD[j] for j in range(ncols)]
        xNormalized.append(rowNormalized)
    meanLabel = sum(labels)/nrows
    sdLabel = sqrt(sum([(labels[i]-meanLabel) * (labels[i] - meanLabel) for i in range(nrows)])/nrows)

    labelNormalized = [(labels[i] - meanLabel)/sdLabel for i in range(nrows)]

    beta = [0.0] * ncols
    betaMat = []
    betaMat.append(list(beta))
    for i in range(nSteps):
        print("Step "+str(i)+" ...")
        residuals = [0.0] * nrows
        for j in range(nrows):
            labelsHat = sum([xNormalized[j][k] * beta[k] for k in range(ncols)])
            residuals[j] = labelNormalized[j] - labelsHat
        corr = [0.0] * ncols

        for j in range(ncols):
            corr[j] = sum([xNormalized[k][j]*residuals[k] for k in range(nrows)]) / nrows
        iStar = 0
        corrStar = corr[0]

        for j in range(1,(ncols)):
            if abs(corrStar) < abs(corr[j]):
                iStar = j
                corrStar = corr[j]
        beta[iStar] += stepSize * corrStar / abs(corrStar)
        betaMat.append(list(beta))
    with open(paramsFile,'w') as f:
        json.dump(betaMat,f)
    return betaMat

def dataPlot(labelIndex, nSteps,stepSize,paramsFolder='./data/processed/'):
    pos = ('pf','sf','c','pg','sg')
    paramsFolder = paramsFolder +str(OutputData.colName()[labelIndex])+'/'+str(nSteps)+'-'+str(stepSize)
    if not os.path.exists(paramsFolder):
        os.makedirs(paramsFolder)
    paramsFile = os.path.join(paramsFolder,'params.txt')
    print("Start Plotting")
    with open(paramsFile ,'r') as f:
        betaMat = json.load(f)
    for idx_player in range(5):
        plot.figure(idx_player,figsize=(8,6),dpi=120)
        for i in range(11):
            #print(i,[betaMat[k][idx_player*11+i]  for k in range(nSteps)])
            plot.subplot(11,1,i+1)
            d = [betaMat[k][idx_player*11+i]  for k in range(nSteps)]
            plot.plot(d)
            plot.legend(PlayerData.colName()[i])
            if i==5:
                plot.ylabel('Coefficient Values')
        print(paramsFolder,pos[idx_player]+'.png')
        plot.xlabel('Step')
        plot.savefig(os.path.join(paramsFolder,pos[idx_player]+'.png'))

class DataAnalysis(object):
    def __init__(self,labelIndex, nSteps=350, stepSize=0.004):
        self.labelindex = labelIndex
        self.nSteps = nSteps
        self.stepSize = stepSize
        self.__dataAnalysis = dataAnalysis
        self.__dataPlot = dataPlot
    def analysis(self):
        return self.__dataAnalysis(self.labelindex,self.nSteps,self.stepSize)
    def picture(self):
        self.__dataPlot(self.labelindex,self.nSteps,self.stepSize)

class DataLabel(object):
    def __init__(self, maskFile,rawDataFile):
        self.maskFile = maskFile 
        self.rawDataFile = rawDataFile
        self.__mask()
        self.__rawData()
    def __mask(self):
        with open(self.maskFile,'r') as f:
            mask = json.load(f)
        self.mask = np.array(mask) #
    def __rawData(self):
        with open(self.rawDataFile,'r') as f:
            rawData = [[float(v) for v in line.strip().split(',')] for line in f.read().strip().split('\n')]
        self.rawData = np.array(rawData)
        self.data = self.rawData[:,:55]
    def saveLabel(self, labelFile):
        label = self.rawData[:,-9:]
        with open(labelFile,'w') as f:
            json.dump(label.tolist(),f)
    def createData(self, labelName):
        inputData = []
        labelIndex = OutputData.colName().index(labelName)
        #data = self.data.reshape((-1,5,11))
        for row in self.data:
            #m = np.sum(self.mask[labelIndex],axis=0)
            #m = self.mask[labelIndex]
            #r = row[m!=0].reshape((-1,))
            
            inputData.append(row)
        self.inputData = np.array(inputData)
    def save(self,datafile):
        if not os.path.exists(os.path.dirname(datafile)):
            os.makedirs(os.path.dirname(datafile))
        with open(datafile,'w') as f:
            #print(type(self.inputData.tolist()[0]))
            json.dump(self.inputData.tolist(),f) 
if __name__ == '__main__':
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-a','--analysis',type=bool,default=False)
    # parser.add_argument('-ns','--nsteps',type=int,default=350)
    # #parser.add_argument('-f','--file',type=str,default='./data/processed/data.csv')
    # parser.add_argument('-ss','--stepsize',type=float,default=0.005)
    # #parser.add_argument('-pf','--paramsfile',type=str,default='data.txt')
    # parser.add_argument('labelindex',type=int)

    # args = parser.parse_args()
    # #print(args)
    # if args.analysis:
    #     dataAnalysis(args.labelindex,args.nsteps,args.stepsize,)
    # else:
    #     dataPlot(args.labelindex,args.nsteps,args.stepsize)
    DataLabel('./data/processed/mask.txt','./data/processed/data.csv')


