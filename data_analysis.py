from utils import normalize
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from math import sqrt
from write_data import PlayerData

# data = pd.read_csv('./data/processed/data.csv').iloc[:,:-9]
# datam = data.corr()
# d = pd.DataFrame(datam)
# plt.pcolor(d)
# plt.show()

with open('./data/processed/data.csv','r') as f:
    data = [[float(v) for v in line.strip().split(',')] for line in f.read().strip().split('\n')]

xList = []
labels = []
names = []
firstLine = True
labelIndex = -1
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

nSteps = 350
stepSize = 0.005

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
print("Start Plotting")
for i in range(ncols):
    coefCurve = [betaMat[k][i] for k in range(nSteps)]
    res = ""
    names = ['player {0}'.format(i) + " {0} : ".format(colName) for i in range(5) for colName in PlayerData.colName()]
    for row in zip(names,coefCurve):
        res += row[0]
        res += str(row[1]) + '\n'
    xaxis = range(nSteps)
    plot.plot(xaxis, coefCurve)
with open('./data.txt','w') as f:
    f.write(res)
plot.xlabel("Step Taken")
plot.ylabel(("Coefficient Values"))
plot.legend(PlayerData.colName())
plot.show()


