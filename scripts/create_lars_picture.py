import sys
sys.path.append('..')
from dev.data_analysis import DataAnalysis
import threading

class PictureThread(threading.Thread):
    def __init__(self,labelIndex,nSteps=350,stepSize=0.004):
        self.labelIndex = labelIndex
        self.nSteps = nSteps
        self.stepSize = stepSize
        super(PictureThread,self).__init__()
    def run(self):
        dataAnalysis = DataAnalysis(self.labelIndex,self.nSteps,self.stepSize)
        betaMat = dataAnalysis.analysis()
        dataAnalysis.picture()

if __name__ == '__main__':
    labelIndexs = list(range(-1,-10,-1))
    threads = []
    for labelIndex in labelIndexs:
        threads.append(PictureThread(labelIndex))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

