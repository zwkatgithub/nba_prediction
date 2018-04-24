import json
from trainers import MLPTrainer, mx, OutputData, nd
from utils import selectLoss, 

with open('./config.ini','r') as f:
    config = json.load(f)

class Predicter(object):
    labelNames = ('three_pt', 'in_pts', 'ft')
    def __init__(self, config):
        self.trainers = {}
        for labelName in self.labelNames:
            self.trainers[labelName] = MLPTrainer(labelName,
                selectLoss(config[labelName].lossfunction),
                config[labelName].learning_rate)
    def load(self):
        for trainer in self.trainers:
            trainer.load(mx.cpu())
    def winOrLoss(self,label):
        a = OutputData.colName().index('three_pt')
        b = OutputData.colName().index('in_pts')
        c = OutputData.colName().index('ft')
        res = label[:,a]*3 + label[:,b] + label[:,c]
        return res>0
    def predict(self, data, label):
        label = self.winOrLoss(nd.array(label))
        tp = self.trainers['three_pt'].predict(data).reshape(label.shape) * 3
        ip = self.trainers['in_pts'].net(data).reshape(label.shape)
        fp = self.trainers['ft'].net(data).reshape(label.shape)
        r = tp+ip+fp
        r = r > 0
        res = r == label
        w = nd.sum(nd.cast(res,'int32'))
        print(w)
        print(w.asscalar() / len(res))
        return w.asscalar() / len(res)

if __name__ == '__main__':
    predicter = Predicter(config)
    
    predicter.predict



