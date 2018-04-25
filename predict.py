import json
from trainers import MLPTrainer, mx, OutputData, nd
from utils import selectLoss

with open('./config.ini','r') as f:
    config = json.load(f)

class Predicter(object):
    labelNames = ('three_pt', 'in_pts', 'ft')
    def __init__(self, config):
        self.trainers = {}
        for labelName in self.labelNames:
            self.trainers[labelName] = MLPTrainer(labelName,
                selectLoss(config[labelName]['lossfunction']),
                config[labelName]['learning_rate'])
        self.load()
    def load(self):
        for _, trainer in self.trainers.items():
            trainer.load(mx.cpu())
    def winOrLoss(self):
        res = self.trainers['three_pt'].train_label*3 + self.trainers['in_pts'].train_label + self.trainers['ft'].train_label
        return res>0
    def predict(self):
        label = self.winOrLoss()
        tp = self.trainers['three_pt'].predict(self.trainers['three_pt'].train_data).reshape(label.shape) * 3
        ip = self.trainers['in_pts'].predict(self.trainers['in_pts'].train_data).reshape(label.shape)
        fp = self.trainers['ft'].predict(self.trainers['ft'].train_data).reshape(label.shape)
        r = tp+ip+fp
        r = r > 0
        res = r == label
        w = nd.sum(nd.cast(res,'int32'))
        print(w)
        #print(w.asscalar() / len(res))
        return w.asscalar() / len(res)

if __name__ == '__main__':
    predicter = Predicter(config)

    
    pre = predicter.predict()
    print(pre)



