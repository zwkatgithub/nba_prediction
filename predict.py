import json
from trainers import MLPTrainer, mx, OutputData, nd
from utils import selectLoss, loadDataLabel2, loadDataLabel3

with open('./config.ini','r') as f:
    config = json.load(f)

class Predicter(object):
    labelNames = ('three_pt', 'in_pts', 'ft')
    def __init__(self, config, all=False):
        self.all = all
        self.trainers = {}
        for labelName in self.labelNames:
            self.trainers[labelName] = MLPTrainer(labelName,
                selectLoss(config[labelName]['lossfunction']),
                bn=config[labelName]['bn'],
                dropout=config[labelName]['dropout'],all=self.all)
            self.trainers[labelName].initnet(config[labelName]['learningrate'],config[labelName]['wd'])
            self.trainers[labelName].dataload(*[nd.array(v) for v in loadDataLabel3(labelName,rate=0.7)])
            print(self.trainers[labelName].train_data.shape)
        self.load()
    def load(self):
        for _, trainer in self.trainers.items():
            #print(_)
            trainer.load(mx.cpu())
    def winOrLoss(self, which):
        return self.trainers['three_pt'].train_label
        #label = getattr(trainer,which+'_label')
        
    def predict(self,which='train'):
        r = []
        for _, trainer in self.trainers.items():
            data = getattr(trainer,which+'_data')
            output = trainer.net(data)
            #print("output shape" ,output.shape)
            if _ == 'three_pt':
                r.append(output*3)
            else:
                r.append(output)
        temp = r[0]+r[1]+ r[2]
        
        temp = nd.array(temp > 0).reshape((-1,))
        print(temp) 
        a,b,c,d = loadDataLabel3("three_pt",rate=0.7,alllabel=True)
        if which == 'train':
            l = b
        else:
            l = d
        r = l[:,OutputData.colName().index('three_pt')]*3 + l[:,OutputData.colName().index('ft')]+l[:,OutputData.colName().index('in_pts')]
        r = nd.array(r > 0)
        res = temp==r
        print(res)
        w = nd.sum(temp==r).asscalar()
        print(w,r.shape)
        return w / len(r)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--which')
    parser.add_argument('-a','--all',type=bool,default=False)
    args = parser.parse_args()
    predicter = Predicter(config,args.all)

    
    pre = predicter.predict(args.which)
    print(pre)



