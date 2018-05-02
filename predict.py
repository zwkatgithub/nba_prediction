import json
from trainers import MLPTrainer, mx, OutputData, nd
from utils import selectLoss, loadDataLabel2, loadDataLabel

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
            self.trainers[labelName].dataload(*[nd.array(v) for v in loadDataLabel3(labelName,rate=0.7,shuffle=True,alllabel=True)])
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
        label = getattr(self.trainers['three_pt'],'{0}_label'.format(which))
        print('label shape : ',label.shape)
        label = label[:,OutputData.colName().index('three_pt')]*3 + label[:,OutputData.colName().index('ft')]+label[:,OutputData.colName().index('in_pts')]
        label = label > 0
        print('label shape : ',label.shape)
        temp = nd.zeros(label.shape)
        for _,trainer in self.trainers.items():
            print(_+' data shape : ',getattr(trainer,which+'_data').shape)
            if _ == 'three_pt':
                temp += trainer.predict(getattr(trainer,which+'_data'))*3
            else:
                temp += trainer.predict(getattr(trainer,which+'_data'))

        #print("temp shape",temp.shape)
        #temp = temp[:,OutputData.colName().index("three_pt")]*3 + temp[:,OutputData.colName().index('in_pts')] + temp[:,OutputData.colName().index('ft')]
        print("temp shape",temp.shape)
        r = temp > 0.0 #nd.cast(temp == 1,'float32')+ nd.cast(temp == 2,'float32')+nd.cast(temp == 3,'float32')
        res = r == label
        with open('r.txt','w') as f:
            f.write(str(res.asnumpy().tolist()))
        w = nd.sum(nd.cast(res,'int32'))
        print(w.asscalar())
        print(len(res))
        #print(w.asscalar() / len(res))
        return w.asscalar() / len(res)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--which')
    parser.add_argument('-a','--all',type=bool,default=False)
    args = parser.parse_args()
    predicter = Predicter(config,args.all)

    
    pre = predicter.predict(args.which)
    print(pre)



