import json
from trainers import MLPTrainer, mx, OutputData, nd
from utils import selectLoss

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
                config[labelName]['learningrate'],
                config[labelName]['wd'],bn=config[labelName]['bn'],
                dropout=config[labelName]['dropout'],all=self.all)
        # for _, trainer in self.trainers.items():
        #     print(trainer.net.collect_params())
        self.load()
    def load(self):
        for _, trainer in self.trainers.items():
            #print(_)
            trainer.load(mx.cpu())
    def winOrLoss(self, which):
        # res = self.trainers['three_pt'].train_label*3 + self.trainers['in_pts'].train_label + self.trainers['ft'].train_label
        # return res>0
        # if which == 'train':
        #     temp = nd.zeros(self.trainers['three_pt'].train_label.shape)
        # elif which == 'test':
        #     temp = nd.zeros(self.trainers['three_pt'].test_label.shape)

        return self.trainers['three_pt'].train_label
        #label = getattr(trainer,which+'_label')
        
    def predict(self,which='train'):
        label = getattr(self.trainers['three_pt'],'{0}_label'.format(which))
        print('label shape : ',label.shape)
        temp = nd.zeros(label.shape)
        for _,trainer in self.trainers.items():
            temp += trainer.predict(getattr(trainer,which+'_data'))
        

        r = temp > 0.5 #nd.cast(temp == 1,'float32')+ nd.cast(temp == 2,'float32')+nd.cast(temp == 3,'float32')
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



