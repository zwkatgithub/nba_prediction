from trainers import MLPTrainer, mx, nd
import argparse
from utils import selectLoss, loadDataLabel, loadDataLabel2, loadDataLabel3
import json

with open('./config.ini', 'r') as f:
    config = json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument('-ln','--labelname')
#parser.add_argument('-lr','--learningrate',type=float)
parser.add_argument('-ep','--epoch',type=int,default=1000)
parser.add_argument('-bs','--batchsize',type=int)
#parser.add_argument('-lf','--lossfunction')
parser.add_argument('-c','--con',type=bool,default=False)
parser.add_argument('-a','--all',type=bool,default=False)
parser.add_argument('-n',"--net")
args = parser.parse_args()

lr = config[args.labelname]['learningrate']
lossfunc = config[args.labelname]['lossfunction']
wd = config[args.labelname]['wd']
dropout = config[args.labelname]['dropout']
bn = config[args.labelname]['bn']
opt = config[args.labelname]['opt']
init = config[args.labelname]['init']
#a,b,c,d = loadDataLabel3()
#a,b,c,d = loadDataLabel2()
a,b,c,d = loadDataLabel3(args.labelname,rate=0.7)
print("abcd : ",a.shape,b.shape,c.shape,d.shape)
trainer = MLPTrainer(args.labelname, selectLoss(lossfunc),bn=bn,dropout=dropout,all=args.all)
trainer.initnet(lr,wd,opt=opt,init=init)
trainer.dataload(nd.array(a),nd.array(b),nd.array(c),nd.array(d))
trainer.train(args.epoch,args.batchsize,con=args.con,ctx=mx.cpu())