from trainers import MLPTrainer, mx
import argparse
from utils import selectLoss
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
args = parser.parse_args()

lr = config[args.labelname]['learningrate']
lossfunc = config[args.labelname]['lossfunction']
wd = config[args.labelname]['wd']
dropout = config[args.labelname]['dropout']
bn = config[args.labelname]['bn']

trainer = MLPTrainer(args.labelname, selectLoss(lossfunc),lr,wd,bn=bn,dropout=dropout,all=args.all)
trainer.train(args.epoch,args.batchsize,con=args.con,ctx=mx.cpu())