from trainers import MLPTrainer, mx
import argparse
from utils import selectLoss


parser = argparse.ArgumentParser()
parser.add_argument('-ln','--labelname')
parser.add_argument('-lr','--learningrate',type=float)
parser.add_argument('-ep','--epoch',type=int,default=1000)
parser.add_argument('-bs','--batchsize',type=int)
parser.add_argument('-lf','--lossfunction')
parser.add_argument('-c','--con',type=bool,default=False)
args = parser.parse_args()

trainer = MLPTrainer(args.labelname, selectLoss(args.lossfunction),args.learningrate)
trainer.train(args.epoch,args.batchsize,con=args.con,ctx=mx.cpu())
print('Test Loss : ',trainer.eval())
r = input('save params ? (y/n)')
if r.lower() == 'y':
    trainer.save()