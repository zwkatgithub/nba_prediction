from models import MLP
from data_analysis import OutputData
from utils import DataLoader, loadDataLabel
import mxnet as mx
from mxnet import nd, gluon, autograd
import os


class MLPTrainer(object):
    paramsFloder = './params/MLP'
    def __init__(self,labelName, loss, lr):
        self.labelName = labelName
        self.labelIndex = OutputData.colName().index(labelName)
        self.train_data, self.train_label, self.test_data, self.test_label = [nd.array(y) for y in loadDataLabel(self.labelName)]
        self.dataLoader = DataLoader(self.train_data,self.train_label)
        self.loss = loss #gluon.loss.LogisticLoss()
        self.net = MLP()
        self.lr = lr
        self.net.initialize()
        self.trainer = gluon.Trainer(self.net.collect_params(),
            'adam',
            {'learning_rate':self.lr})
    def train(self,epochs,batch_size, con=False,ctx=None):
        if con:
            self.load(ctx=ctx)
        for epoch in range(epochs):
            train_loss = 0.0
            for data, label in self.dataLoader.dataIter(batch_size):
                with autograd.record():
                    output = self.net(data)
                    lossv = self.loss(output,label)
                lossv.backward()
                self.trainer.step(batch_size)

                train_loss += nd.sum(lossv).asscalar()
            print('Epoch %d : Train loss - %f' % (epoch, train_loss / len(self.train_data)))
        #self.save()
    def eval(self):
        test_data = nd.array(self.test_data)
        test_label = nd.array(self.test_label)
        output = self.net(test_data)
        #print(output[:,0]-test_label)
        loss = nd.sum(output.reshape(test_label.shape) - test_label).asscalar()
        return loss
    def predict(self, x):
        x = nd.array(x)
        l = self.net(x)
        return l
    def save(self):
        name = self.labelName+str(self.lr)+'.txt'
        paramsFile = os.path.join(self.paramsFloder,name)
        self.net.save_params(paramsFile)
    def load(self,ctx=None):
        name = self.labelName+str(self.lr)+'.txt'
        paramsFile = os.path.join(self.paramsFloder,name)
        self.net.load_params(paramsFile,ctx)



if __name__=='__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-ln','--labelname')
    # parser.add_argument('-lr','--learningrate',type=float)
    # parser.add_argument('-ep','--epoch',type=int,default=1000)
    # parser.add_argument('-bs','--batchsize',type=int)
    # parser.add_argument('-c','--con',type=bool,default=False)
    # args = parser.parse_args()

    # three_pt = MLPTrainer(args.labelname, gluon.loss.L2Loss(),args.learningrate)
    # three_pt.train(1000,256,con=args.con,ctx=mx.cpu())
    # print('Test Loss : ',three_pt.eval())
    # r = input('save params ? (y/n)')
    # if r.lower() == 'y':
    #     three_pt.save()
    
    
    #in_pts = MLPTrainer('in_pts', gluon.loss.L2Loss(), 0.0001)

    
    
    # in_pts = train(train_data, train_label, 'in_pts', gluon.loss.L2Loss(),0.0001,512,5)
    # print('Test Loss : ',in_pts.eval(test_data,test_label,'in_pts'))
    ft = MLPTrainer('ft',gluon.loss.L2Loss(),0.0001)
    #ft.train(1000,512)
    #print('Test Loss : ', ft.eval())

    # tp = three_pt.predict(train_data) * 3
    # ip = in_pts.predict(train_data)
    # ftp = ft.predict(train_data)
    # r = tp+ip+ftp
    # r = r.reshape((1,r.shape[0]))
    # r = r > 0
    # nd.cast(r,'int32')
    # wol = winOrLoss(nd.array(train_label))
    # nd.cast(wol,'int32')
    # w = nd.sum(nd.cast(r==wol,'int32'))
    # print(w)
    # print(w.asscalar() / len(wol))
    
    