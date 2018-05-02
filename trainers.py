from models import MLP, CMLP
from data_analysis import OutputData
from utils import DataLoader, loadDataLabel, mse, genDictIndex
import mxnet as mx
from mxnet import nd, gluon, autograd
import os
import matplotlib.pyplot as plt
import signal

def sigint_handler(signum, frame):
    global is_sigint_up
    is_sigint_up = True

signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGHUP, sigint_handler)
signal.signal(signal.SIGTERM, sigint_handler)
is_sigint_up = False

class MLPTrainer(object):
    paramsFloder = './params/MLP'
    def __init__(self,labelName, loss, lr,wd, bn=False, dropout=None, all=False):
        self.all = all
        self.labelName = labelName
        self.labelIndex = OutputData.colName().index(labelName)
        self.train_data, self.train_label, self.test_data, self.test_label = \
            [nd.array(y) for y in loadDataLabel(self.labelName,all=all,shuffle=False,CMLP=True)]
        self.dataLoader = DataLoader(self.train_data,self.train_label)
        self.loss = loss #gluon.loss.LogisticLoss()
        self.bn = bn
        self.dropout = dropout
        self.net = MLP(bn,dropout)
        self.lr = lr
        self.wd = wd
        #self.net.initialize()
        self.net.collect_params().initialize(mx.init.Xavier(magnitude=3),ctx=mx.cpu())
        self.trainer = gluon.Trainer(self.net.collect_params(),
            'adam',
            {'learning_rate':self.lr,
            'wd':self.wd})
    def train(self,epochs,batch_size, con=False,ctx=None):
        self.train_loss = []
        self.test_loss = []
        if con:
            self.load(ctx=ctx)
        for epoch in range(epochs):
            train_loss = 0.0
            if is_sigint_up:
                break
            for data, label in self.dataLoader.dataIter(batch_size):
                with autograd.record():
                    output = self.net(data)
                    lossv = self.loss(output,label)
                lossv.backward()
                self.trainer.step(batch_size)

                train_loss += nd.sum(lossv).asscalar()
            ptrain_loss = self.eval(self.train_data,self.train_label)
            ptest_loss = self.eval(self.test_data, self.test_label)
            self.test_loss.append(ptest_loss / len(self.test_data))
            self.train_loss.append(ptrain_loss / len(self.train_data))
            print('Epoch %d : Train loss -> %f Test loss -> %f ' % (epoch, self.train_loss[-1], self.test_loss[-1]))
        p = input('plot ? (y/n)')
        if p.lower() == 'y':
            self.plot()
        r = input('save params ? (y/n)')
        if r.lower() == 'y':
            self.save()
    def plot(self):
        plt.figure(figsize=(8,6))
        plt.plot(self.train_loss)
        plt.plot(self.test_loss)
        plt.legend(['train','test'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.show()
    def eval(self, data, label):
        data = nd.array(data)
        label = nd.array(label)
        #print(output[:,0]-test_label)
        output = self.net(data)
        return nd.sum(self.loss(output,label)).asscalar()
        #loss = nd.sqrt(2*nd.sum(nd.power(output.reshape(label.shape) - label,2))).asscalar()
        #return loss
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

class CMLPTrainer(object):
    def __init__(self,labelName, loss, lr,wd, dictindex,bn=False, dropout=None, all=False):
        self.labelName, self.loss, self.lr, self.wd, self.bn, self.dropout, self.all = \
            labelName, loss, lr, wd, bn, dropout, all
        self.train_data, self.train_label, self.test_data, self.test_label = \
            [nd.array(y) for y in loadDataLabel(self.labelName,all=all,shuffle=False,CMLP=True)]
        self.dataloader = DataLoader(self.train_data, self.train_label)
        self.dict, self.index = dictindex
        self.output_dim = len(self.index)
        self.net = CMLP(self.output_dim,self.bn,self.dropout)
        self.net.collect_params().initialize(mx.init.Xavier())
        
        self.trainer = gluon.Trainer(self.net.collect_params(),'adam',{
            'learning_rate':self.lr,
            'wd':self.wd
        })
    def train(self,epochs,batch_size):
        
        for epoch in range(epochs):
            train_acc = 0.0
            train_loss = 0.0
            test_acc = 0.0
            for data, label in self.dataloader.dataIter(batch_size):
                label = nd.array([self.dict[ll.asscalar()] for ll in label])
                with autograd.record():
                    output = self.net(data)
                    loss = self.loss(output,label)
                loss.backward()
                self.trainer.step(batch_size)
                train_loss += nd.sum(loss).asscalar()
            train_acc = self.accuracy(self.train_data, self.train_label)
            test_acc = self.accuracy(self.test_data, self.test_label)
            print('Epoch %d : Train loss -> %f Train acc -> %f Test acc -> %f' %
                (epoch, train_loss/len(self.train_data), train_acc, test_acc))
    def accuracy(self, data, label):
        l = nd.array([self.dict[ll.asscalar()] for ll in label])
        output = self.net(data)
        return nd.mean(output.argmax(axis=1)==l).asscalar()

        


if __name__=='__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-ln','--labelname')
    # parser.add_argument('-lr','--learningrate',type=float)
    # parser.add_argument('-ep','--epoch',type=int,default=1000)
    # parser.add_argument('-bs','--batchsize',type=int)
    # parser.add_argument('-c','--con',type=bool,default=False)
    # args = parser.parse_args()
    train_data, train_label,_,_ = loadDataLabel('three_pt',all=True)
    print('Start')
    three_pt = CMLPTrainer('three_pt',gluon.loss.SoftmaxCrossEntropyLoss(),0.1,0.005,genDictIndex(train_label))
    three_pt.train(1000,256)
    
    # r = input('save params ? (y/n)')
    # if r.lower() == 'y':
    #     three_pt.save()
    
    
    #in_pts = MLPTrainer('in_pts', gluon.loss.L2Loss(), 0.0001)

    
    
    # in_pts = train(train_data, train_label, 'in_pts', gluon.loss.L2Loss(),0.0001,512,5)
    # print('Test Loss : ',in_pts.eval(test_data,test_label,'in_pts'))
    
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
    
    