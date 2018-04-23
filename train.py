from models import MLP
from data_analysis import OutputData
from utils import DataLoader, loadDataLabel
from mxnet import nd, gluon, autograd

batch_size = 256
learning_rate = 0.001
train_data, train_label, test_data, test_label = loadDataLabel()

class Trainer(object):
    def __init__(self,train_data,train_label,labelName, loss, lr):
        self.labelName = labelName
        self.labelIndex = OutputData.colName().index(labelName)
        self.train_data = nd.array(train_data)
        self.train_label = nd.array(train_label)
        self.dataLoader = DataLoader(self.train_data,self.train_label[:,self.labelIndex])
        self.loss = loss #gluon.loss.LogisticLoss()
        self.net = MLP()
        self.lr = lr
        self.net.initialize()
        self.trainer = gluon.Trainer(self.net.collect_params(),
            'sgd',
            {'learning_rate':self.lr})
    def train(self,epochs,batch_size):
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
    def eval(self, test_data, test_label, labelName):
        assert self.labelName == labelName
        test_data = nd.array(test_data)

        test_label = nd.array(test_label[:,OutputData.colName().index(labelName)])
        output = self.net(test_data)
        #print(output[:,0]-test_label)
        loss = nd.sum(output[:,0] - test_label).asscalar()
        return loss
        
        
        

if __name__=='__main__':
    trainer = Trainer(train_data,train_label,'three_pt',gluon.loss.LogisticLoss(),learning_rate)
    trainer.train(5,batch_size)
    test_loss = trainer.eval(test_data, test_label, 'three_pt')
    print('Test Loss : %f' % test_loss)