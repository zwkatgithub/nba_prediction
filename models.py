from mxnet import nd
from mxnet.gluon import nn


class SampleMLP(nn.Block):
    def __init__(self, bn=False, dropout=None,**kwargs):
        super(SampleMLP,self).__init__(**kwargs)
        self.bn = bn
        self.dropout = dropout
        with self.name_scope():
            self.dense0 = nn.Dense(32,activation='relu')
            self.dense1 = nn.Dense(16,activation='relu')
            self.output = nn.Dense(2)
    def forward(self,x):
        return self.output(self.dense1(self.dense0(x)))

class MLP(nn.Block):
    def __init__(self, bn=False, dropout=None,**kwargs):
        super(MLP,self).__init__(**kwargs)
        self.bn = bn
        self.dropout = dropout
        self.build()


    def build(self):
        with self.name_scope():
            self.dense0 = nn.Dense(256,activation='relu')
            self.dense1 = nn.Dense(128,activation='relu')
            self.dense2 = nn.Dense(64,activation='relu')
            self.dense3 = nn.Dense(32,activation='relu')
            if self.bn:
                print(self.bn)
                self.bn0 = nn.BatchNorm(axis=1)
                self.bn1 = nn.BatchNorm(axis=1)
                self.bn2 = nn.BatchNorm(axis=1)
                self.bn3 = nn.BatchNorm(axis=1)
            if self.dropout is not None:
                self.dropout0 = nn.Dropout(self.dropout[0])
                self.dropout1 = nn.Dropout(self.dropout[1])
                self.dropout2 = nn.Dropout(self.dropout[2])
                self.dropout3 = nn.Dropout(self.dropout[3])
            self.output = nn.Dense(2)
    def forward(self, data):
        if self.bn and not self.dropout:
            return self.output(self.bn3(self.dense3(self.bn2(self.dense2(self.bn1(self.dense1(self.bn0(self.dense0(data)))))))))
        elif not self.bn and self.dropout:
            return self.output(self.dropout3(self.dense3(self.dropout2(self.dense2(self.dropout1(self.dense1(self.dropout0(self.dense0(data)))))))))
        elif not self.bn and not self.dropout:
            return self.output(self.dense3(self.dense2(self.dense1(self.dense0(data)))))
        return self.output(self.dropout3(self.bn3(self.dense3(self.dropout2(self.bn2(self.dense2(self.dropout1(self.bn1(self.dense1(self.dropout0(self.bn0(self.dense0(data)))))))))))))



class CMLP(MLP):
    def __init__(self,output_dim,bn=False,dropout=None,**kwargs):
        self.output_dim = output_dim
        super(CMLP,self).__init__(bn=bn,dropout=dropout,**kwargs)
        
    def build(self):
        with self.name_scope():
            self.dense0 = nn.Dense(512,activation='relu')
            self.dense1 = nn.Dense(256, activation='relu')
            self.dense2 = nn.Dense(128, activation='relu')
            self.dense3 = nn.Dense(64, activation='relu')
            #self.dense2 = nn.Dense(32, activation='relu')
            self.output = nn.Dense(self.output_dim)
    def forward(self, x):
        a1 = self.dense0(x)
        a2 = self.dense1(a1)
        a3 = self.dense2(a2)
        a4 = self.dense3(a3)
        o = self.output(a4)
        
        return o


            
