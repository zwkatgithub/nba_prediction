from mxnet import nd
from mxnet.gluon import nn

class MLP(nn.Block):
    def __init__(self, bn=False, dropout=None,**kwargs):
        super(MLP,self).__init__(**kwargs)
        self.bn = bn
        self.dropout = dropout
        self.build()

    def build(self):
        with self.name_scope():
            self.dense0 = nn.Dense(40,activation='tanh')
            self.dense1 = nn.Dense(10,activation='tanh')
            if self.bn:
                print(self.bn)
                self.bn0 = nn.BatchNorm(axis=1)
                self.bn1 = nn.BatchNorm(axis=1)
            if self.dropout is not None:
                self.dropout0 = nn.Dropout(self.dropout[0])
                self.dropout1 = nn.Dropout(self.dropout[1])
            self.output = nn.Dense(1)
    def forward(self, data):
        if self.bn and not self.dropout:
            return self.output(self.bn1(self.dense1(self.bn0(self.dense0(data)))))
        elif not self.bn and self.dropout:
            return self.output(self.dropout1(self.dense1(self.dropout0(self.dense0(data)))))
        elif not self.bn and not self.dropout:
            return self.output(self.dense1(self.dense0(data)))
        return self.output(self.dropout1(self.bn1(self.dense1(self.dropout0(self.bn0(self.dense0(data)))))))



class CMLP(MLP):
    def __init__(self,output_dim,bn=False,dropout=None,**kwargs):
        self.output_dim = output_dim
        super(CMLP,self).__init__(bn=bn,dropout=dropout,**kwargs)
        
    def build(self):
        with self.name_scope():
            self.dense0 = nn.Dense(256,activation='relu')
            self.dense1 = nn.Dense(64, activation='relu')
            self.dense2 = nn.Dense(32, activation='relu')
            self.output = nn.Dense(self.output_dim)
    def forward(self, x):
        a1 = self.dense0(x)
        a2 = self.dense1(a1)
        a3 = self.dense2(a2)
        o = self.output(a3)
        
        return o


            
