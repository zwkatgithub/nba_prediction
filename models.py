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
            self.dense0 = nn.Dense(128,activation='tanh')
            self.dense1 = nn.Dense(64,activation='tanh')
            if self.bn:
                self.bn0 = nn.BatchNorm(axis=1)
                self.bn1 = nn.BatchNorm(axis=1)
            if self.dropout is not None:
                self.dropout0 = nn.Dropout(self.dropout[0])
                self.dropout1 = nn.Dropout(self.dropout[1])
            self.output = nn.Dense(1)
    def forward(self, data):
        if self.bn and not self.dropout:
            return self.output(self.bn1(self.dense1(self.bn0(self.dense0(data)))))
        if not self.bn and self.dropout:
            return self.output(self.dropout1(self.dense1(self.dropout0(self.dense0(data)))))
        a1 = self.dense0(data)
        a1 = self.bn0(a1)
        a2 = self.dense1(a1)
        #a2 = self.dropout1(a2)
        a2 = self.bn1(a2)
        #a3 = self.dense2(a2)
        #a4 = self.dense3(a3)
        o = self.output(a2)
        # a1 = self.dropout0(self.dense0(data))
        # a2 = self.dropout1(self.dense1(a1))
        # a3 = self.dropout2(self.dense2(a2))
        # a4 = self.
        # o = self.output(a3)
        return o

class CMLP(MLP):
    def __init__(self,output_dim ,**kwargs):
        super(CMLP,self).__init__(**kwargs)
        self.output_dim = output_dim
    def build(self):
        with self.name_scope():
            self.dense0 = nn.Dense(128)
            self.dense1 = nn.Dense(64)
            self.dense2 = nn.Dense(32)
            self.dense3 = nn.Dense(16)
            self.output = nn.Dense(self.output_dim)
    def forward(self, x):
        a1 = self.dense0(x)
        a2 = self.dense1(a1)
        a3 = self.dense2(a2)
        a4 = self.dense3(a3)
        o = self.output(a4)
        
        return o


            
