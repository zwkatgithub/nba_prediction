from mxnet import nd
from mxnet.gluon import nn

class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP,self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = nn.Dense(128,activation='tanh')
            self.dense1 = nn.Dense(64,activation='tanh')
            self.dense1 = nn.Dense(32,activation='tanh')
            self.dense1 = nn.Dense(16,activation='tanh')
            self.output = nn.Dense(1)
    def forward(self, data):
        a1 = self.dense0(data)
        a2 = self.dense1(a1)
        o = self.output(a2)
        return o
