from mxnet import nd
from mxnet.gluon import nn, rnn
import mxnet as mx


class SampleMLP(nn.Block):
    def __init__(self, bn=False, dropout=None,**kwargs):
        super(SampleMLP,self).__init__(**kwargs)
        self.bn = bn
        self.dropout = dropout
        with self.name_scope():
            self.dense0 = nn.Dense(128,activation='relu')
            self.dense1 = nn.Dense(64,activation='relu')
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

class RNN(nn.Block):
    def __init__(self, mode, embed_dim, hidden_dim,
                 num_layers, dropout=0.5, **kwargs):
        super(RNN, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            # self.encoder = nn.Embedding(vocab_size, embed_dim,
            #                             weight_initializer=mx.init.Uniform(0.1))
            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(hidden_dim, num_layers, activation='relu',
                                   dropout=dropout, input_size=embed_dim)
            elif mode == 'rnn_tanh':
                self.rnn = rnn.RNN(hidden_dim, num_layers, activation='tanh',
                                   dropout=dropout, input_size=embed_dim)
            elif mode == 'lstm':
                self.rnn = rnn.LSTM(hidden_dim, num_layers, dropout=dropout,
                                    input_size=embed_dim)
            elif mode == 'gru':
                self.rnn = rnn.GRU(hidden_dim, num_layers, dropout=dropout,
                                   input_size=embed_dim)
            else:
                raise ValueError("Invalid mode %s. Options are rnn_relu, "
                                 "rnn_tanh, lstm, and gru"%mode)

            self.decoder = nn.Dense(2, in_units=hidden_dim)
            self.hidden_dim = hidden_dim

    def forward(self, inputs, state):
        #emb = self.drop(self.encoder(inputs))
        output, state = self.rnn(inputs, state)
        #output = self.drop(output)
        decoded = self.decoder(output.reshape((-1, self.hidden_dim)))
        return decoded, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)

def detach(state):
    if isinstance(state, (tuple, list)):
        state = [i.detach() for i in state]
    else:
        state = state.detach()
    return state
if __name__ == '__main__':
    from utils import loadDataLabel2, DataLoader
    a,b,c,d = loadDataLabel2()
    print(a.shape)
    rnn = RNN("rnn_relu",11,100,2)
    rnn.collect_params().initialize(mx.init.Xavier())
    batch_size = 32
    clipping_norm = 0.2
    num_steps = 5
    loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = mx.gluon.Trainer(rnn.collect_params(),'adam',{
        'learning_rate':1.0,
        "wd":0.0
    })
    dataLoader = DataLoader(a,b)
    for epoch in range(5):
        total_L = 0.0
        hidden = rnn.begin_state(func=mx.nd.zeros,batch_size = batch_size,ctx=mx.cpu())

        for data,label in dataLoader.dataIter(batch_size):
            label = nd.array(label)
            d = nd.array(data.reshape((32,5,11)).swapaxes(0,1))
            print(d.shape)
            hidden = detach(hidden)
            with mx.autograd.record():
                output, hidden = rnn(d,hidden)
                lv = loss(output,label)
                lv.backward()
            grads = [i.grad() for i in rnn.collect_params().values()]
            mx.gluon.utils.clip_global_norm(grads,clipping_norm*num_steps*batch_size)
            trainer.step(batch_size)
            total_L += mx.nd.sum(lv).asscalar()

        print("Epoch %d loss %.2f " %(epoch, total_L))

            
        
    
    
            
