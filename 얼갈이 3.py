import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


step =30
dim = 4


data = pd.read_excel('20161.xlsx')
data = np.array(data)
test = pd.read_excel('2017.xlsx')
test = np.array(test)
da16= data[0:184]
da15= data[184:368]
da14=data[368:552]
da13=data[552:736]
da12=data[736:920]
da11=data[920:1104]
da10=data[1104:]


da161= []
da151= []
da141=[]
da131=[]
da121=[]
da111=[]
da101=[]
def tra(x,y):
    for i in range(len(x)):
        tra = x[i][0:5]
        y.append(tra)

tra(da16,da161)
tra(da15,da151)
tra(da14,da141)
tra(da13,da131)
tra(da12,da121)
tra(da11,da111)
tra(da10,da101)











def price(x):
    price = x[:, [-1]]
    return price

dataX=[]
dataY=[]
testX=[]
testY=[]

def qnsfb(x):
    for i in range(0,len(x)-step):
        _x = x[i:i+step]

        dataX.append(_x)


def qnsfb1(x):
    for i in range(0,len(x)-step):

        _y = price(x)[i+step]

        dataY.append(_y)
qnsfb(da161)
qnsfb(da151)
qnsfb(da141)
qnsfb(da131)
qnsfb(da121)
qnsfb(da111)
qnsfb(da101)
qnsfb1(da16)
qnsfb1(da15)
qnsfb1(da14)
qnsfb1(da13)
qnsfb1(da12)
qnsfb1(da11)
qnsfb1(da10)


test1=[]
tra(test,test1)

def xptmxm(x):
    for i in range(0,len(x)-step):
        _x = x[i:i+step]

        testX.append(_x)


def xptmxm1(x):
    for i in range(0,len(x)-step):

        _y = price(x)[i+step]

        testY.append(_y)
xptmxm1(test)
xptmxm(test1)


X=tf.placeholder(tf.float32,[None,step,dim])
Y=tf.placeholder(tf.float32,[None,1])
cell = rnn.BasicLSTMCell(
    num_units=4,  activation=tf.tanh)
multi_cells = rnn.MultiRNNCell([cell for _ in range(2)], state_is_tuple=True)

outputs ,_states=tf.nn.dynamic_rnn(multi_cells,X,dtype=tf.float32)

Y_pre = tf.contrib.layers.fully_connected(outputs[:,-1],1,activation_fn=None)
A=tf.square(Y_pre - Y)

loss = tf.reduce_sum(A)

op = tf.train.AdamOptimizer(0.01)
train = op.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    _,l=sess.run([train,loss],feed_dict={X:dataX,Y:dataY})
    print(i,l)




import  matplotlib.pyplot as plt
testp=sess.run(Y_pre,feed_dict={X:testX})
plt.plot(testY)
plt.plot(testp)
plt.show()