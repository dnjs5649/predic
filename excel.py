import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc
from tensorflow.contrib import rnn

step = 10
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


def price(x):
    price = x[:, [0]]
    return price

dataX=[]
dataY=[]
testX=[]
testY=[]

def qnsfb(x):
    for i in range(0,len(x)-step):
        _x = x[i:i+step]
        _y = price(x)[i+step]
        dataX.append(_x)
        dataY.append(_y)
qnsfb(da16)
qnsfb(da15)
qnsfb(da14)
qnsfb(da13)
qnsfb(da12)
qnsfb(da11)
qnsfb(da10)




def xptmxm(x):
    for i in range(0,len(x)-step):
        _x = x[i:i+step]
        _y = price(x)[i+step]
        testX.append(_x)
        testY.append(_y)

xptmxm(test)
testY = np.array(testY)
X=tf.placeholder(tf.float32,[None,step,dim])
Y=tf.placeholder(tf.float32,[None,1])
cell = tf.contrib.rnn.BasicLSTMCell(num_units=4, state_is_tuple=True, activation=tf.tanh)
multi_cells = rnn.MultiRNNCell([cell for _ in range(2)], state_is_tuple=True)

outputs ,_states=tf.nn.dynamic_rnn(multi_cells,X,dtype=tf.float32)

Y_pre = tf.contrib.layers.fully_connected(outputs[:,-1],1,activation_fn=None)
A=tf.square(Y_pre - Y)

loss = tf.reduce_mean(A)

op = tf.train.AdamOptimizer(0.01)
train = op.minimize(loss)

from pprint import pprint
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    _,l=sess.run([train,loss],feed_dict={X:dataX,Y:dataY})
    print(i,l)

testp=sess.run(Y_pre,feed_dict={X:testX})

testY *= 15003
testp *= 15003

import  matplotlib.pyplot as plt

plt.plot(testY)
plt.plot(testp)
plt.show()



