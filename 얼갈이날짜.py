import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import tensorflow.contrib as tfc

step =10
dim = 9


data = pd.read_excel('201610날짜.xlsx')
data = np.array(data)
test = pd.read_excel('201710날짜.xlsx')
test = np.array(test)
data1 = pd.read_excel('201610an.xlsx')
data1 = np.array(data1)
test1 = pd.read_excel('2017an.xlsx')
test1 = np.array(test1)



da16= data[0:184]
da15= data[184:368]
da14=data[368:552]
da13=data[552:736]
da12=data[736:920]
da11=data[920:1104]
da10=data[1104:]
da161= data1[0:184]
da151= data1[184:368]
da141=data1[368:552]
da131=data1[552:736]
da121=data1[736:920]
da111=data1[920:1104]
da101=data1[1104:]







def price(x):
    price = x[:, [-1]]
    return price

dataX=[]
dataY=[]
testX=[]
testY1=[]

def qnsfb(x,y):
    for i in range(0,len(x)-step):
        _x = x[i:i+step]
        _y = y[i+step]
        dataX.append(_x)
        dataY.append(_y)

qnsfb(da16,da161)
qnsfb(da15,da151)
qnsfb(da14,da141)
qnsfb(da13,da131)
qnsfb(da12,da121)
qnsfb(da11,da111)
qnsfb(da10,da101)

def xptmxm(x,y):
    for i in range(0,len(x)-step):
        _x = x[i:i+step]
        _y = y[i+step]
        testX.append(_x)
        testY1.append(_y)

dataX=np.array(dataX)
print(dataX.shape)
xptmxm(test, test1)

dataY=np.array(dataY)

print(dataY.shape)



X=tf.placeholder(tf.float32,[None,step,dim])
X1=tf.placeholder(tf.float32,[None,step,dim])
Y=tf.placeholder(tf.float32,[None,1])
cell = rnn.BasicLSTMCell(
    num_units=9,  activation=tf.tanh)
multi_cells = rnn.MultiRNNCell([cell for _ in range(3)], state_is_tuple=True)

outputs ,_states=tf.nn.dynamic_rnn(multi_cells,X,dtype=tf.float32)
outputsy ,_states=tf.nn.dynamic_rnn(multi_cells,X1,dtype=tf.float32)

outputs = tf.reshape(outputs, [len(dataX),dim*step])
outputsy = tf.reshape(outputsy, [len(testX),dim*step])
W2=tf.get_variable("W2",shape=[dim*step,step],initializer=tfc.layers.xavier_initializer(dim*step,step))

softmax_b = tf.get_variable("softmax_b",[10])

outputs = tf.nn.relu(tf.matmul(outputs,W2) + softmax_b)
outputsy = tf.nn.relu(tf.matmul(outputsy,W2) + softmax_b)
outputs=tf.nn.dropout(outputs,keep_prob=0.7)
W3=tf.get_variable("W3",shape=[10,1],initializer=tfc.layers.xavier_initializer(10,1))
outputsy = tf.nn.dropout(outputsy,keep_prob=1)
softmax_b1 = tf.get_variable("softmax_b1",[1])

outputs = tf.matmul(outputs,W3) + softmax_b1
outputsy = tf.matmul(outputsy,W3) + softmax_b1
outputs=tf.nn.dropout(outputs,keep_prob=0.7)
outputsy = tf.nn.dropout(outputsy,keep_prob=1)
Y_pre = outputs
Y_prey = tf.nn.sigmoid(outputsy)

A=tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_pre,labels=Y)


loss = tf.reduce_mean(A)

op = tf.train.AdamOptimizer(0.000005)
train = op.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000001):
    _,l=sess.run([train,loss],feed_dict={X:dataX,Y:dataY})
    print(i,l)
    if i % 10000== 0:
        testp = sess.run(Y_prey, feed_dict={X1: testX})
        testY = np.array(testY1)
        testY *= 15003
        testp *= 15003

        import matplotlib.pyplot as plt

        plt.plot(testY)
        plt.plot(testp)
        plt.show()

