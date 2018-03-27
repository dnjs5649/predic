import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import time

def save_model(save_path):
    saver = tf.train.Saver()
    saver.save(sess, save_path)

with open('토큰1','rb') as mysavedata:
    data = pickle.load(mysavedata)
with open('토큰2','rb') as mysavedata:
    test = pickle.load(mysavedata)
with open('토큰3','rb') as mysavedata:
    data1 = pickle.load(mysavedata)
with open('토큰4','rb') as mysavedata:
    test1 = pickle.load(mysavedata)




def xavier_initializer(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


data = np.array(data)

test = np.array(test)

data1 = np.array(data1)

test1 = np.array(test1)
step =10
dim = 9
X = tf.placeholder('float',[None,1])
X1 = tf.placeholder('float',[None,step,dim])
Y = tf.placeholder('float',[None,1])
Y1 = tf.placeholder('float',[None,10])

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
xptmxm(test, test1)
dataY=np.array(dataY)
testY1=np.array(testY1)

Y_tr=tf.one_hot(dataY,depth=10)
Y_te=tf.one_hot(testY1,depth=10)
Y_tr=tf.reshape(Y_tr, [-1, 10])
Y_te =tf.reshape(Y_te, [-1,10])

import tensorflow.contrib as tfc

is_training = tf.placeholder(tf.bool)

def conv1d_weightnorm(inputs, layer_idx, out_dim, kernel_size, padding="SAME", dropout=0.7,
                      var_scope_name="conv_layer"):

    with tf.variable_scope("conv_layer_" + str(layer_idx)):
        in_dim = int(inputs.get_shape()[-1])
        V = tf.get_variable('V', shape=[kernel_size, in_dim, out_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(
                                4.0 * dropout / (kernel_size * in_dim))), trainable=True)
        V_norm = tf.norm(V.initialized_value(), axis=[0, 1])  # V shape is M*N*k,  V_norm shape is k
        g = tf.get_variable('g', dtype=tf.float32, initializer=V_norm, trainable=True)
        b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)


        W = tf.reshape(g, [1, 1, out_dim]) * tf.nn.l2_normalize(V, [0, 1])
        inputs = tf.nn.bias_add(tf.nn.conv1d(value=inputs, filters=W, stride=1, padding=padding), b)
        return inputs

def linear_mapping_weightnorm(inputs, out_dim, in_dim=None, dropout=0.7, var_scope_name="linear_mapping"):
    with tf.variable_scope(var_scope_name):
        input_shape = inputs.get_shape().as_list()
        input_shape_tensor = tf.shape(inputs)

        V = tf.get_variable('V', shape=[int(input_shape[-1]), out_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(
                                dropout * 1.0 / int(input_shape[-1]))), trainable=True)
        V_norm = tf.norm(V.initialized_value(), axis=0)  # V shape is M*N,  V_norm shape is N
        g = tf.get_variable('g', dtype=tf.float32, initializer=V_norm, trainable=True)
        b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(),
                            trainable=True)

        assert len(input_shape) == 3
        inputs = tf.reshape(inputs, [-1, input_shape[-1]])
        inputs = tf.matmul(inputs, V)
        inputs = tf.reshape(inputs, [input_shape_tensor[0], -1, out_dim])


        scaler = tf.div(g, tf.norm(V, axis=0))
        inputs = tf.reshape(scaler, [1, out_dim]) * inputs + tf.reshape(b, [1, out_dim])
        return inputs

def gated_linear_units(inputs):
  input_shape = inputs.get_shape().as_list()
  assert len(input_shape) == 3
  input_pass = inputs[:,:,0:int(input_shape[2]/2)]
  input_gate = inputs[:,:,int(input_shape[2]/2):]
  input_gate = tf.sigmoid(input_gate)
  return tf.multiply(input_pass, input_gate)

def conv_encoder_stack(inputs, nhids_list, kwidths_list, dropout_dict,is_training):
    next_layer = inputs
    for layer_idx in range(len(nhids_list)):
        nin = nhids_list[layer_idx] if layer_idx == 0 else nhids_list[layer_idx - 1]
        nout = nhids_list[layer_idx]
        res_inputs = next_layer

        if layer_idx == 1:
            next_layer = tf.contrib.layers.dropout(
                inputs=next_layer,
                keep_prob=dropout_dict['hid'], is_training=is_training)

            next_layer = conv1d_weightnorm(inputs=next_layer, layer_idx=layer_idx, out_dim=nout * 2,
                                       kernel_size=kwidths_list[layer_idx], padding="SAME", dropout=dropout_dict['hid'],
                                       var_scope_name="conv_layer_" + str(layer_idx))

            next_layer = tf.contrib.layers.conv2d(inputs=next_layer,num_outputs=nout*2,kernel_size=kwidths_list[layer_idx],
                                              padding="SAME",   weights_initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(4 * dropout_dict['hid'] / (kwidths_list[layer_idx] * next_layer.get_shape().as_list()[-1]))),biases_initializer=tf.zeros_initializer(),activation_fn=None,scope="conv_layer_"+str(layer_idx))
            next_layer = gated_linear_units(next_layer)
            next_layer = (next_layer + res_inputs) * tf.sqrt(0.5)
            next_layer = tf.nn.pool(next_layer, [2], 'MAX', 'SAME', strides=[1])

        elif layer_idx == 2:
            next_layer = tf.contrib.layers.dropout(
                inputs=next_layer,
                keep_prob=dropout_dict['hid'], is_training=is_training)

            next_layer = conv1d_weightnorm(inputs=next_layer, layer_idx=layer_idx, out_dim=nout * 2,
                                           kernel_size=kwidths_list[layer_idx], padding="SAME",
                                           dropout=dropout_dict['hid'],
                                           var_scope_name="conv_layer_" + str(layer_idx))

            next_layer = tf.contrib.layers.conv2d(inputs=next_layer, num_outputs=nout * 2,
                                                  kernel_size=kwidths_list[layer_idx],
                                                  padding="SAME",
                                                  weights_initializer=tf.random_normal_initializer(mean=0,
                                                                                                   stddev=tf.sqrt(
                                                                                                       4 * dropout_dict[
                                                                                                           'hid'] / (
                                                                                                               kwidths_list[
                                                                                                                   layer_idx] *
                                                                                                               next_layer.get_shape().as_list()[
                                                                                                                   -1]))),
                                                  biases_initializer=tf.zeros_initializer(), activation_fn=None,
                                                  scope="conv_layer_" + str(layer_idx))

            next_layer = gated_linear_units(next_layer)
            next_layer = (next_layer + res_inputs) * tf.sqrt(0.5)
            next_layer = tf.nn.pool(next_layer, [2], 'MAX', 'SAME', strides=[1])

        elif layer_idx == 3:
            next_layer = tf.contrib.layers.dropout(
                inputs=next_layer,
                keep_prob=dropout_dict['hid'], is_training=is_training)

            next_layer = conv1d_weightnorm(inputs=next_layer, layer_idx=layer_idx, out_dim=nout * 2,
                                           kernel_size=kwidths_list[layer_idx], padding="SAME",
                                           dropout=dropout_dict['hid'],
                                           var_scope_name="conv_layer_" + str(layer_idx))

            next_layer = tf.contrib.layers.conv2d(inputs=next_layer, num_outputs=nout * 2,
                                                  kernel_size=kwidths_list[layer_idx],
                                                  padding="SAME",
                                                  weights_initializer=tf.random_normal_initializer(mean=0,
                                                                                                   stddev=tf.sqrt(
                                                                                                       4 * dropout_dict[
                                                                                                           'hid'] / (
                                                                                                               kwidths_list[
                                                                                                                   layer_idx] *
                                                                                                               next_layer.get_shape().as_list()[
                                                                                                                   -1]))),
                                                  biases_initializer=tf.zeros_initializer(), activation_fn=None,
                                                  scope="conv_layer_" + str(layer_idx))

            next_layer = gated_linear_units(next_layer)
            next_layer = (next_layer + res_inputs) * tf.sqrt(0.5)


    return next_layer

def encode(inputs):
    embed_size = inputs.get_shape().as_list()[-1]

    with tf.variable_scope("encoder_cnn"):
        next_layer = inputs
        if 4 > 0:
            nhids_list = [128,128,128,128]
            kwidths_list = [3,3,3,3]
            # mapping emb dim to hid dim
            next_layer = linear_mapping_weightnorm(next_layer, nhids_list[0],
                                                   dropout=0.7,
                                                   var_scope_name="linear_mapping_before_cnn")
            next_layer = conv_encoder_stack(next_layer, nhids_list, kwidths_list,
                                            {'src': 0.7,
                                             'hid': 0.7}, is_training)

            next_layer = linear_mapping_weightnorm(next_layer, embed_size, var_scope_name="linear_mapping_after_cnn")

        cnn_c_output = (next_layer + inputs) * tf.sqrt(0.5)

    final_state = tf.reduce_mean(cnn_c_output, 1)

    return  next_layer,final_state,cnn_c_output

def liner(lay2):
    lay2 = tf.reshape(lay2, [-1, step*dim])
    W3 = tf.get_variable("W2", shape=[step*dim, 10], initializer=tfc.layers.xavier_initializer(step*dim,10))
    b = tf.Variable(tf.random_normal([10]))
    f = tf.matmul(lay2, W3) + b
    fi =tf.nn.softmax(f)
    return f , fi

def train(c,Y):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=c,labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    return cost,optimizer

def final(X,Y):

    li2=encode(X)
    z,x,c =li2
    f, fi=liner(z)
    cost=train(f,Y)
    return cost ,fi

def pred(logic,label):
    pre = tf.argmax(logic, 1)
    lab=tf.argmax(label, 1)
    c=[]
    for i in range(len(testY1)):
        d=(pre[i]+1)/(lab[i]+1)
        c.append(d)
    acc = tf.reduce_mean(c)

    return pre, acc


with tf.variable_scope("en")as scopes:
    cost,h=final(X1,Y1)
    tf.get_variable_scope().reuse_variables()
    li=encode(X1)
    z,x,c=li
    q,w =liner(z)
    out = w
    pr,a=pred(out,Y1)



saver = tf.train.Saver()
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    Y_te = sess.run(Y_te)
    Y_tr = sess.run(Y_tr)
    sess.run(init_op)
    save_path = "./save1/model.ckpt"
    saver.restore(sess, save_path)
    testp = sess.run(pr, feed_dict={X1: testX, Y1: Y_te, is_training: False})
    testY = np.array(testY1)
    testa = sess.run(a, feed_dict={X1: testX, Y1: Y_te, is_training: False})
    print(testa)
    import matplotlib.pyplot as plt

    start_time = time.time()
    plt.plot(testY)
    plt.plot(testp)
    plt.show()
    print("start_time", start_time)
    print("--- %s seconds ---" % (time.time() - start_time))
