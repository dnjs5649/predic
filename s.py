
import tensorflow as tf
import pandas as pd
import numpy as np







import tensorflow.contrib as tfc



is_training = tf.placeholder(tf.bool)



def conv1d_weightnorm(inputs, layer_idx, out_dim, kernel_size, padding="SAME", dropout=1.0,
                      var_scope_name="conv_layer"):  # padding should take attention

    with tf.variable_scope("conv_layer_" + str(layer_idx)):
        in_dim = int(inputs.get_shape()[-1])
        V = tf.get_variable('V', shape=[kernel_size, in_dim, out_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(
                                4.0 * dropout / (kernel_size * in_dim))), trainable=True)
        V_norm = tf.norm(V.initialized_value(), axis=[0, 1])  # V shape is M*N*k,  V_norm shape is k
        g = tf.get_variable('g', dtype=tf.float32, initializer=V_norm, trainable=True)
        b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(g, [1, 1, out_dim]) * tf.nn.l2_normalize(V, [0, 1])
        inputs = tf.nn.bias_add(tf.nn.conv1d(value=inputs, filters=W, stride=1, padding=padding), b)
        return inputs


def linear_mapping_weightnorm(inputs, out_dim, in_dim=None, dropout=1.0, var_scope_name="linear_mapping"):
    with tf.variable_scope(var_scope_name):
        input_shape = inputs.get_shape().as_list()  # static shape. may has None
        input_shape_tensor = tf.shape(inputs)
        # use weight normalization (Salimans & Kingma, 2016)  w = g* v/2-norm(v)
        V = tf.get_variable('V', shape=[int(input_shape[-1]), out_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(
                                dropout * 1.0 / int(input_shape[-1]))), trainable=True)
        V_norm = tf.norm(V.initialized_value(), axis=0)  # V shape is M*N,  V_norm shape is N
        g = tf.get_variable('g', dtype=tf.float32, initializer=V_norm, trainable=True)
        b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(),
                            trainable=True)  # weightnorm bias is init zero

        assert len(input_shape) == 3
        inputs = tf.reshape(inputs, [-1, input_shape[-1]])
        inputs = tf.matmul(inputs, V)
        inputs = tf.reshape(inputs, [input_shape_tensor[0], -1, out_dim])
        # inputs = tf.matmul(inputs, V)    # x*v

        scaler = tf.div(g, tf.norm(V, axis=0))  # g/2-norm(v)
        inputs = tf.reshape(scaler, [1, out_dim]) * inputs + tf.reshape(b, [1, out_dim])
        return inputs



def gated_linear_units(inputs):
  input_shape = inputs.get_shape().as_list()
  assert len(input_shape) == 3
  input_pass = inputs[:,:,0:int(input_shape[2]/2)]
  input_gate = inputs[:,:,int(input_shape[2]/2):]
  input_gate = tf.sigmoid(input_gate)
  return tf.multiply(input_pass, input_gate)




next_layer= tf.placeholder('float',[None,10,128])
res_inputs = next_layer
next_layer = conv1d_weightnorm(inputs=next_layer, layer_idx=1, out_dim=128 * 2,
                                       kernel_size=3, padding="SAME", dropout=1,
                                       var_scope_name="conv_layer_" + str(1))
print(next_layer)
next_layer = tf.contrib.layers.conv2d(inputs=next_layer,num_outputs=128*2,kernel_size=3,
                                              padding="SAME",   weights_initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(4 * 1 / (3 * next_layer.get_shape().as_list()[-1]))),biases_initializer=tf.zeros_initializer(),activation_fn=None,scope="conv_layer_"+str(3))
print(next_layer)
next_layer = gated_linear_units(next_layer)
next_layer = (next_layer + res_inputs) * tf.sqrt(0.5)
print(next_layer)
next_layer =  tf.nn.pool(next_layer, [2], 'MAX', 'SAME', strides = [1])
print(next_layer)
layer_idx=2
inputs=tf.placeholder('float',[None,10,128])
kernel_size=3
out_dim=256
with tf.variable_scope("conv_layer_" + str(layer_idx)):
    in_dim =  int(inputs.get_shape()[-1])
    V = tf.get_variable('V', shape=[kernel_size, in_dim, out_dim], dtype=tf.float32,
                    initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(
                    4.0 * 1 / (kernel_size * in_dim))), trainable=True)
    V_norm = tf.norm(V.initialized_value(), axis=[0, 1])  # V shape is M*N*k,  V_norm shape is k
    g = tf.get_variable('g', dtype=tf.float32, initializer=V_norm, trainable=True)
    b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)

    # use weight normalization (Salimans & Kingma, 2016)
    W = tf.reshape(g, [1, 1, out_dim]) * tf.nn.l2_normalize(V, [0, 1])
    inputs = tf.nn.bias_add(tf.nn.conv1d(value=inputs, filters=W, stride=1, padding='SAME'), b)
    next_layer = tf.contrib.layers.conv2d(inputs=inputs, num_outputs=256, kernel_size=3,
                                          padding="SAME", weights_initializer=tf.random_normal_initializer(mean=0,
                                           stddev=tf.sqrt(4 *0.9 / (3 *next_layer.get_shape().as_list()[-1]))),
                                          biases_initializer=tf.zeros_initializer(), activation_fn=None,
                                          scope="conv_layer_" + str(layer_idx))

    next_layer = gated_linear_units(next_layer)
    next_layer = (next_layer + res_inputs) * tf.sqrt(0.5)
print(W)
print(inputs)
print(b)
print(next_layer)

