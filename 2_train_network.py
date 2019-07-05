#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 09:18:26 2018

@author: maximoskaliakatsos-papakostas
"""

# import music21 as m21
import os
import numpy as np
# import matplotlib.pyplot as plt
# import pickle
import tensorflow as tf
# import data2midi as d2m

cwd = os.getcwd()

main_path = cwd

# npz_data = np.load('saved_data/training_data.npz')
npz_data = np.load('saved_data/training_data.npz')

max_len = npz_data['max_len']
batch_size = npz_data['batch_size']
step = npz_data['step']
input_rows = npz_data['input_rows']
output_rows = npz_data['output_rows']
train_data = npz_data['train_data']
target_data = npz_data['target_data']

num_units = [128, 256, 128]
learning_rate = 0.001
epochs = 2000
temperature = 1.0

all_training_errors = np.zeros(epochs)

tf.reset_default_graph()

# LSTM ========================================================================
def rnn(x, weight, bias, input_rows):
    '''
     define rnn cell and prediction
    '''
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, input_rows])
    x = tf.split(x, max_len, 0)
    
    cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n) for n in num_units]
    stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)
    # cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)
    # outputs, states = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32)
    outputs, states = tf.contrib.rnn.static_rnn(stacked_rnn_cell, x, dtype=tf.float32)
    prediction = tf.matmul(outputs[-1], weight) + bias
    return prediction
# end rnn

'''
def sample(predicted_in):
    # keep only positive
    predicted = np.zeros(len(predicted_in))
    passes = np.where(predicted_in >= 0.0)[0]
    next_event = np.zeros( (len(predicted),1) )
    if len(passes) > 4:
        # get the 4 most possible events
        predicted[ passes ] = predicted_in[ passes ]
        passes = predicted.argsort()[-4:][::1]
    elif len(passes) > 0:
        passes = passes[0:np.min([4, len(passes)])]
    
    next_event[passes] = 1
    return next_event
# end sample
'''
def sample(predicted_in):
    exp_predicted = np.exp(predicted_in/temperature)
    predicted = exp_predicted / np.sum(exp_predicted)
    probabilities = np.random.multinomial(1, predicted, 1)
    return probabilities
# end sample

x = tf.placeholder("float", [None, max_len, input_rows])
y = tf.placeholder("float", [None, output_rows])
weight = tf.Variable(tf.random_normal([num_units[-1], output_rows]))
bias = tf.Variable(tf.random_normal([output_rows]))

prediction = rnn(x, weight, bias, input_rows)
dist = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
# dist = tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y)
cost = tf.reduce_mean(dist)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

num_batches = int(len(train_data)/batch_size)

min_cost = 10000.0
min_save_epoch = 10

for i in range(epochs):
    print("----------- Epoch", str(i+1), "/", str(epochs), " -----------")
    count = 0
    for i_batch in range(num_batches):
        # print("batch: ", str(i_batch+1), "/", str(num_batches), "epoch: ", str(i+1), "/", str(epochs))
        train_batch, target_batch = train_data[count:count+batch_size], target_data[count:count+batch_size]
        count += batch_size
        sess.run([optimizer] ,feed_dict={x:train_batch, y:target_batch})
    cost_value = sess.run([cost] ,feed_dict={x:train_data, y:target_data})
    print( "cost_value: ", cost_value[0] )
    all_training_errors[i] = cost_value[0]
    if min_cost > cost_value[0] and i > min_save_epoch:
        tmpW_1 = sess.run(weight)
        min_cost = cost_value[0]
        print("saving model")
        # save model
        all_vars = tf.global_variables()
        saver = tf.train.Saver()
        saver.save(sess, 'saved_model/file.ckpt')
        directory = 'all_saved_models/epoch_' + str(i) + '/saved_model/'
        if not os.path.exists(directory):
            os.makedirs(directory)
            saver.save(sess, directory + 'file.ckpt')


# save training errors
np.savez('saved_results/training_erros.npz', all_training_errors=all_training_errors)