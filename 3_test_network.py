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
import pickle
import tensorflow as tf
import data2midi as d2m

cwd = os.getcwd()

main_path = cwd

npz_data = np.load('saved_data/training_data.npz')
with open('saved_data/melodies.pickle', 'rb') as handle:
    p = pickle.load(handle)

max_len = npz_data['max_len']
batch_size = npz_data['batch_size']
step = npz_data['step']
input_rows = npz_data['input_rows']
output_rows = npz_data['output_rows']
train_data = npz_data['train_data']
target_data = npz_data['target_data']

num_units = [128, 256, 128]
'''
learning_rate = 0.001
epochs = 100
'''
temperature = 0.5

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
    print('predicted_in: ', predicted_in)
    # exp_predicted = np.exp(predicted_in/temperature)
    # predicted = exp_predicted / np.sum(exp_predicted)
    predicted_distr = predicted_in - np.min(predicted_in)
    predicted_distr = predicted_distr/np.sum(predicted_distr)
    predicted_distr = np.power(predicted_distr, 10)
    predicted_distr = predicted_distr/np.sum(predicted_distr)
    print('predicted: ', predicted_distr)
    probabilities = np.random.multinomial(1, predicted_distr, size=1)
    print('probabilities:, ', probabilities)
    return probabilities
# end sample
'''
def sample(predicted_in):
    norm_predicted = predicted_in - np.min(predicted_in)
    predicted = norm_predicted / np.sum(norm_predicted)
    print('predicted: ', predicted)
    probabilities = np.random.multinomial(1, predicted, size=1)
    print('probabilities:, ', probabilities)
    return probabilities
# end sample
'''
'''
def sample(predicted_in):
    predicted = np.zeros( predicted_in.shape )
    predicted[np.argmax(predicted_in)] = 1
    print('predicted: ', predicted)
    probabilities = np.array( predicted )
    print('probabilities:, ', probabilities)
    return probabilities
# end sample
'''
x = tf.placeholder("float", [None, max_len, input_rows])
y = tf.placeholder("float", [None, output_rows])
weight = tf.Variable(tf.random_normal([num_units[-1], output_rows]))
bias = tf.Variable(tf.random_normal([output_rows]))

prediction = rnn(x, weight, bias, input_rows)
dist = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
cost = tf.reduce_mean(dist)

# restore saved model
sess = tf.Session()
saver = tf.train.Saver()
# saver.restore(sess, 'saved_model/file.ckpt')
saver.restore(sess, 'all_saved_models/epoch_25/saved_model/file.ckpt')
# init_op = tf.global_variables_initializer()
# sess.run(init_op)

# GENERATE
# generate seed
seed = train_data[:1:]
composition = np.array(seed[0,:,:]).transpose()

for i in range(64):
    print('i: ', i)
    if i > 0:
        remove_fist_event = seed[:,1:,:]
        new_input = predicted_output
        seed = np.append(remove_fist_event, np.reshape(new_input, [1, 1, input_rows]), axis=1)
    predicted = sess.run([prediction], feed_dict = {x:seed})
    predicted = np.asarray(predicted[0]).astype('float64')[0]
    predicted_output = sample(predicted)
    composition = np.hstack( (composition, predicted_output.T) )

# make midi of composition
composition = composition[:, 32:]
d2m.onehot2midi(composition, 'network.midi')