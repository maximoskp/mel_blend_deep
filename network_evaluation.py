#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 00:18:16 2018

@author: maximoskaliakatsos-papakostas
"""

import numpy as np
import tensorflow as tf
import scipy.stats as sp
import melody_features as mf

class NetEval:
    'Loading a tensorflow model, dictionaries and (neutralised) seed and evaluating a given melody'
    
    def __init__(self, model_folder):
        self.main_path = '/Users/maximoskaliakatsos-papakostas/Documents/python/melody_blending_deep/simple_evo'
        self.npz_data = np.load('saved_data/training_data.npz')
        self.train_data = self.npz_data['train_data']
        self.initial_seed = self.train_data[:1:]
        self.max_len = self.npz_data['max_len']
        self.input_rows = self.npz_data['input_rows']
        self.output_rows = self.npz_data['output_rows']
        self.num_units = [128, 256, 128]
        # model
        tf.reset_default_graph()
        
        self.x = tf.placeholder("float", [None, self.max_len, self.input_rows])
        self.y = tf.placeholder("float", [None, self.output_rows])
        self.weight = tf.Variable(tf.random_normal([self.num_units[-1], self.output_rows]))
        self.bias = tf.Variable(tf.random_normal([self.output_rows]))
        
        self.prediction = self.rnn(self.x, self.weight, self.bias, self.input_rows)
        
        # restore saved model
        self.sess = tf.Session()
        saver = tf.train.Saver()
        # saver.restore(sess, 'saved_model/file.ckpt')
        saver.restore(self.sess, 'all_saved_models/'+model_folder+'/saved_model/file.ckpt')
    # end constructor
    
    def rnn(self, x, weight, bias, input_rows):
        '''
         define rnn cell and prediction
        '''
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, input_rows])
        x = tf.split(x, self.max_len, 0)
        cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n) for n in self.num_units]
        stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)
        # cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)
        # outputs, states = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32)
        outputs, states = tf.contrib.rnn.static_rnn(stacked_rnn_cell, x, dtype=tf.float32)
        # cell = tf.contrib.rnn.BasicLSTMCell(self.num_units, forget_bias=1.0)
        # outputs, states = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32)
        prediction = tf.matmul(outputs[-1], weight) + bias
        return prediction
    # end rnn
    
    def eval_nn_integer_mel(self, m):
        'm is a list of integers that will be translated to 1-hot for evaluation'
        seed = self.initial_seed
        tmpSum = 0
        for i in range(len(m)):
            if i > 0:
                remove_fist_event = seed[:,1:,:]
                new_input = melody_output
                seed = np.append(remove_fist_event, np.reshape(new_input, [1, 1, self.input_rows]), axis=1)
            predicted = self.sess.run([self.prediction], feed_dict = {self.x:seed})
            predicted = np.asarray(predicted[0]).astype('float64')[0]
            predicted_distr = predicted - np.min(predicted)
            predicted_distr = predicted_distr/np.sum(predicted_distr)
            # do a bit of softmax
            predicted_distr = np.power(predicted_distr, 10)
            predicted_distr = predicted_distr/np.sum(predicted_distr)
            melody_output = np.zeros(self.input_rows)
            melody_output[ m[i] ] = 1
            kl = sp.entropy(melody_output, predicted_distr)
            if kl > 15:
                kl = 100
            tmpSum += kl
        return tmpSum/len(m)
    # end make_matrix_padding
    
    def eval_feats_nn_integer_mel(self, m, target_features):
        f = mf.compute_array_melody_features(m)
        nn_eval = max( self.eval_nn_integer_mel(m) , 0)
        f = np.append( f, nn_eval )
        target_features = np.append( target_features, 0.0 )
        nn_contrib = 1.0/7.0
        d = f - target_features
        d[0:6] = (1-nn_contrib)*d[0:6]
        d[6] = nn_contrib*d[6]
        # weigh each feature according to max
        w = np.array( [0.97650401, 1.96652508, 0.94117647, 0.83333333, 0.60526316, 0.83673469, 2.0] )
        d = np.divide(d, w)
        n = np.linalg.norm( d )
        '''
        # add a penalty for very few notes
        dens = np.sum( np.array(m)>0 )/np.array(m).size
        p = 5.0*abs(dens - 0.3)
        if dens > 0.3:
            p = 0
        return n+p
        '''
        return n
    
    def get_feats_nn_integer_mel(self, m):
        f = mf.compute_array_melody_features(m)
        nn_eval = max( self.eval_nn_integer_mel(m) , 0)
        f = np.append( f, nn_eval )
        return f