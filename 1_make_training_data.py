#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 09:18:26 2018

@author: maximoskaliakatsos-papakostas
"""

# import music21 as m21
import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# import tensorflow as tf
# import data2midi as d2m

main_path = '/Users/maximoskaliakatsos-papakostas/Documents/python/melody_blending_deep/simple_evo'

npz_data = np.load('saved_data/initial_data.npz')
b_in = npz_data['b_in']
b_out = npz_data['b_out']
'''
with open('saved_data/melodies.pickle', 'rb') as handle:
    p = pickle.load(handle)
with open('saved_data/pd_to_oh_dict.pickle', 'rb') as handle:
    pd_to_oh_dict = pickle.load(handle)
with open('saved_data/oh_to_pd_dict.pickle', 'rb') as handle:
    oh_to_pd_dict = pickle.load(handle)
'''

max_len = 32
batch_size = 320
step = 1
input_rows = b_in.shape[0]
output_rows = b_out.shape[0]

# divide data in input-output pairs
input_mats = []
output_mats = []

for i in range(0, b_in.shape[1] - max_len, step):
    input_mats.append(b_in[:,i:i+max_len])
    output_mats.append(b_out[:,i+max_len])

# make training and testing tensors
train_data = np.zeros((len(input_mats), max_len, input_rows))
target_data = np.zeros((len(output_mats), output_rows))
for i in range(len(input_mats)):
    train_data[i,:,:] = input_mats[i].transpose()
    target_data[i,:] = output_mats[i]

# make batches
num_batches = int( np.size(b_in, axis=0)/batch_size )
count = 0
all_batches = [] # actually don't need it
for _ in range(num_batches):
    train_batch, target_batch = train_data[count:count+batch_size], target_data[count:count+batch_size]
    count += batch_size

np.savez('saved_data/training_data.npz', max_len=max_len, batch_size=batch_size, step=step, input_rows=input_rows, output_rows=output_rows, train_data=train_data, target_data=target_data)