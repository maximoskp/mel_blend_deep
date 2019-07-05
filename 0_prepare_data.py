#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 09:18:26 2018

@author: maximoskaliakatsos-papakostas
"""

# import music21 as m21
import sys
# import os
import glob
import piece_info
import numpy as np
import matplotlib.pyplot as plt
import pickle
# import tensorflow as tf
# import data2midi as d2m

main_path = '/Users/maximoskaliakatsos-papakostas/Documents/python/melody_blending_deep/simple_evo'
chinese_folder = '/data/test_data';
# chinese_folder = '/data/chinese_tests';
# german_folder = '/data/german';

# read chinese data
chinese_docs = glob.glob(main_path+chinese_folder+"/*.krn")
allDocs = chinese_docs
currFolder = chinese_folder

if len(allDocs) < 1:
    sys.exit("run_test.py: No files there!")

# keeping all pieces
p = []
# keeping unique pitch-duration words - start with the "no action" event
unique_words = [tuple(np.array([0,0]))]
# we know that the first word is going to be (0,0), so in the one-hot encoding
# we know that [1,0,0...,0] will represent this word

# parse all pieces
for pieceName in allDocs:
    print("Processing piece: "+pieceName+"... ")
    # p = m21.converter.parse(pieceName)
    p_i = piece_info.PieceClass( pieceName )
    p.append( p_i )

# make matrix for each piece structure
for p_i in p:
    # p_i.make_matrix()
    p_i.make_matrix_padding(padding=32)

# make the unified binary input and output LSTM matrices for all pieces
b_in = p[0].binary_matrix
b_out = p[0].binary_matrix
for i in range(1, len(p)):
    print('i:', i)
    b_in = np.hstack( (b_in, p[i].binary_matrix) )
    b_out = np.hstack( (b_out, p[i].binary_matrix) )
# cut the last element from b_in and the first from b_out
b_in = np.delete(b_in, -1, axis=1)
b_out = np.delete(b_out, 0, axis=1)

fig_testMat = plt.figure()
plt.imshow(b_in, cmap='gray_r', interpolation='none')
fig_testMat.savefig('b_in.eps', format='eps', dpi=1000); plt.clf()

fig_testMat = plt.figure()
plt.imshow(b_out, cmap='gray_r', interpolation='none')
fig_testMat.savefig('b_out.eps', format='eps', dpi=1000); plt.clf()

# save training data
np.savez('saved_data/initial_data.npz', b_in=b_in, b_out=b_out)
with open('saved_data/melodies.pickle', 'wb') as handle:
    pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)

# just for testing midi export
# d2m.onehot2midi(p[0].binary_matrix, p[0].oh_to_pd_dict, 'lala.midi')

'''
LOADING BACK
npz_data = np.load('saved_data/training_data.npz')
b_in = npz_data['b_in']
b_out = npz_data['b_out']
with open('saved_data/melodies.pickle', 'rb') as handle:
    p = pickle.load(handle)
with open('saved_data/pd_to_oh_dict.pickle', 'rb') as handle:
    pd_to_oh_dict = pickle.load(handle)
with open('saved_data/oh_to_pd_dict.pickle', 'rb') as handle:
    oh_to_pd_dict = pickle.load(handle)
'''