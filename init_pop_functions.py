#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 17:24:13 2018

@author: maximoskaliakatsos-papakostas
"""

# assuming padding of 32

import numpy as np

def bin_mat_to_array(b, padding=32):
    a = []
    for i in range(b.shape[1]-2*padding):
        if b[0, padding+i] == 0:
            # + 1 since 0 is preserved for "no event"
            a.append( np.nonzero( b[:, padding+i] )[0][0] + 1 )
        else:
            a.append( 0 )
    return a

def get_random_array_from_piece(p, length):
    idx = np.random.randint( len(p) )
    m = bin_mat_to_array( p[idx].binary_matrix )
    size_diff = length - len(m)
    if size_diff > 0:
        for i in range(size_diff):
            m.append(0)
    return m

def get_specific_array_from_piece(p, idx, length):
    m = bin_mat_to_array( p[idx].binary_matrix )
    size_diff = length - len(m)
    if size_diff > 0:
        for i in range(size_diff):
            m.append(0)
    return m

def get_specific_array_by_name(p, name, length):
    tmp_idx = []
    for idx, i in enumerate(p):
        if i.name == name:
            tmp_idx = idx
    m = bin_mat_to_array( p[tmp_idx].binary_matrix )
    size_diff = length - len(m)
    if size_diff > 0:
        for i in range(size_diff):
            m.append(0)
    return m

def array_to_bin_mat(m):
    b = np.zeros( (37, m.size) )
    b[0,:] = 1
    for i in range(len(m)):
        if m[i] > 0:
            b[ 0, i ] = 0
            b[ m[i], i ] = 1
    return b