#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 09:09:22 2018

@author: maximoskaliakatsos-papakostas
"""

# import piece_info as pi
import numpy as np
import scipy.stats as sp

def compute_melody_features(m):
    n = m.pitches_c
    t = m.offsets
    pcp = m.pcp
    f = np.zeros(6)
    # rhythm inhomogeneity
    f[0] = np.std( np.diff(t) )/np.mean( np.diff(t) )
    # pcp complexity
    f[1] = sp.entropy( pcp )
    # small intervals percentage
    d = np.diff(n)
    dd = d[ d!=0 ]
    f[2] = np.sum(np.abs(dd) < 3)/np.size(dd)
    # range as percentage of two octaves
    f[3] = (np.max(n)-np.min(n))/24
    # or maybe rhythm density
    # f[3] = np.size(t)/(4*t[-1])
    # percentage of note repetitions
    f[4] = np.sum(d == 0)/np.size(d)
    # percentage of steady time intervals
    d = np.diff(t)
    dd = np.diff(d)
    f[5] = np.sum(dd==0)/np.size(dd)
    
    return f

def compute_array_melody_features(m):
    f = np.zeros(6)
    n = []
    t = []
    for idx, i in enumerate(m):
        if i > 0:
            n.append( 60 + i - 1 )
            t.append( idx*0.25 )
    pcp = np.histogram( np.mod(n, 12), bins=12 )[0]
    # rhythm inhomogeneity
    f[0] = np.std( np.diff(t) )/np.mean( np.diff(t) )
    # pcp complexity
    f[1] = sp.entropy( pcp )
    # small intervals percentage
    d = np.diff(n)
    dd = d[ d!=0 ]
    f[2] = np.sum(np.abs(dd) < 3)/np.size(dd)
    # range as percentage of two octaves
    f[3] = (np.max(n)-np.min(n))/24
    # or maybe rhythm density
    # f[3] = np.size(t)/(4*t[-1])
    # percentage of note repetitions
    f[4] = np.sum(d == 0)/np.size(d)
    # rhythm density
    dens = np.sum( np.array(m)>0 )/np.array(m).size
    f[5] = dens
    
    return f