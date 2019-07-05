#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 00:18:16 2018

@author: maximoskaliakatsos-papakostas
"""

import music21 as m21
import numpy as np
import melody_features as mf

class PieceClass:
    'All necessary information for blending, GA and LSTM processing of pieces'
    
    def __init__(self, filePath):
        self.name = filePath.split('/')[-1].split('.')[0]
        m21piece = m21.converter.parse(filePath)
        m21piece.quantize([4], processOffsets=True, processDurations=True, inPlace=True)
        notes = m21piece.flat.notes
        # get important info
        self.pitches = np.array( [m.pitch.midi for m in notes] )
        self.durations = np.array( [d.duration.quarterLength for d in notes] )
        self.offsets = np.array( [f.offset for f in notes] )
        # transpose to normalised key
        self.key = m21piece.analyze('key')
        # self.key.show('text')
        kpc = self.key.tonic.pitchClass
        if self.key.type == 'major':
            if kpc <= 6:
                self.pitches_c = self.pitches - kpc
            else:
                self.pitches_c = self.pitches - kpc + 12
        else:
            if kpc <= 3:
                self.pitches_c = self.pitches - kpc - 3
            else:
                self.pitches_c = self.pitches - kpc + 12 - 3
        # we not only have to transpose, but also use a common lowest note
        p_min = np.min(self.pitches_c)
        # adjust integer difference in octaves
        diff_oct_min = np.floor( (p_min - 60)/12 )
        self.pitches_c = self.pitches_c - diff_oct_min*12
        # pitch class profile
        self.pcp = np.histogram( np.mod(self.pitches_c, 12), bins=12 )[0]
        # features
        self.features = mf.compute_melody_features(self)
        # make pitch-duration tuples for LSTM
        pdt = []
        t = np.array( (self.pitches_c, self.durations) ).T
        for w in t:
            pdt.append( tuple(w) )
        self.pitch_duration = pdt
        # binary one hot matrix
        self.binary_matrix = []
        self.feature_matrix = []
        self.composite_matrix = []
        self.pd_to_oh_dict = []
        self.oh_to_pd_dict = []
    # end constructor
    
    def make_matrix(self, pd_to_oh_dict, oh_to_pd_dict):
        self.pd_to_oh_dict = pd_to_oh_dict
        self.oh_to_pd_dict = oh_to_pd_dict
        b = np.zeros( (len(self.oh_to_pd_dict), int(4*self.offsets[-1]+4*self.durations[-1])) )
        # we know that the first word is going to be (0,0), so in the one-hot encoding
        # we know that [1,0,0...,0] will represent this word
        b[0,:] = 1
        f = np.zeros( ( len(self.features) , int(4*self.offsets[-1]+4*self.durations[-1]) ) )
        for idx, i in enumerate(self.offsets):
            b[:, int(4*i)] = ( np.array( self.pd_to_oh_dict[self.pitch_duration[idx]] ) )
        for i in range(f.shape[1]):
            f[:, i] = self.features
        self.binary_matrix = b
        self.feature_matrix = f
        self.composite_matrix = np.vstack( (f, b) )