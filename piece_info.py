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
        self.offsets = np.array( [f.offset for f in notes] )
        # make durations and offsets of pieces comparable
        self.offsets = self.offsets/( 2.0*np.median( np.diff( self.offsets ) ) )
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
        # binary one hot matrix
        self.binary_matrix = []
    # end constructor
    
    def make_matrix(self):
        # assuming three octaves and final note of quarter duration
        # assuming that the 0-th row is for indicating no event
        b = np.zeros( (37, int(4*self.offsets[-1] + 1)) )
        b[0,:] = 1
        for idx, i in enumerate(self.offsets):
            b[ 0, int(4*i) ] = 0
            b[ int(self.pitches_c[idx]-60 + 1) , int(4*i) ] = 1
        self.binary_matrix = b
    # end make_matrix
    
    def make_matrix_padding(self, padding=32):
        # assuming three octaves and final note of quarter duration
        # assuming that the 0-th row is for indicating no event
        b = np.zeros( (37, 2*padding + int(4*self.offsets[-1] + 1)) )
        b[0,:] = 1
        for idx, i in enumerate(self.offsets):
            b[ 0, padding+int(4*i) ] = 0
            b[ int(self.pitches_c[idx]-60 + 1) , padding+int(4*i) ] = 1
        self.binary_matrix = b
    # end make_matrix_padding