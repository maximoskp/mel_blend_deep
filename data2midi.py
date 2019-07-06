#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 23:50:50 2018

@author: maximoskaliakatsos-papakostas
"""

import music21 as m21
import numpy as np
import os

cwd = os.getcwd()

def onehot2midi(bmt, fileName, padding=0):
    sc = m21.stream.Stream()
    # find all sums to get offsets
    s = np.sum(bmt[1:,:], axis=0)
    nzs = np.nonzero(s)[0]
    print('nzs.shape: ', nzs.shape)
    dur_array = np.diff( nzs )
    # assume that the last note has quarter duration
    dur_array = np.append( dur_array, 4 )
    print('dur_array.shape: ', dur_array.shape)
    dur_idx = 0
    for i in range(padding, bmt.shape[1]-padding, 1):
        if bmt[0,i] == 0:
            tmpPitch = 60 + int( np.nonzero( bmt[:,i] )[0]) - 1
            tmpNote = m21.note.Note(tmpPitch)
            print('i: ', i, ' - dur_idx: ', dur_idx)
            tmpDur = dur_array[dur_idx]
            tmpNote.duration = m21.duration.Duration(tmpDur/4.0)
            dur_idx += 1
            sc.insert(0.25*(i-padding), tmpNote)
            # print(tmpPitch,' - ', tmpDur/4.0,' - ', 0.25*i)
    mf = m21.midi.translate.streamToMidiFile(sc)
    destination = cwd
#     destination="/Users/maximoskaliakatsos-papakostas/Documents/python/melody_blending_deep/simple_evo/"
    mf.open(destination + fileName, 'wb')
    mf.write()
    mf.close()
    return sc