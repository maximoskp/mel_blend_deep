#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 09:18:26 2018

@author: maximoskaliakatsos-papakostas
"""

# import music21 as m21
import numpy as np
import matplotlib.pyplot as plt
# import pickle
#import network_evaluation as ne
#import init_pop_functions as ipf
# import tensorflow as tf
# import data2midi as d2m

# choose features
npz_data = np.load('saved_results/best_choose.npz')
best_ind = npz_data['best_ind']
npz_data = np.load('saved_results/all_choose_feats.npz')
all_best_feats = npz_data['all_best_feats']

target_features = [0.2871393034605969,
 1.9665250770314202,
 0.9411764705882353,
 0.8333333333333334,
 0.0,
 0.8367346938775511]

# proximity of explicit features to target
d_explicit = []
w = np.array( [0.97650401, 1.96652508, 0.94117647, 0.83333333, 0.60526316, 0.83673469] )
# proximity of implicit feature/KL to 0
d_implicit = []
d_net_max = np.max(all_best_feats[:,-1])

for i in range(all_best_feats.shape[0]):
    d_explicit.append( np.linalg.norm( np.divide(all_best_feats[i,:(all_best_feats.shape[1]-1)] - target_features, w)) )
    d_implicit.append( all_best_feats[i,-1]/d_net_max )

# normalise distances
d_explicit = np.array(d_explicit)
d_explicit = d_explicit/np.max(d_explicit)
d_implicit = np.array(d_implicit)
d_implicit = d_implicit/np.max(d_implicit)

# plot
plt.plot(d_explicit, '-x', color='black', label='explicit')
plt.plot(d_implicit, '-o', color='grey', label='implicit')
plt.xlabel('generation'); plt.ylabel('distance')
leg = plt.legend(loc='best')
plt.savefig('figs/convergence.eps', format='eps', dpi=1000); plt.clf()