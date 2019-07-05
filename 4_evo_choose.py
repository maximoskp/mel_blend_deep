#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 09:18:26 2018

@author: maximoskaliakatsos-papakostas
"""

# import music21 as m21
import os
import numpy as np
import random
from deap import base
from deap import creator
from deap import tools
# import matplotlib.pyplot as plt
import pickle
import network_evaluation as ne
import init_pop_functions as ipf
# import tensorflow as tf
import data2midi as d2m
import melody_features as mf

mel_length = 64

netEval = ne.NetEval('epoch_201')

'''
# f han and deut
fhan = [0.70, 0.68, 0.00, 0.21, 0.61, 0.78]
fdeut = [0.34, 1.97, 0.79, 0.54, 0.24, 0.78]
# highest pcp
# target_features = [0.85, 1.91, 0.00, 0.21, 0.54, 0.76]
target_features = [0.70, 1.97, 0.00, 0.21, 0.61, 0.78]
# highest small intervals
# target_features = [0.85, 0.69, 1.00, 0.21, 0.54, 0.76]
'''

cwd = os.getcwd()

main_path = cwd

# npz_data = np.load('saved_data/training_data.npz')
with open('saved_data/melodies.pickle', 'rb') as handle:
    p = pickle.load(handle)

all_feats = []
mel_all_feats = []
for p_i in p:
    all_feats.append( mf.compute_melody_features(p_i) )
    mel_all_feats.append( netEval.get_feats_nn_integer_mel( ipf.bin_mat_to_array(p_i.binary_matrix) ) )
f = np.array( all_feats )
f_max = np.max(f, axis=0)
f_min = np.min(f, axis=0)

melody_features = np.array(mel_all_feats)
np.savez('saved_data/melody_features.npz', melody_features=melody_features)

target_features = [f_min[0], f_max[1], f_max[2], f_max[3], f_min[4], f_max[5]]
'''
[0.25983113920591533,
 1.9669382871780208,
 0.9411764705882353,
 1.0,
 0.0,
 0.8367346938775511]
'''

# evo
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# toolbox.register("attr_bool", ipf.get_random_array_from_piece, p)
toolbox.register("attr_bool", random.randint, 0, p[0].binary_matrix.shape[0]-1)

toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_bool, mel_length)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalMelody(individual):
    # print('evaluating')
    return (netEval.eval_feats_nn_integer_mel(individual, target_features), )

toolbox.register("evaluate", evalMelody)

toolbox.register("mate", tools.cxTwoPoint)
# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
nPop = 100
toolbox.register("mutate1", tools.mutShuffleIndexes, indpb=0.1)
toolbox.register("mutate2", tools.mutUniformInt, low=0, up=36, indpb=0.1)
# toolbox.register("select", tools.selTournament, tournsize=nPop)
toolbox.register("select", tools.selRoulette)
toolbox.register("select_new", tools.selBest)

# evolution loop
# nPop = 100
print('initialising population')
pop = toolbox.population(n=nPop)
'''
for i in range(nPop):
    if i < int(nPop/3):
        tmpInd = ipf.get_specific_array_by_name(p, 'han0160', mel_length)
    elif i < int(2*nPop/3):
        tmpInd = ipf.get_specific_array_by_name(p, 'deut3968', mel_length)
    else:
        tmpInd = ipf.get_random_array_from_piece(p, mel_length)
    for j in range(mel_length):
        pop[i][j] = tmpInd[j]

for i in range(nPop):
    if i < int(nPop/2):
        tmpInd = ipf.get_specific_array_by_name(p, 'han0160', mel_length)
    else:
        tmpInd = ipf.get_specific_array_by_name(p, 'deut3968', mel_length)
    for j in range(mel_length):
        pop[i][j] = tmpInd[j]
'''
for i in range(nPop):
    tmpInd = ipf.get_random_array_from_piece(p, mel_length)
    for j in range(mel_length):
        pop[i][j] = tmpInd[j]

print('initialised')

# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

print("  Evaluated %i individuals" % len(pop))

# Extracting all the fitnesses of 
fits = [ind.fitness.values[0] for ind in pop]

# Variable keeping track of the number of generations
g = 0
nGen = 100

CXPB, MUT1PB, MUT2PB = 0.9, 0.1, 0.1

all_best_feats = []

while max(fits) > 0.0001 and g < nGen:
    # A new generation
    g = g + 1
    print("-- Generation %i --" % g)
    
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):

        # cross two individuals with probability CXPB
        if random.random() < CXPB:
            toolbox.mate(child1, child2)

            # fitness values of the children
            # must be recalculated later
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:

        # mutate an individual with probability MUTPB
        if random.random() < MUT1PB:
            toolbox.mutate1(mutant)
            del mutant.fitness.values
        if random.random() < MUT2PB:
            toolbox.mutate2(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    print("  Evaluated %i individuals" % len(invalid_ind))
    
    # The population is entirely replaced by the offspring
    # pop[:] = offspring
    pop[:] = toolbox.select_new(pop+offspring, k=nPop)
    
    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]
    
    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5
    
    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)
    
    # save intermediate
    best_ind = tools.selBest(pop, 1)[0]
    composition = ipf.array_to_bin_mat(np.array(best_ind))
    d2m.onehot2midi(composition, 'choose_interm.midi')
    best_feats = netEval.get_feats_nn_integer_mel(best_ind)
    print(' best features: ', best_feats)
    all_best_feats.append(best_feats)

print("-- End of (successful) evolution --")

best_ind = tools.selBest(pop, 1)[0]
print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
best_feats = netEval.get_feats_nn_integer_mel(best_ind)
print(' best features: ', best_feats)
all_best_feats.append(best_feats)

composition = ipf.array_to_bin_mat(np.array(best_ind))

'''
# make midi of composition
with open('saved_data/oh_to_pd_dict.pickle', 'rb') as handle:
    oh_to_pd_dict = pickle.load(handle)
'''
d2m.onehot2midi(composition, 'choose.midi')
# save result
all_best_feats = np.array(all_best_feats)
np.savez('saved_results/best_choose.npz', best_ind=best_ind)
np.savez('saved_results/all_choose_feats.npz', all_best_feats=all_best_feats)
