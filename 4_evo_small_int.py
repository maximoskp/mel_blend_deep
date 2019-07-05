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

netEval = ne.NetEval('epoch_394')
# highest pcp
# target_features = [0.85, 1.91, 0.00, 0.21, 0.54, 0.76]
# highest small intervals
target_features = [0.85, 0.69, 1.00, 0.21, 0.54, 0.76]

main_path = '/Users/maximoskaliakatsos-papakostas/Documents/python/melody_blending_deep/evo_feature_LSTM'

# npz_data = np.load('saved_data/training_data.npz')
with open('saved_data/melodies.pickle', 'rb') as handle:
    p = pickle.load(handle)
with open('saved_data/pd_to_oh_dict.pickle', 'rb') as handle:
    pd_to_oh_dict = pickle.load(handle)
with open('saved_data/oh_to_pd_dict.pickle', 'rb') as handle:
    oh_to_pd_dict = pickle.load(handle)


# evo
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# toolbox.register("attr_bool", ipf.get_random_array_from_piece, p)
toolbox.register("attr_bool", random.randint, 0, len(oh_to_pd_dict)-1)

toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_bool, 64)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalMelody(individual):
    # print('evaluating')
    return (netEval.eval_feats_nn_integer_mel(individual, oh_to_pd_dict, target_features), )

toolbox.register("evaluate", evalMelody)

toolbox.register("mate", tools.cxTwoPoint)
# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# evolution loop
nPop = 300
print('initialising population')
pop = toolbox.population(n=nPop)
for i in range(nPop):
    if i < int(nPop/3):
        tmpInd = ipf.get_specific_array_by_name(p, 'han0160', 64)
    elif i < int(2*nPop/3):
        tmpInd = ipf.get_specific_array_by_name(p, 'deut3844', 64)
    else:
        tmpInd = ipf.get_random_array_from_piece(p, 64)
    for j in range(64):
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
nGen = 200

CXPB, MUTPB = 0.5, 0.2

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
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    print("  Evaluated %i individuals" % len(invalid_ind))
    
    # The population is entirely replaced by the offspring
    pop[:] = offspring
    
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

print("-- End of (successful) evolution --")

best_ind = tools.selBest(pop, 1)[0]
print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

composition = ipf.array_to_bin_mat(np.array(best_ind), oh_to_pd_dict)

'''
# make midi of composition
with open('saved_data/oh_to_pd_dict.pickle', 'rb') as handle:
    oh_to_pd_dict = pickle.load(handle)
'''
d2m.onehot2midi(composition, oh_to_pd_dict, 'int_decrease.midi')
# save result
np.savez('saved_results/best_int.npz', best_ind=best_ind)

'''
d2m.onehot2midi(p[32].binary_matrix, oh_to_pd_dict, 'lowest_pcp.midi')
d2m.onehot2midi(p[21].binary_matrix, oh_to_pd_dict, 'highest_pcp.midi')
'''
