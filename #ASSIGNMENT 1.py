# ASSIGNMENT 1
import numpy as np
import math
import random as rd
rd.seed(10)


def population(population_size, length):
    # define the individuals table
    population = population_size*[length*[0]]
    for i in range(population_size):
        for j in range(length):
            population[i][j] = rd.randint(0, 1)

    return population


pop = population(20, 10)
print(pop)


def fitness(population):
    # calculate fitness
    fitness = []
    for i in range(20):
        fitness.append(sum(population[i]))
    return fitness


def probability_of_selection(fitness):
    # calculate the probability of selection
    probability_of_selection = []
    sum_of_fitness = sum(fitness)
    if sum_of_fitness != 0:
        for i in range(len(fitness)):
            probability_of_selection.append(fitness[i]/sum_of_fitness)
    else:
        probability_of_selection = len(fitness)*[0]
    return probability_of_selection


def cumulative_probability_of_selection(prob_of_selection):
    #  calculate the cumulative probability of selection

    cumulative = np.zeros_like(prob_of_selection)
    cumulative[0] = prob_of_selection[0]

    for i in range(1, len(prob_of_selection)):
        cumulative[i] = cumulative[i-1]+prob_of_selection[i]

    return cumulative


def Roulette_Wheel(cumulative):
    R = np.random.random()
    for i in range(len(cumulative)):
        if R <= cumulative[i]:
            return i


def Roulette_Wheel_selection(cumulative, population):

    parents_matrix = np.zeros((2, np.size(population, 1)))
    parent1_indx = Roulette_Wheel(cumulative)
    parent2_indx = Roulette_Wheel(cumulative)
    parents_matrix[0] = population[parent1_indx]
    parents_matrix[1] = population[parent2_indx]
    return parents_matrix


def binCross(twoParents, pcross, clen):
    R = rd.random()
    print(R)
    twochildren = twoParents.copy()  # deep copy not shallow copy
    if R < pcross:
        cutting_point = rd.randint(1, 5)
       #cutting_point = math.floor(cutting_point)
        print("cutting point: ", cutting_point)
        for i in range(1, clen):
            if i == cutting_point:
                twochildren[0, i:] = twoParents[1, i:]
                twochildren[1, i:] = twoParents[0, i:]
                return twochildren
    else:
        return twochildren


def binMutate(individual, pmute, clen):
    mutatedInd = individual.copy()
    for i in range(clen):
        R = rd.random()
        if R < pmute:
            mutatedInd[i] = 1-individual[i]
    return mutatedInd


def Elitism(pop=[], fitness=[]):
    elitism = np.zeros((2, np.size(pop, 1)))
    fit_copy = fitness.copy()
    fit_copy.sort(reverse=True)
    max_fitness = fit_copy[0]
    second_max_fitness = fit_copy[1]
    for i in range(len(fitness)):
        if fitness[i] == max_fitness:
            indx1 = i
        if fitness[i] == second_max_fitness:
            indx2 = i
    elitism[0, :] = pop[indx1]
    elitism[1, :] = pop[indx2]
    fitness.remove(max_fitness)
    fitness.remove(second_max_fitness)
    # to delete row(axis 0 = horizontal = row) of indx1 in matrix pop
    pop = np.delete(pop, indx1, 0)
    pop = np.delete(pop, indx2-1, 0)

    return elitism


def runBinGA(npop, clen, ngen, pcross, pmute):  # without elitism

    best_hist = []
    avg_best_hist = 0
    pop = population(npop, clen)
    new_generation = np.zeros_like(pop)
    fit = fitness(pop)
    for i in range(ngen):
        probs = probability_of_selection(fit)
        cumulative = cumulative_probability_of_selection(probs)
        for j in range(npop, 2):
            twoparents = Roulette_Wheel_selection(cumulative, pop)
            twochildren = binCross(twoparents, pcross, clen)
            new_generation[i, :] = twochildren[0, :]
            new_generation[i+1, :] = twochildren[1, :]
        for k in range(npop):
            mutedInd = binMutate(new_generation[k, :], pmute, clen)
            new_generation[k, :] = mutedInd
        pop = new_generation.copy()
        fit = fitness(pop)
        best_hist.append(max(fit))
        avg_best_hist = (sum(best_hist)/len(best_hist))

    return new_generation, best_hist, avg_best_hist


# print(runBinGA(20, 5, 100, 0.6, 0.05))  # to run the code without elitism


def runBinGA_with_Elitism(npop, clen, ngen, pcross, pmute):

    best_hist = []
    avg_best_hist = 0
    pop = population(npop, clen)
    new_generation = np.zeros_like(pop)
    fit = fitness(pop)
    elitism = Elitism(pop, fit)
    new_generation[0, :] = elitism[0, :]
    new_generation[1, :] = elitism[1, :]
    for i in range(ngen):
        probs = probability_of_selection(fit)
        cumulative = cumulative_probability_of_selection(probs)
        for j in range(npop-2, 2):
            twoparents = Roulette_Wheel_selection(cumulative, pop)
            twochildren = binCross(twoparents, pcross, clen)
            new_generation[i+2, :] = twochildren[0, :]
            new_generation[i+3, :] = twochildren[1, :]
        for k in range(npop-2):
            mutedInd = binMutate(new_generation[k, :], pmute, clen)
            new_generation[k+2, :] = mutedInd
        pop = new_generation.copy()
        fit = fitness(pop)
        best_hist.append(max(fit))
        avg_best_hist = math.floor(sum(best_hist)/len(best_hist))
        elitism = Elitism(pop, fit)
        new_generation[0, :] = elitism[0, :]
        new_generation[1, :] = elitism[1, :]

    return new_generation, best_hist, avg_best_hist


# to run the code with elitism
#print(runBinGA_with_Elitism(20, 5, 100, 0.6, 0.05))
