"""
File Name: tsp.py
Author: Calvin D'Souza nt3587
Date Created: 04/06/2021

Open terminal and run -  python /path/tsp.py

Description: Using Genetic Algorithm I will determine the best path for the Traveling Salesman Problem
Goal: find the shortest path from starting city, to all other cities, back to starting city
State will be represented as a list of cities ex: [3, 7, 2, 6, 4, 1, 9, 8, 5]

Findings:
*Report is at bottom of code including 5 attempts, starting at line 160
Parameters: Pop=100, f_thres=41, ngen=100, pmut=0.2
Best Path:  [8, 6, 9, 4, 1, 2, 5, 7, 3]
8  ->  6  =  7
6  ->  9  =  4
9  ->  4  =  1
4  ->  1  =  3
1  ->  2  =  2
2  ->  5  =  5
5  ->  7  =  6
7  ->  3  =  2
3  ->  8  =  5
Cost:  35
"""

import random as rnd
import bisect

def shuffled(iterable):
    """Randomly shuffle a copy of iterable."""
    items = list(iterable)
    rnd.shuffle(items)
    return items

def weighted_sampler(seq, weights):
    """Return a random-sample function that picks from seq weighted by weights."""
    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)
    return lambda: seq[bisect.bisect(totals, rnd.uniform(0, totals[-1]))]

def select(r, population, fitness_fn):
    fitnesses = map(fitness_fn, population)
    sampler = weighted_sampler(population, fitnesses)
    return [sampler() for i in range(r)]


cities = [1,2,3,4,5,6,7,8,9]

distance = [
    # 1  2  3   4   5   6  7   8   9
    [0,  2, 11, 3, 18, 14, 20, 12, 5],  # 1
    [2,  0, 13, 10, 5, 3,  8,  20, 17], # 2
    [11, 13, 0, 5, 19, 21, 2,  5,  8],  # 3
    [3,  10, 5, 0, 6,  4,  12, 15, 1],  # 4
    [18, 5, 19, 6,  0, 12, 6,  9,  7],  # 5
    [14, 3, 21, 4,  12, 0, 19, 7,  4],  # 6
    [20, 8, 2,  12, 6, 19, 0,  21, 13], # 7
    [12, 20, 5, 15, 9, 7,  21, 0,  6],  # 8
    [5, 17, 8,  1,  7, 4,  13, 6,  0]   # 9
]

# NEXT-PERMUTATION (INITIALIZE POPULATION)
def init_population(pop_number, gene_pool, state_length):
    """Initializes population for genetic algorithm
    pop_number  :  Number of individuals in population
    gene_pool   :  List of possible values for individuals
    state_length:  The length of each individual"""

    population = []
    for i in range(pop_number):
        population.append(shuffled(gene_pool))
    return population



# CROSSOVER FUNCTION (Reproduce)
def reproduce(x, y):
    """Reproduce 1 offspring from 2 parents"""
    start = rnd.randint(0, len(x) - 1)
    end = rnd.randint(start + 1, len(x))
    new_state = x[start:end]
    for city in y:
        if city not in new_state:
            new_state.append(city)
    return new_state


# MUTATE FUNCTION
def mutate(x, gene_pool, mutation_probability):
    """mutate problem states"""
    if rnd.uniform(0, 1) < mutation_probability:
        positions = rnd.sample(range(len(x)), 2)
        x[positions[0]], x[positions[1]] = x[positions[1]], x[positions[0]]
    return x


# FITNESS FUNCTION
def fitness_fn(x):
    cost = 0
    for i in range(len(x) - 1):
        from_index = x[i]-1
        to_index = x[i+1]-1
        cost += distance[from_index][to_index]

    first_city = x[0]-1
    last_city = x[-1]-1
    cost += distance[last_city][first_city]

    return cost


# FITNESS THRESHOLD
def fitness_threshold(fitness_fn, f_thres, population):
    if not f_thres:
        return None
    fittest_individual = max(population, key=fitness_fn)
    if fitness_fn(fittest_individual) <= f_thres:
        return fittest_individual
    return None


# GENETIC ALGORITHM
def genetic_algorithm(population, fitness_fn, gene_pool=[0, 1], f_thres=41, ngen=100, pmut=0.2):
    # Generational Reproduction
    for i in range(ngen):
        population = [mutate(reproduce(*select(2, population, fitness_fn)), gene_pool, pmut)
                      for i in range(len(population))]

        fittest_individual = fitness_threshold(fitness_fn, f_thres, population)

        if fittest_individual:
            return fittest_individual
    return min(population, key=fitness_fn)


# Display Solution
def genetic_search():
    pop = init_population(100, cities, 9)
    best = genetic_algorithm(pop, fitness_fn)
    best_score = fitness_fn(best)
    print("Best Path: ", best)

    for i in range(len(best) - 1):
        from_index = best[i]-1
        to_index = best[i+1]-1
        print(best[i], " -> ", best[i+1], " = " , distance[from_index][to_index])

    first_city = best[0]-1
    last_city = best[-1]-1

    print(last_city+1, " -> ", first_city+1, " = " , distance[last_city][first_city])
    print("Cost: ", best_score)

genetic_search()


# REPORT
# Attempt 1: Pop=100, f_thres=60, ngen=100, pmut=0.1 
# Best Path:  [9, 4, 1, 6, 2, 7, 3, 8, 5]
# 9  ->  4  =  1
# 4  ->  1  =  3
# 1  ->  6  =  14
# 6  ->  2  =  3
# 2  ->  7  =  8
# 7  ->  3  =  2
# 3  ->  8  =  5
# 8  ->  5  =  9
# 5  ->  9  =  7
# Cost:  52

# Attempt 2: Pop=100, f_thres=52, ngen=100, pmut=0.1 
# Best Path:  [4, 6, 5, 8, 3, 7, 2, 1, 9]
# 4  ->  6  =  4
# 6  ->  5  =  12
# 5  ->  8  =  9
# 8  ->  3  =  5
# 3  ->  7  =  2
# 7  ->  2  =  8
# 2  ->  1  =  2
# 1  ->  9  =  5
# 9  ->  4  =  1
# Cost:  48

# Attempt 3: Pop=100, f_thres=47, ngen=100, pmut=0.1 
# Best Path:  [5, 4, 3, 7, 6, 2, 1, 9, 8]
# 5  ->  4  =  6
# 4  ->  3  =  5
# 3  ->  7  =  2
# 7  ->  6  =  19
# 6  ->  2  =  3
# 2  ->  1  =  2
# 1  ->  9  =  5
# 9  ->  8  =  6
# 8  ->  5  =  9
# Cost:  57

# Attempt 4: Pop=10, f_thres=47, ngen=100, pmut=0.5
# Best Path:  [6, 8, 9, 5, 7, 3, 4, 1, 2]
# 6  ->  8  =  7
# 8  ->  9  =  6
# 9  ->  5  =  7
# 5  ->  7  =  6
# 7  ->  3  =  2
# 3  ->  4  =  5
# 4  ->  1  =  3
# 1  ->  2  =  2
# 2  ->  6  =  3
# Cost:  41

# BEST ATTEMPT
# Attempt 5: Pop=100, f_thres=41, ngen=100, pmut=0.2
# Best Path:  [8, 6, 9, 4, 1, 2, 5, 7, 3]
# 8  ->  6  =  7
# 6  ->  9  =  4
# 9  ->  4  =  1
# 4  ->  1  =  3
# 1  ->  2  =  2
# 2  ->  5  =  5
# 5  ->  7  =  6
# 7  ->  3  =  2
# 3  ->  8  =  5
# Cost:  35