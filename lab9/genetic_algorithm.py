import numpy as np
import lab9_lib
from genetic_op import select, mutate, crossover, two_point_crossover
from tqdm import tqdm
from matplotlib import pyplot as plt

# PROBLEM_INSTANCES = (1, 2, 5, 10)
PROBLEM_INSTANCE = 10
GENOME_SIZE = 1000

fitness = lab9_lib.make_problem(PROBLEM_INSTANCE)

num_generations = 500
# the population size
lbd = 50

population = np.random.randint(0, 2, (lbd, GENOME_SIZE,), np.byte)
fitnesses = np.zeros((lbd,), np.float64)
offspring = np.zeros((lbd, GENOME_SIZE,), np.byte)

# initialize population and best individual
best = np.zeros((GENOME_SIZE,), np.byte)
best_fitness = None
fitness_history = []
for gen in tqdm(range(num_generations)):
    for i in range(lbd):
        fitnesses[i] = fitness(population[i,:])
        if best_fitness == None or best_fitness < fitnesses[i]:
            best[:] = population[i,:]
            best_fitness = fitnesses[i]
    fitness_history.append(best_fitness)
    for i in range(lbd//2):
        pa = select(population, fitnesses)
        pb = select(population, fitnesses)
        ca, cb = offspring[i*2,:], offspring[i*2+1,:]
        # crossover and mutation happen in-place in the offspring array
        crossover(pa, pb, ca, cb)
        mutate(ca)
        mutate(cb)
    population, offspring = offspring, population
# one last round of comparing "best"
for i in range(lbd):
    if best_fitness < fitnesses[i]:
        best[:] = population[i,:]
        best_fitness = fitnesses[i]
        
print(f"Calls to the fitness function: {fitness.calls}")
print(f"Best fitness: {best_fitness}")
with open(f"best_individual_{PROBLEM_INSTANCE}.txt", 'w') as fp:
    fp.write("".join(str(i) for i in best))
with open(f"fitness_history_{PROBLEM_INSTANCE}.txt", "w") as fp:
    fp.write(" ".join(str(i) for i in fitness_history))
plt.plot(fitness_history)
plt.show()