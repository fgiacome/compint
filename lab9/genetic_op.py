import numpy as np

# a tournament selection
def select(population, fitnesses, t=7):
    best = np.random.randint(0, population.shape[0])
    for i in range(1,t):
        next = np.random.randint(0, population.shape[0])
        if fitnesses[next] > fitnesses[best]:
            best = next
    return population[best,:]

# a bit-flip mutation
def mutate(a, p=0.001):
    for i in range(len(a)):
        if p > np.random.rand():
            a[i] = 1-a[i]

# a uniform crossover
def crossover(pa, pb, ca, cb, p=0.001):
    for i in range(len(pa)):
        if p > np.random.rand():
            ca[i], cb[i] = pb[i], pa[i]
        else:
            ca[i], cb[i] = pa[i], pb[i]
        
def two_point_crossover(pa, pb, ca, cb):
    ca[:] = pa[:]
    cb[:] = pb[:]
    c = np.random.randint(0, len(pa))
    d = np.random.randint(0, len(pa))
    if c > d: d, c = c, d
    if c != d:
        for i in range(c, d):
            ca[i], cb[i] = cb[i], ca[i]