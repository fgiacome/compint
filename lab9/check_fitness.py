import lab9_lib
import numpy as np

PROBLEM_INSTANCE = 5
fitness = lab9_lib.make_problem(PROBLEM_INSTANCE)
with open(f"best_individual_{PROBLEM_INSTANCE}.txt", 'r') as fp:
    raw = fp.read()
individual = np.array([int(i) for i in raw], np.byte)
print(f"Instance {PROBLEM_INSTANCE}. Fitness: {fitness(individual)}")