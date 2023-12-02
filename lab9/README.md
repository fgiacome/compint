# Lab 9
The goal of this laboratory is to solve an instance of black-box optimization.
The file `lab9_lib.py` defines a family of objective functions that should be maximized without making any assumptions on them.

I solved this problem via a textbook genetic algorithm.
The file `genetic_algorithm.py` can be run to make a round of optimization.
The constant `PROBLEM_INSTANCE` can be modified to solve a different problem instance.
The assignment requires working with 1000-loci genomes.
Furthermore, the genomes are arrays $\in \{0,1\}^{1000}$.

The constants `num_generations` and `lbd` control respectively the number of generations and the population size (lbd stands for lambda).
The number of calls to the fitness function equals the product of these numbers:
in all my experiments, it is 500 (`num_generations`) * 50 (`lbd`) = 25'000.

## Genetic operators and selection
The genetic algorithm requires two genetic operators (mutation and crossover) and one selection routine.

The mutation operator is implemented as a bit-flip mutation, where each bit in the genome is mutated with probability 1/1000 (the inverse of the length of the genome).
The crossover operator is implemented as uniform crossover, where the loci of the parents are swapped with probability 1/1000 when producing the children.
The selection routine is a tournament of 7 individuals picked from the population uniformly with replacement.

A two-point crossover was also implemented and tested, but the results did not appear significantly different (although a low number of experiments was performed with this strategy, and only with problem instance 5).
Consequently, only the uniform crossover was used for all other problem instances.

## A word on technical aspects
To make the algorithm more efficient, I made sure to work consistently with `numpy` arrays, rather than mixing python-native types (such as lists) and arrays.

Furthermore, I tried to minimize the number of copies and memory allocations by avoiding the creation of new objects and rather make changes in-place, when possible.
The arrays that contain the population and offspring are defined at the beginning of the training loop and no further arrays are created later, but only array views that are used to edit the arrays in-place.
Finally, rather than copying the offspring array into the population array, their names are simply swapped.

## Results
The plots of the fitness function of the overall best individual, over the number of generations, can be seen in the `fitness_*.svg` files, where the problem instance is in place of the asterisk.
With similar nomenclature, a file with the genome of the best individual and one with the fitness history (a textual version of the graph) can also be found for each problem instance.

By running multiple experiments with the same problem instance, it is possible to observe that the final result (in terms of fitness) differes each time in the order of $10^{-2}$, with some exceptions of very low or very high fitness.
In the latter cases, the fitness is usually unable to improve over several generations and remains flat.
In general, however, the fitness improves frequently.
The graph is smoother in the case of problem instances 1 and 2, indicating an overall smoother fitness landscape.

## References
The file `lab9_lib.py` containes code extracted from an interaticve python notebook in the repository of Professor Squillero, and is an integral part of the problem definition (assignment).
I read it quickly only to make sure that I could correctly execute it on my computer, without making any changes.

The core code of the genetic algorithm, the genetic operators and the selection routine is adapted from the pseudocode found in Sean Luke, Essentials of Metaheuristics, II ed., 2016.