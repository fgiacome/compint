import logging
import numpy as np
from players import (
    NimPlayer,
    RandomPlayer,
    RandomlyOptimal,
    StochasticRulesBased,
    TakeAllFromTallest,
    TakeAllButOneFromTallest,
    MakeTwinTowers,
    KeepTaller,
    ChangeTaller,
    TakeA1Line,
)
from game import match
import logging
from tqdm import tqdm
from matplotlib import pyplot as plt

# Number of "simple" players (aka strategies)
NUMBER_PLAYERS = 6
# Number of booleans in the reduced game state representation
STATES_DIM = 4
# Number of games per fitness evalutaion
NUM_GAMES = 100
# Number of stacks in the nim games
NIM_DIM = 5
mu = 5
lbd = 50
sig = 1
# Number of generations (iterations)
iters = 1_500 // lbd
adversary = RandomPlayer()  # RandomlyOptimal(0.3)
logging.getLogger().setLevel(logging.INFO)


def get_stochastic_rb_player(weight_matrix: np.ndarray) -> StochasticRulesBased:
    stochastic_rule_player = StochasticRulesBased(
        [
            TakeAllFromTallest(),
            TakeAllButOneFromTallest(),
            MakeTwinTowers(),
            KeepTaller(),
            ChangeTaller(),
            TakeA1Line(),
        ],
        weight_matrix,
    )
    return stochastic_rule_player


def evaluate_individuals(genotypes: np.ndarray, adversary: NimPlayer):
    fitness = np.ones(genotypes.shape[0])
    for i in range(genotypes.shape[0]):
        genotype = genotypes[i, :].reshape(NUMBER_PLAYERS, 2**STATES_DIM)
        player = get_stochastic_rb_player(genotype)
        fitness[i] = match(player, adversary, NUM_GAMES) / NUM_GAMES
    return fitness


population = np.random.randn(mu, NUMBER_PLAYERS * 2**STATES_DIM + 1)
population[:, -1] = np.random.random(mu)

best_fitness = None
history = list()
for step in tqdm(range(iters)):
    # offspring <- select λ random points from the population of μ
    offspring = population[np.random.randint(0, sig, size=(lbd,))]
    # mutate all σ (last column) and replace negative values with a small number
    offspring[:, -1] = np.random.normal(loc=offspring[:, -1], scale=0.2)
    offspring[offspring[:, -1] < 1e-5, -1] = 1e-5
    # mutate all v (all columns but the last), using the σ in the last column
    offspring[:, 0:-1] = np.random.normal(
        loc=offspring[:, 0:-1], scale=offspring[:, -1].reshape(-1, 1)
    )
    # add an extra column with the evaluation and sort
    fitness = evaluate_individuals(offspring[:, 0:-1], adversary)
    offspring = offspring[fitness.argsort()]
    # save best (just for the plot)
    if best_fitness is None or best_fitness < np.max(fitness):
        best_fitness = np.max(fitness)
    history.append((step, np.max(fitness)))
    # select the μ with max fitness and discard fitness
    population = np.copy(offspring[-mu:])

logging.getLogger().setLevel(logging.DEBUG)

fitness = evaluate_individuals(population[:, 0:-1], adversary)
logging.info(
    f"Best solution: {fitness.max()} (with σ={population[fitness.argmax(), -1]:0.3g})"
)

logging.getLogger().setLevel(logging.INFO)

history = np.array(history)
plt.figure(figsize=(14, 4))
plt.plot(history[:, 0], history[:, 1], marker=".")
plt.show()

with open("history.txt", "a") as fp:
    fp.write("#" * 80 + "\n")
    fp.write(f"Experiment log.\n")
    fp.write(f"Number players: {NUMBER_PLAYERS}\n")
    fp.write(f"State dim: {STATES_DIM}\n")
    fp.write(f"Num games per match: {NUM_GAMES}\n")
    fp.write(f"Nim dim: {NIM_DIM}\n")
    fp.write(f"mu: {mu}, lbd: {lbd}, sig: {sig}\n")
    fp.write(f"Adversary: {adversary}\n")
    fp.write(f"Iterations: {iters}\n")
    fp.write(f"History:\n{history}\n")
    fp.write(
        f"Best individual:\n{population[fitness.argmax(), :-1].reshape(NUMBER_PLAYERS, 2**STATES_DIM)}\n"
    )
    fp.write(f"Best sig: {population[fitness.argmax(), -1]}\n")
    fp.write(f"With fitness: {fitness.max()}\n")
