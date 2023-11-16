import logging
from pprint import pformat
import random
from copy import deepcopy
import numpy as np
from nim import Nim, Nimply
from players import optimal, pure_random

logging.getLogger().setLevel(logging.INFO)

strategy = (optimal, pure_random)

TOTAL_ROUNDS = 50
optimal_wins = 0
for round in range(TOTAL_ROUNDS):
    nim = Nim(5)
    logging.info(f"init : {nim}")
    player = np.random.randint(0,2)
    while nim:
        ply = strategy[player](nim)
        logging.info(f"ply: player {player} plays {ply}")
        nim.nimming(ply)
        logging.info(f"status: {nim}")
        player = 1 - player
    logging.info(f"status: Player {player} won!")
    if player == 0: optimal_wins+=1

logging.info(f"Optimal wins {optimal_wins} out of {TOTAL_ROUNDS} rounds.")
