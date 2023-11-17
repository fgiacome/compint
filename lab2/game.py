import logging
import numpy as np
from nim import Nim
from players import RandomPlayer, NimPlayer, RandomlyOptimal, OptimalPlayer


def match(player_1: NimPlayer, player_2: NimPlayer, num_games: int, nim_size=5) -> int:
    """Returns the wins of player_1 out of num_games games"""
    strategy = [player_1, player_2]
    optimal_wins = 0
    for round in range(num_games):
        logging.debug(f"Starting game {round}.")
        nim = Nim(nim_size)
        logging.debug(f"init : {nim}")
        player = np.random.randint(0, 2)
        while nim:
            ply = strategy[player](nim)
            logging.debug(f"ply: player {player} plays {ply}")
            nim.nimming(ply)
            logging.debug(f"status: {nim}")
            player = 1 - player
        logging.debug(f"status: Player {player} won!")
        if player == 0:
            optimal_wins += 1
    return optimal_wins


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    TOTAL_ROUNDS = 500
    NIM_SIZE = 5
    player_1, player_2 = RandomlyOptimal(0.1), RandomPlayer()
    optimal_wins = match(player_1, player_2, TOTAL_ROUNDS, 5)

    logging.info(f"Optimal wins {optimal_wins} out of {TOTAL_ROUNDS} rounds.")
