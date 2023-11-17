import logging
import random
import numpy as np
from nim import Nim, Nimply
from scipy.special import softmax


class NimPlayer:
    def __call__(self, state: Nim) -> Nimply:
        pass

    def __str__(self):
        return str(type(self))


class OptimalPlayer(NimPlayer):
    def nim_sum(self, a):
        tmp = np.array([tuple(int(x) for x in f"{i:032b}") for i in a])
        xor = tmp.sum(axis=0) % 2
        return int("".join(str(_) for _ in xor), base=2)

    def __init__(self):
        self.random_player = RandomPlayer()

    def __call__(self, state: Nim) -> Nimply:
        desired_ns = int(np.sum([c - 1 > 0 for c in state.rows]) <= 1)
        ns = self.nim_sum(state.rows)
        for r, c in enumerate(state.rows):
            if not c > 0:
                continue
            residual = self.nim_sum((ns, c))
            if desired_ns == 0:
                if residual < c:
                    return Nimply(r, c - residual)
            else:
                if residual == 0 and c > 1:
                    return Nimply(r, c - 1)
                elif residual == 1:
                    return Nimply(r, c)
        return self.random_player(state)


class RandomPlayer(NimPlayer):
    def __call__(self, state: Nim) -> Nimply:
        """A completely random move"""
        row = random.choice([r for r, c in enumerate(state.rows) if c > 0])
        num_objects = random.randint(1, min(state.rows[row], state.k))
        return Nimply(row, num_objects)


class RandomlyOptimal(NimPlayer):
    def __init__(self, p: int):
        """Makes an optimal move with probabilty 1-p, otherwise a random move."""
        assert p >= 0 and p <= 1
        self.p = p
        self.expert = OptimalPlayer()
        self.random = RandomPlayer()
        self.players = [self.expert, self.random]

    def __call__(self, state: Nim) -> Nimply:
        player = self.players[int(np.random.random() > 1 - self.p)]
        return player(state)

    def __str__(self):
        return str(type(self)) + f"({self.p})"


class StochasticRulesBased(NimPlayer):
    def get_weights(self):
        return self.weights

    def state_project(self, state: Nim) -> tuple[bool]:
        """This function projects a nim state to a lower-dimensional space."""
        rows = sorted(state.rows)
        m = sum((1 for i in state.rows if i > 0))
        n = rows[0]
        n_p = rows[1]
        return (m % 2 == 0, n == n_p, n_p > 1, n_p % 2 == 0)

    def __init__(self, players: list[NimPlayer], weights: np.ndarray):
        """Pass a list of determinisic players (rules) and a list of
        weights (probabilities), one for each player."""
        self._state_dims = 4
        self.players = players
        assert weights.shape == (len(players), 2**self._state_dims)
        self.weights = weights
        self.probs = softmax(self.weights, axis=0)
        self._decoder = np.power(
            np.ones(self._state_dims, dtype=int) * 2,
            np.arange(self._state_dims, dtype=int),
            dtype=int,
        )

    def __call__(self, state: Nim) -> Nimply:
        state_projection = np.array(self.state_project(state), dtype=int)
        weights_col = state_projection @ self._decoder
        logging.debug(f"StochasticRulesBased: Selected col: {weights_col}.")
        player = self.players[
            np.random.choice(len(self.players), p=self.probs[:, weights_col])
        ]
        logging.debug(f"StochasticRulesBased: chosen player {type(player)}")
        return player(state)


class NimPlayerWithBackup(NimPlayer):
    def backup(self, state: Nim) -> Nimply:
        random = RandomPlayer()
        return random(state)


class TakeAllFromTallest(NimPlayer):
    def __call__(self, state: Nim) -> Nimply:
        tallest_row = np.argmax(state.rows)
        logging.debug(f"Player: TakeAllFromTallest: tallest_row is {tallest_row}.")
        return Nimply(tallest_row, min(state.rows[tallest_row], state.k))


class TakeAllButOneFromTallest(NimPlayerWithBackup):
    def __call__(self, state: Nim) -> Nimply:
        tallest_row = np.argmax(state.rows)
        if not state.rows[tallest_row] > 1:
            return self.backup(state)
        logging.debug(
            f"Player: TakeAllButOneFromTallest: tallest_row is {tallest_row}."
        )
        return Nimply(tallest_row, min(state.rows[tallest_row] - 1, state.k))


def get_two_tallest_with_indeces(state: Nim) -> tuple[int]:
    rows = list(state.rows)
    tallest_row = np.argmax(rows)
    rows[tallest_row] = 0
    second_tallest_row = np.argmax(rows)
    rows = state.rows
    return (
        tallest_row,
        second_tallest_row,
        rows[tallest_row],
        rows[second_tallest_row],
    )


class MakeTwinTowers(NimPlayerWithBackup):
    def __call__(self, state: Nim) -> Nimply:
        tallest_row, _, n, n_p = get_two_tallest_with_indeces(state)
        if n_p < n:
            return Nimply(tallest_row, n - n_p)
        else:
            logging.debug(f"Player: MakeTwinTowers: calling backup")
            return self.backup(state)


class ChangeTaller(NimPlayerWithBackup):
    def __call__(self, state: Nim):
        tallest_row, _, n, n_p = get_two_tallest_with_indeces(state)
        if not n_p >= 1:
            logging.debug(f"Player: ChangeTaller: calling backup")
            return self.backup(state)
        return Nimply(tallest_row, n - n_p + 1)


class KeepTaller(NimPlayerWithBackup):
    def __call__(self, state: Nim):
        tallest_row, _, n, n_p = get_two_tallest_with_indeces(state)
        if n_p + 1 < n:
            return Nimply(tallest_row, n - n_p - 1)
        else:
            logging.debug(f"Player: KeepTaller: calling backup")
            return self.backup(state)


class TakeA1Line(NimPlayerWithBackup):
    def __call__(self, state: Nim):
        for i, n in enumerate(state.rows):
            if n == 1:
                break
        if state.rows[i] == 1:
            return Nimply(i, 1)
        else:
            logging.debug(f"Player: TakeA1Line: calling backup")
            return self.backup(state)
