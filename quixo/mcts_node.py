import numpy as np


class MctsNode:
    def __init__(self, board: np.ndarray, player_id):
        self.board: tuple = tuple(board.ravel())
        self.turn: int = player_id
        # parent in the *current* simulation
        self.parent: "MctsNode" = None
        # action that must be taken to reach this node from the *current* parent
        self.action = None
        # whether the nodes has been visited (ie a simulation has started from it),
        # this includes expanding the children
        self.visited: bool = False
        self.children: list = []
        # minimax evaluation: depth and value
        self.minimax_evaluated: int = -1
        self.minimax_heuristic: float = 0.0
        self.minimax_value: int = -1
        # simulations that have started out from this node
        self.simulations: int = 0
        # winning simulations that have started out from this node
        self.wins: int = 0

    def __hash__(self):
        return hash((self.state, self.turn))

    def __eq__(self, other: "MctsNode"):
        return self.board == other.board and self.turn == other.turn

    def get_board(self):
        return np.array(self.board, dtype=int).reshape(5, 5)

    def get_turn(self):
        return self.turn

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def get_action(self):
        return self.action

    def set_action(self, action):
        self.action = action

    def set_visited(self, children: list["MctsNode"]):
        self.children = children
        self.visited = True

    def set_minimax_value(self, minimax_value, depth):
        self.minimax_evaluated = depth
        self.minimax_value = minimax_value

    def get_visited(self):
        return self.visited

    def get_children(self) -> list["MctsNode"]:
        return self.children.copy()

    def get_number_of_children(self):
        return len(self.children)

    def get_minimax_evaluated(self):
        return self.minimax_evaluated

    def get_minimax_value(self):
        return self.minimax_value

    def get_minimax_heuristic(self):
        return self.minimax_heuristic

    def set_minimax_heuristic(self, value):
        self.minimax_heuristic = value

    def get_simulations(self):
        return self.simulations

    def add_simulations(self, simulations=1):
        self.simulations += simulations

    def get_wins(self):
        return self.wins

    def add_wins(self, wins=1):
        self.wins += wins
