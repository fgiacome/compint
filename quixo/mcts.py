from game import Game, Player, Move
import numpy as np
from copy import deepcopy
import random
import torch
import multiprocessing


class MctsNode:
    def __init__(self, board, player_id):
        self.state = tuple(board.ravel())
        self.turn = player_id
        self.parent = None

    def __hash__(self):
        return hash((self.state, self.turn))

    def __eq__(self, other):
        return self.state == other.state and self.turn == other.turn

    def get_board(self):
        return np.array(self.state, dtype=int).reshape(5, 5)


class MctsPlayer(Player):
    def __init__(self, print_board=False, policy=None):
        self.print_board = print_board
        self.policy = policy

    @staticmethod
    def get_cell_sides(cell):
        cell_sides = set()
        if cell[0] == 0:
            cell_sides.add(2)
        if cell[0] == 4:
            cell_sides.add(3)
        if cell[1] == 0:
            cell_sides.add(0)
        if cell[1] == 4:
            cell_sides.add(1)
        return cell_sides

    @staticmethod
    def get_new_board(board, my_id, action):
        """Assumes the action is valid"""
        board = deepcopy(board)
        if action[1] == Move.TOP:
            for i in range(action[0][1], 0, -1):
                board[i, action[0][0]] = board[i - 1, action[0][0]]
            board[0, action[0][0]] = my_id
        if action[1] == Move.LEFT:
            for i in range(action[0][0], 0, -1):
                board[action[0][1], i] = board[action[0][1], i - 1]
            board[action[0][1], 0] = my_id
        if action[1] == Move.RIGHT:
            for i in range(action[0][0], 4, 1):
                board[action[0][1], i] = board[action[0][1], i + 1]
            board[action[0][1], 4] = my_id
        if action[1] == Move.BOTTOM:
            for i in range(action[0][1], 4, 1):
                board[i, action[0][0]] = board[i + 1, action[0][0]]
            board[4, action[0][0]] = my_id
        return board

    @staticmethod
    def check_win(board):
        """0: win 0, 1: win 1"""
        winner = -1
        for i in range(5):
            if board[i,0] != -1 and all(board[i, :] == board[i, 0]):
                winner = board[i, 0]
            if board[0,i] != -1 and all(board[:, i] == board[0, i]):
                winner = board[0, i]
        if board[0,0] != -1 and all(board[i, i] == board[0, 0] for i in range(5)):
            winner = board[0, 0]
        if board[0,4] != -1 and all(board[i, 4 - i] == board[0, 4] for i in range(5)):
            winner = board[0, 4]
        return winner

    @staticmethod
    def obtain_possible_actions(board, my_id):
        actions = []
        boards = []
        for i in range(5):
            for j in range(5):
                if i in {0, 4} or j in {0, 4}:
                    if board[j, i] in {-1, my_id}:
                        for k in {0, 1, 2, 3} - MctsPlayer.get_cell_sides((i, j)):
                            actions.append(((i, j), Move(k)))
                            boards.append(
                                MctsPlayer.get_new_board(
                                    board, my_id, ((i, j), Move(k))
                                )
                            )
        return boards, actions

    @staticmethod
    def rollout(board, turn):
        """board will be changed by this function"""
        game = Game()
        player1, player2 = RandomPlayer(), RandomPlayer()
        game.current_player_idx = turn
        game._board = board
        return game.play(player1, player2)

    @staticmethod
    def backpropagation(node, win_id, node_stats):
        while node.parent is not None:
            if node.parent.turn == win_id:
                node_stats[node][0] += 1
            if node.parent.turn == (win_id+1)%2:
                node_stats[node][0] -= 1
            node_stats[node][1] += 1
            node = node.parent
        node_stats[node][1] += 1

    @staticmethod
    def map_board(x: int):
        if x // 4 == 0:
            pos = (x % 4, 0)
        elif x // 4 == 1:
            pos = (4, x % 4)
        elif x // 4 == 2:
            pos = (4 - x % 4, 4)
        elif x // 4 == 3:
            pos = (0, 4 - x % 4)
        return pos

    @staticmethod
    def inverse_map_board(t: tuple[int, int]):
        i = None
        for i in range(16):
            if MctsPlayer.map_board(i) == t:
                break
        return i

    @staticmethod
    def inverse_map_move(t: Move):
        s = None
        if t == Move.TOP:
            s = 0
        elif t == Move.BOTTOM:
            s = 1
        elif t == Move.LEFT:
            s = 2
        elif t == Move.RIGHT:
            s = 3
        return s

    @staticmethod
    def get_allowed_action_probs(
        y_side: torch.Tensor, y_cell: torch.Tensor, board, my_id
    ):
        y_side = y_side.reshape(4, 1)
        y_cell = y_cell.reshape(1, 16)
        actions = y_side @ y_cell
        actions = actions + 1e-8
        for i in range(16):
            cell = MctsPlayer.map_board(i)
            if board[cell[1], cell[0]] not in {-1, my_id}:
                actions[:, i] = 0
                continue
            for side in MctsPlayer.get_cell_sides(cell):
                actions[side, i] = 0
        actions = actions / actions.sum()
        return actions

    @staticmethod
    def get_child_probs(policy, board, children, actions, my_id):
        y_side, y_cell = policy.forward(
            torch.tensor(board.reshape((25,)), dtype=torch.float32)
        )
        action_probs = MctsPlayer.get_allowed_action_probs(y_side, y_cell, board, my_id)
        child_probs = {}
        for k, action in enumerate(actions):
            i = MctsPlayer.inverse_map_board(action[0])
            j = MctsPlayer.inverse_map_move(action[1])
            child_probs[children[k]] = action_probs[j, i]
        return child_probs

    @staticmethod
    def simulation(
        root: MctsNode,
        node_stats: None | dict[MctsNode : list[int, int]] = None,
        policy=None,
    ):
        end = False
        winner = None
        current_node = root
        if node_stats is None:
            node_stats[root] = [0, 0]

        while not end:
            children_boards, actions = MctsPlayer.obtain_possible_actions(
                current_node.get_board(), current_node.turn
            )
            children = []
            for child_board in children_boards:
                children.append(MctsNode(child_board, (current_node.turn + 1) % 2))

            if len(children) == 0:
                # terminal node
                winner = MctsNode.check_win(current_node.get_board())
                end = True

            elif node_stats[current_node][1] == 0:
                # leaf node
                # initilize all children (if not already present)
                for child in children:
                    if child not in node_stats.keys():
                        node_stats[child] = [0, 0]
                # quit iteration and start rollout from here
                end = True

            elif (
                not all(node_stats[child][1] > 0 for child in children)
                and policy is None
            ):
                # pick a random child with 0 visits
                lonely_children = [
                    child for child in children if node_stats[child][1] == 0
                ]
                child = random.choice(lonely_children)
                child.parent = current_node
                current_node = child

            else:
                # All children have statistics. Great! Use UCB1.
                N = sum(node_stats[child][1] for child in children)
                if policy is None:
                    child = max(
                        children,
                        key=lambda child: node_stats[child][0] / node_stats[child][1]
                        + np.sqrt(2 * np.log(N) / node_stats[child][1]),
                    )
                else:
                    policy_child_probs = MctsPlayer.get_child_probs(
                        policy,
                        current_node.get_board(),
                        children,
                        actions,
                        current_node.turn,
                    )
                    child = max(
                        children,
                        key=lambda child: (
                            node_stats[child][0] + policy_child_probs[child]
                        )
                        / (node_stats[child][1] + 1)
                        + np.sqrt(
                            2 * np.log(N + len(children)) / (node_stats[child][1] + 1)
                        ),
                    )
                child.parent = current_node
                current_node = child

        if winner is None:
            winner = MctsPlayer.rollout(current_node.get_board(), current_node.turn)

        MctsPlayer.backpropagation(current_node, winner, node_stats)
        return node_stats

    @staticmethod
    def many_simulations(num, root, node_stats, policy=None):
        """It runs num simulations. Just a shell for the process pool"""
        for _ in range(num):
            MctsPlayer.simulation(root, node_stats, policy=policy)
        return node_stats

    @staticmethod
    def merge_node_stats(node_stats, ns):
        for s in ns.keys():
            if s not in node_stats.keys():
                node_stats[s] = [0, 0]
            node_stats[s][0] += ns[s][0]
            node_stats[s][1] += ns[s][1]
        return node_stats

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        board = game.get_board()
        state = MctsNode(board, game.get_current_player())
        node_stats = {state: [0, 0]}

        pool = multiprocessing.Pool()
        works = []
        for _ in range(8):
            works.append(
                pool.apply_async(
                    MctsPlayer.many_simulations,
                    (100, state, deepcopy(node_stats), self.policy),
                )
            )
        for i in range(32):
            ns = works[i % 8].get()
            MctsPlayer.merge_node_stats(node_stats, ns)
            if i < 24:
                works[i % 8] = pool.apply_async(
                    MctsPlayer.many_simulations,
                    (10, state, deepcopy(node_stats), self.policy),
                )

        children_boards, actions = MctsPlayer.obtain_possible_actions(
            board, game.get_current_player()
        )
        children = [
            MctsNode(child_board, (game.get_current_player() + 1) % 2)
            for child_board in children_boards
        ]
        cell, side = actions[
            max(
                (i for i in range(len(children))),
                key=lambda i: node_stats[children[i]][1],
            )
        ]
        if self.print_board:
            print(board)
            print(f"Mcts Player going for {cell}, {side}.")
        return cell, side


class RandomPlayer(Player):
    def __init__(self, print_board=None) -> None:
        super().__init__()
        self.print_board = print_board

    def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
        board = game.get_board()
        _, actions = MctsPlayer.obtain_possible_actions(
            board, game.get_current_player()
        )
        cell, side = random.choice(actions)
        if self.print_board:
            print(board)
            print(f"Random Player going for {cell}, {side}.")
        return cell, side
