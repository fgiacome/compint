from game import Game, Player, Move
import utils
from mcts_node import MctsNode
import numpy as np
import random


class MctsPlayer(Player):
    def __init__(self, print_board=False):
        self.print_board = print_board
        self.node_table: dict[tuple[tuple[int], int], int] = {}

    @staticmethod
    def minimax(node: MctsNode, node_table, depth=2):
        if node.get_minimax_evaluated() >= depth or node.get_minimax_value() != -1:
            return node.get_minimax_value(), node.get_minimax_heuristic()

        board, turn = node.get_board(), node.get_turn()
        winner = utils.check_win(board, turn)
        if winner in {1, 0}:
            heuristic = -1.0 if winner == 1 else 1.0
        else:
            heuristic = utils.minimax_heuristic(board, turn)
        if winner == -1 and utils.win_possible(board, turn, depth):
            winner = (turn + 1) % 2
            heuristic = -1.0 if turn == 0 else 1.0
            child_boards, _ = utils.get_possible_actions(board, turn)
            for child_board in child_boards:
                child_board_ = tuple(child_board.ravel())
                if not (child_board_, (turn + 1) % 2) in node_table.keys():
                    child = MctsNode(child_board, (turn + 1) % 2)
                    node_table[(child_board_, (turn + 1) % 2)] = child
                else:
                    child = node_table[(child_board_, (turn + 1) % 2)]
                value, h_value = MctsPlayer.minimax(child, node_table, depth=depth - 1)
                if value == turn:
                    winner = turn
                    heuristic = 1.0 if turn == 0 else -1.0
                    break
                elif value == -1:
                    if turn == 0 and h_value > heuristic:
                        heuristic = h_value
                    elif turn == 1 and h_value < heuristic:
                        heuristic = h_value
                    winner = -1

        node.set_minimax_value(winner, depth)
        node.set_minimax_heuristic(heuristic)
        return winner, heuristic

    @staticmethod
    def rollout(board, turn):
        """board will be changed by this function"""
        game = Game()
        player1, player2 = RandomPlayer(), RandomPlayer()
        game.current_player_idx = turn
        game._board = board
        return game.play(player1, player2)

    @staticmethod
    def backpropagation(node: MctsNode, win_id: int):
        node.add_simulations()
        parent = node.get_parent()
        while parent is not None:
            if parent.get_turn() == win_id:
                node.add_wins()
            if parent.get_turn() == (win_id + 1) % 2:
                node.add_wins(wins=-1)
            node = parent
            node.add_simulations()
            parent = node.get_parent()
        if node.get_turn() == win_id:
            node.add_wins(wins=-1)
        if node.get_turn() == (win_id + 1) % 2:
            node.add_wins()

    @staticmethod
    def simulation(
        root: MctsNode,
        node_table: dict[np.ndarray : MctsNode],
    ):
        """IMPORTANT: this function assumes root node has been visited.
        This is because the "actions" of the children must be initialized and
        I want this to be dealt with outside this function, which does not
        initialize actions but only execute simulations."""

        end = False
        winner = -1
        current_node = root
        player = root.get_turn()
        traversed = [root]

        while not end:
            # if node has not been visited, visit it and run the simulation
            if not current_node.get_visited():
                board, turn = current_node.get_board(), current_node.get_turn()
                child_boards, _ = utils.get_possible_actions(board, turn)
                children = []
                for child_board in child_boards:
                    child_board_ = tuple(child_board.ravel())
                    if (child_board_, (turn + 1) % 2) not in node_table.keys():
                        child = MctsNode(child_board, (turn + 1) % 2)
                        node_table[(child_board_, (turn + 1) % 2)] = child
                    else:
                        child = node_table[(child_board_, (turn + 1) % 2)]
                    if child not in children:
                        children.append(child)
                current_node.set_visited(children)
                end = True
            # else, check if we can end the traversal here (ie the node is terminal)
            elif (
                current_node.get_number_of_children() == 0
                or current_node.get_minimax_value() != -1
            ):
                end = True
            # else, the node has been visited and it is time to ROLLOUT!
            else:
                children = current_node.get_children()
                # inform MCTS: discard children that surely lead to losing
                good_children = [
                    child
                    for child in children
                    if child.get_minimax_value() != (player + 1) % 2
                ]
                # ...if there any good ones at all
                good_children = children if len(good_children) == 0 else good_children
                children = good_children
                ucb = True
                for child in children:
                    if child.get_simulations() == 0:
                        ucb = False
                        end = True
                        break
                if ucb:
                    N = sum(child.get_simulations() for child in children)
                    alpha = 0.8
                    h_coeff = 1 if current_node.get_turn() == 0 else -1
                    key1 = lambda c: alpha * h_coeff * c.get_minimax_heuristic() + (
                        1 - alpha
                    ) * (c.get_wins() / c.get_simulations())
                    key2 = lambda c: np.sqrt(2 * np.log(N) / c.get_simulations())
                    child = max(children, key=lambda c: key1(c) * key2(c))
                if child in traversed:
                    # loop detected
                    end = True
                else:
                    child.set_parent(current_node)
                    current_node = child
                    traversed.append(current_node)

        winner = current_node.get_minimax_value()
        if winner == -1:
            winner = MctsPlayer.rollout(current_node.get_board(), current_node.turn)

        MctsPlayer.backpropagation(current_node, winner)

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        board = game.get_board()
        board_ = tuple(board.ravel())
        turn = game.get_current_player()
        if (board_, turn) not in self.node_table.keys():
            root = MctsNode(board, turn)
        else:
            root = self.node_table[(board_, turn)]

        # visit root
        # this should be done even if root had been visited before,
        # so that I can initalize the actions
        board, turn = root.get_board(), root.get_turn()
        child_boards, actions = utils.get_possible_actions(board, turn)
        children = []
        for i, child_board in enumerate(child_boards):
            child_board_ = tuple(child_board.ravel())
            if (child_board_, (turn + 1) % 2) not in self.node_table.keys():
                child = MctsNode(child_board, (turn + 1) % 2)
                self.node_table[(child_board_, (turn + 1) % 2)] = child
            else:
                child = self.node_table[(child_board_, (turn + 1) % 2)]
            if child not in children:
                children.append(child)
            child.set_action(actions[i])
        root.set_visited(children)
        MctsPlayer.minimax(root, self.node_table, depth=4)
        root.set_parent(None)

        cell, side = None, None
        for child in children:
            if child.get_minimax_value() == turn:
                cell, side = child.get_action()
                break

        if cell is None:
            for _ in range(300):
                MctsPlayer.simulation(root, self.node_table)

            cell, side = max(
                root.get_children(), key=lambda c: c.get_simulations()
            ).get_action()

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
        _, actions = utils.get_possible_actions(board, game.get_current_player())
        cell, side = random.choice(actions)
        if self.print_board:
            print(board)
            print(f"Random Player going for {cell}, {side}.")
        return cell, side
