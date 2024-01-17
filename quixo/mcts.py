from game import Game, Player, Move
import numpy as np
from copy import deepcopy
import random

class MctsNode():
    def __init__(self, board, player_id):
        self.state = tuple(board.ravel())
        self.turn = player_id
        self.parent = None
    
    def __hash__(self):
        return hash((self.state, self.turn))
    
    def __eq__(self, other):
        return (self.state == other.state and self.turn == other.turn)
    
    def get_board(self):
        return np.array(self.state, dtype=int).reshape(5,5)

class MctsPlayer(Player):
    def __init__(self, print_board=False):
        self.print_board = print_board

    @staticmethod
    def get_cell_sides(cell):
        cell_sides = set()
        if cell[0] == 0: cell_sides.add(2)
        if cell[0] == 4: cell_sides.add(3)
        if cell[1] == 0: cell_sides.add(0)
        if cell[1] == 4: cell_sides.add(1)
        return cell_sides

    @staticmethod
    def get_new_board(board, my_id, action):
        """Assumes the action is valid"""
        board = deepcopy(board)
        if action[1] == Move.TOP:
            for i in range(action[0][1],0,-1):
                board[i,action[0][0]] = board[i-1,action[0][0]]
                board[0,action[0][0]] = my_id
        if action[1] == Move.LEFT:
            for i in range(action[0][0],0,-1):
                board[action[0][1],i] = board[action[0][1],i-1]
                board[action[0][1],0] = my_id
        if action[1] == Move.RIGHT:
            for i in range(action[0][0],4,1):
                board[action[0][1],i] = board[action[0][1],i+1]
                board[action[0][1],4] = my_id
        if action[1] == Move.BOTTOM:
            for i in range(action[0][1],4,1):
                board[i,action[0][0]] = board[i+1,action[0][0]]
                board[4,action[0][0]] = my_id
        return board

    @staticmethod
    def check_win(board):
        """0: win 0, 1: win 1"""
        winner = -1
        for i in range(5):
            if all(board[i,:]==board[i,0]):
                winner = board[i,0]
            if all(board[:,i]==board[0,i]):
                winner = board[0,i]
        if all(board[i,i]==board[0,0] for i in range(5)):
            winner = board[0,0]
        if all(board[i,4-i]==board[0,0] for i in range(5)):
            winner = board[0,0]
        return winner
        
    @staticmethod
    def obtain_possible_actions(board, my_id):
        actions = []
        boards = []
        for i in range(5):
            for j in range(5):
                if i in {0,4} or j in {0,4}:
                    if board[j,i] in {-1, my_id}:
                        for k in {0,1,2,3}-MctsPlayer.get_cell_sides((i,j)):
                            actions.append(((i,j), Move(k)))
                            boards.append(MctsPlayer.get_new_board(board, my_id, ((i,j), Move(k))))
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
            node_stats[node][1] += 1
            node = node.parent
        node_stats[node][1] += 1


    @staticmethod
    def simulation(root: MctsNode, node_stats: None | dict[MctsNode: list[int, int]] = None):
        end = False
        winner = None
        current_node = root
        if node_stats is None: node_stats[root] = [0,0]
        
        while not end:
            children_boards, _ = MctsPlayer.obtain_possible_actions(current_node.get_board(), current_node.turn)
            children = []
            for child_board in children_boards:
                children.append(MctsNode(child_board, (current_node.turn+1)%2))

            if len(children) == 0:
                # terminal node
                winner = MctsNode.check_win(current_node.get_board())
                end = True

            elif node_stats[current_node][1] == 0:
                # leaf node
                # initilize all children (if not already present)
                for child in children:
                    if child not in node_stats.keys():
                        node_stats[child] = [0,0]
                # quit iteration and start rollout from here
                end = True

            elif not all(node_stats[child][1] > 0 for child in children):
                # pick a random child with 0 visits
                lonely_children = [child for child in children if node_stats[child][1] == 0]
                child = random.choice(lonely_children)
                child.parent = current_node
                current_node = child
            
            else:
                # All children have statistics. Great! Use UCB1.
                N = sum(node_stats[child][1] for child in children)
                child = max(children, key = lambda child: node_stats[child][0]/node_stats[child][1] + np.sqrt(2*np.log(N)/node_stats[child][1]))
                child.parent = current_node
                current_node = child
            
        if winner is None:
            winner = MctsPlayer.rollout(current_node.get_board(), current_node.turn)
        
        MctsPlayer.backpropagation(current_node, winner, node_stats)

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        board = game.get_board()
        state = MctsNode(board, game.get_current_player())
        node_stats = {state: [0,0]}

        for _ in range(300):
            MctsPlayer.simulation(state, node_stats)
        
        children_boards, actions = MctsPlayer.obtain_possible_actions(board, game.get_current_player())
        children = [MctsNode(child_board, (game.get_current_player()+1)%2) for child_board in children_boards]
        cell, side = actions[max((i for i in range(len(children))), key = lambda i: node_stats[children[i]][1])]
        if self.print_board:
            print(board)
            print(f"Mcts Player going for {cell}, {side}.")
        return cell, side

class RandomPlayer(Player):
    def __init__(self, print_board = None) -> None:
        super().__init__()
        self.print_board = print_board

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        board = game.get_board()
        _, actions = MctsPlayer.obtain_possible_actions(board, game.get_current_player())
        cell, side = random.choice(actions)
        if self.print_board:
            print(board)
            print(f"Random Player going for {cell}, {side}.")
        return cell, side