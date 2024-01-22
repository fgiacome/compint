from copy import deepcopy
from game import Move
import numpy as np


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


def how_many_in_line(board, player):
    board = board == player
    t = 0
    for i in range(5):
        s = sum(board[:, i])
        if s > t:
            t = s
        s = sum(board[i, :])
        if s > t:
            t = s
    s = sum(board[i, i] for i in range(5))
    if s > t:
        t = s
    s = sum(board[i, 4 - i] for i in range(5))
    if s > t:
        t = s
    return t


def get_possible_actions(
    board: np.ndarray, my_id: int
) -> tuple[list[np.ndarray], list[tuple]]:
    actions = []
    boards = []
    for i in range(5):
        for j in range(5):
            if i in {0, 4} or j in {0, 4}:
                if board[j, i] in {-1, my_id}:
                    for k in {0, 1, 2, 3} - get_cell_sides((i, j)):
                        actions.append(((i, j), Move(k)))
                        boards.append(get_new_board(board, my_id, ((i, j), Move(k))))
    return boards, actions


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


def inverse_map_board(t: tuple[int, int]):
    i = None
    for i in range(16):
        if map_board(i) == t:
            break
    return i


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


def check_win_board(board, turn, depth=2):
    """0: win 0, 1: win 1"""
    winner = -1
    for i in range(5):
        if board[i, 0] != -1 and all(board[i, :] == board[i, 0]):
            winner = board[i, 0]
        if board[0, i] != -1 and all(board[:, i] == board[0, i]):
            winner = board[0, i]
    if board[0, 0] != -1 and all(board[i, i] == board[0, 0] for i in range(5)):
        winner = board[0, 0]
    if board[0, 4] != -1 and all(board[i, 4 - i] == board[0, 4] for i in range(5)):
        winner = board[0, 4]
    if (
        winner == -1
        and depth > 0
        and (
            5 - how_many_in_line(board, turn) <= depth
            or 5 - how_many_in_line(board, (turn + 1) % 2) <= depth
        )
    ):
        boards, _ = get_possible_actions(board, turn)
        winners = []
        for board_ in boards:
            winner_ = check_win_board(board_, (turn + 1) % 2, depth=depth - 1)
            winners.append(winner_)
            if turn == winner_:
                break
        if turn in winners:
            winner = turn
        elif -1 in winners:
            winner = -1
        else:
            winner = (turn + 1) % 2
    return winner


def check_win(board, turn):
    """0: win 0, 1: win 1"""
    winner = -1
    for i in range(5):
        if board[i, 0] != -1 and all(board[i, :] == board[i, 0]):
            winner = board[i, 0]
        if board[0, i] != -1 and all(board[:, i] == board[0, i]):
            winner = board[0, i]
    if board[0, 0] != -1 and all(board[i, i] == board[0, 0] for i in range(5)):
        winner = board[0, 0]
    if board[0, 4] != -1 and all(board[i, 4 - i] == board[0, 4] for i in range(5)):
        winner = board[0, 4]
    return winner


def win_possible(board, turn, depth):
    return depth > 0 and (
        5 - how_many_in_line(board, turn) <= depth
        or 5 - how_many_in_line(board, (turn + 1) % 2) <= depth
    )


def minimax_heuristic(board, turn):
    return (how_many_in_line(board, 0) - how_many_in_line(board, 1)) / 5
