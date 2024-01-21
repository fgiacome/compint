import argparse
from agent import NeuralPlayer, Policy
from game import Game, Player
from main import RandomPlayer
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=str)
parser.add_argument("--opponent", type=str, default=None)
parser.add_argument("--test_episodes", type=int, default=100)
parser.add_argument("--print_board", action="store_true")
args = parser.parse_args()

policy = Policy()
policy.load_state_dict(torch.load(args.policy))
player = NeuralPlayer(policy=policy, train=False, print_board=args.print_board)
opponent = RandomPlayer()
if args.opponent is not None:
    policy = Policy()
    policy.load_state_dict(torch.load(args.opponent))
    opponent = NeuralPlayer(policy=policy, train=False)

wins = 0
for episode in range(args.test_episodes):
    if args.print_board: print(f"Game #{episode+1}")
    game = Game()
    game.current_player_idx = np.random.randint(0,2)
    result = game.play(player, opponent)
    if result == 0: wins += 1

print(f"Wins: {wins} out of {args.test_episodes} games, rate: {wins/args.test_episodes}")