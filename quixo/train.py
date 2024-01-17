import torch
import numpy as np
import argparse
from agent import Policy, NeuralPlayer
from game import Game
from main import RandomPlayer
from tqdm import tqdm
import random
import os
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=str, default=None, help="Policy checkpoint to use")
parser.add_argument("--episodes", type=int, default=10_000, help="Number of training episodes")
parser.add_argument("--save_period", type=int, default=100, help="How often to generate a new adversary")
parser.add_argument("--checkpoint_period", type=int, default=100, help="How often to save a checkpoint")
parser.add_argument("--load_in_folder", action="store_true", help="Wether to load all .mdl files in the current folder in the opponents pool")
parser.add_argument("--episode_count", type=int, default=0, help="Start episode count from...")
args = parser.parse_args()

policy = Policy()
if args.policy is not None:
    print("Loading policy...")
    policy.load_state_dict(torch.load(args.policy))

player = NeuralPlayer(policy, True)
random_player = RandomPlayer()
episode = 0 + args.episode_count
opponents = [random_player]

if args.load_in_folder:
    print("Loading opponents...")
    count = 0
    for file in os.listdir():
        if file[-4:] == ".mdl":
            opponent_policy = Policy()
            opponent_policy.load_state_dict(torch.load(file))
            opponent = NeuralPlayer(policy=opponent_policy, train=False)
            opponents.append(opponent)
            count += 1
    print(f"Loaded {count} opponents.")

rewards = []

pbar = tqdm(total=args.episodes-args.episode_count)
while episode < args.episodes:
    game = Game()
    game.current_player_idx = np.random.randint(0,2)
    opponent = random.choice(opponents)
    reward = game.play(player, opponent)
    if reward == 0: reward = 1
    elif reward == -1: reward = 0
    else: reward = -1
    player.episode_finished(reward)
    rewards.append(reward)
    episode += 1
    pbar.update(1)
    if episode % args.save_period == 0:
        new_opponent= deepcopy(player)
        new_opponent.train = False
        opponents.append(new_opponent)
    if episode % args.checkpoint_period == 0:
        torch.save(player.policy.state_dict(), f"policy_training_{episode}.mdl")
    if episode % 100 == 0:
        avg_reward = sum(rewards[-min(len(rewards), 50):])/min(len(rewards),50)
        print(f"Avg reward after {episode} training episodes: {avg_reward:0.5f}")

pbar.close()

torch.save(player.policy.state_dict(), f"policy_training_{episode}.mdl")