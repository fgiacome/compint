import torch
import torch.nn.functional as F
import numpy as np
from game import Player, Move, Game

OUTPUT_SPACE = 4 + 16 #4*16
INPUT_SPACE = 25 #5*5
HIDDEN_NEURONS = 64

class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(INPUT_SPACE, HIDDEN_NEURONS)
        self.fc1a = torch.nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
        self.fc1b = torch.nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
        self.fc2 = torch.nn.Linear(HIDDEN_NEURONS, OUTPUT_SPACE)
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.fc1(x)
        x_r = F.relu(x)
        x = F.relu(self.fc1a(x_r))
        x = F.relu(self.fc1b(x))
        x = self.fc2(x + x_r)
        side = F.softmax(x[:4], dim=0)
        cell = F.softmax(x[4:], dim=0)
        return side, cell

class NeuralPlayer(Player):
    def __init__(self, policy: Policy, train=True, print_board=False, eps=0.1):
        self.policy = policy
        self.train = train
        self.print_board = print_board
        self.action_probs = []
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3, weight_decay=0.001)
        self.eps = eps

    def map_board(self, x: int):
        if x//4 == 0:
            pos = (x%4, 0)
        elif x//4 == 1:
            pos = (4,x%4)
        elif x//4 == 2:
            pos = (4-x%4,4)
        elif x//4 == 3:
            pos = (0,4-x%4)
        return pos

    def get_cell_sides(self, cell):
        cell_sides = set()
        if cell[0] == 0: cell_sides.add(2)
        if cell[0] == 4: cell_sides.add(3)
        if cell[1] == 0: cell_sides.add(0)
        if cell[1] == 4: cell_sides.add(1)
        return cell_sides
    
    def get_allowed_action_probs(self, y_side: torch.Tensor, y_cell: torch.Tensor, game: Game):
        y_side = y_side.reshape(4,1)
        y_cell = y_cell.reshape(1,16)
        board = game.get_board()
        my_id = game.get_current_player()
        actions = y_side @ y_cell
        actions = actions + 1e-8
        for i in range(16):
            cell = self.map_board(i)
            if board[cell[1],cell[0]] not in {-1, my_id}:
                actions[:,i] = 0
                continue
            for side in self.get_cell_sides(cell):
                actions[side,i] = 0
        actions = actions / actions.sum()
        return actions 

    def make_move(self, game: Game):
        board = game.get_board()
        x_np = board.reshape((25,))
        x = torch.tensor(x_np, dtype=torch.float32)
        y_side, y_cell = self.policy.forward(x)
        # sample an action from the allowed action space
        action_probs = self.get_allowed_action_probs(y_side, y_cell, game)
        if self.train:
            action_probs[action_probs==0] += self.eps
            action_probs /= action_probs.sum()
        action_probs_np = action_probs.detach().numpy().ravel()
        try:
            action = np.random.choice(64, p = action_probs_np)
        except ValueError:
            breakpoint() 
        cell_ = action % 16
        side_ = action // 16
        cell = self.map_board(cell_)
        side = Move(side_)
        if self.train:
            self.action_probs.append(torch.log(action_probs[side_,cell_]))
        if self.print_board:
            print(board)
            print(f"NeuralPlayer going for {cell}, {side}.")
        return cell, side
    
    def episode_finished(self, reward):
        action_probs = torch.stack(self.action_probs, dim=0).squeeze()
        self.action_probs = []
        if self.train:
            loss = torch.mean(action_probs*(-reward))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        