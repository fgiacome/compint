# Computational Intelligence 2023/24
This repository contains my code for the computational intelligence
course.

## Quixo

End-of-course project: I recently updated this to clean the code and improve performance using PyO3 for writing a Python module in Rust.
The agent in this repository can play Quixo according to the specification provided by the instructors via the `Game` class in `game.py`, using a pure Monte Carlo Tree Search (MCTS) strategy.
Thanks to the use of compiled code, the agent can simulate 100.000 games per move, compared to only 300 with the Python-only implementation.

## Lab 10

A tic-tac-toe player trained with value iteration.

## Lab 9 and Lab 2

Using an evolutionary strategy and a genetic algorithm to train a nim player and to perform black-box optimization, respectively.