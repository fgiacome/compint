# NIM 

## Task description and solution

In this lab, I trained an agent to play the classical mathematcal game "nim" via
an evolutionary search algorithm.  The core code of the agent is located in file
`players.py`.  Several kinds of agents (also called players) are present, each
represented by a subclass of `NimPlayer`.  The player is, essentialy, a piece of
code that given a state returns an action.  The mathematically optimal nim
player is also present in the `players` module.  A random agent (which performs
a completely random move) is present.  Then, several "simple" agents are
present.  All the simple agents calculate the move with a simple algorithm, and
when the algorithm cannot find a legal move, they call a backup function that is
currently set to making a random move.  These simple agents always act on the
tallest stack, by making it either equal to the second tallest
(`MakeTwinTowers`), taller by only one (`KeepTaller`), shorter by only one
(`ChangeTaller`), by removing it entirely (`TakeAllFromTallest`) or by leaving
only one match in it (`TakeAllButOneFromTallest`). The only simple agent that
does not act on the tallest stack is `TakeA1Line`, which removes a stack with
only one element.  These simple agents represent optimal moves in simple
configurations that are likely to appear at the end of a Nim game. Each of these
configurations can be idenfied by a number of attributes that include the parity
of the number of nonempty stacks, whether the second-tallest stack has more than
one element, whether it is equal to the tallest stack, the parity of its number
of elements.  These 4 attributes are encoded as a boolean 4-tuple, for a total
of 16 possible states.  The agent to be trained is coded in
`StochasticRulesBased`, and at its core lies a matrix of real weights with 6
rows (one for each simple agent) and 16 columns (one for each state).  At
inizialitazion the weight matrix is passed to the agent and each column is
turned into a probability distribution via a softmax function.  The normalized
weights encode the probability that a given simple agent is chosen given the
state. The agent projects the nim state onto one of the 16 states described
above, and looks up the corresponding column in the normalized weight matrix to
employ as a probability distribution.  The goal of the agent is to associate to
each of the 16 states the ideal simple agent (if any).  With the correct
association, the agent is not optimal, but it should be able to win in various
situations, specifically whenever only one or two stacks are left with more than
one element (and a win is possible).  The agent is trained with a (mu,lambda)
evolutionary strategy, employing as fitness measure the number of matches won
against an adversary, with the total number of matches being a constant.  All
the constants are configurable in the `evolution_strategy.py` file, and it is
sufficient to execute it to run a training.  At the end of each training, a new
entry is added to the file `history.txt` with all the relevant data, and a plot
of the best win rate per generation is displayed. Since the evolutionary
strategy employed here always replaces the previous generation, the win rate can
decrease because of the loss of a performing individual. Furthermore,
oscillations in the win rate are to be expected due to the stochasticity of the
players and the games.

## Obervations

The trained agent is able to defeat the random agent (employed as adversary in
experiments 1, 2, 3, 4, 5, 6, 9) in most matches after a short training.
However, the learning curve saturates quickly (as can be seen in experiment 5)
and the final performance depends strongly on the starting conditions, rather
than on the length of the training, peaking at about 90-95 games won out of
100.  The agent was also trained against a `RandomlyOptimal` agent, which
performs a random move with probability 0.1, 0.3 and an optimal move with
probability 0.9, 0.7 in experiments 7 and 8 respectively.  After 30 generations
(about 5 minutes of training), the agent was unable to improve its performance
against the `RandomlyOptimal` agent in experiment 7, but showed a significant
increase in performance in experiment 8 (reaching about 45 wins out of 100).
This might indicate that the weaker adversary of experiment 8 resulted in a
smoother fitness landscape. These results show that an evolutionary strategy can
efficiently find the optimal association between the set of 16 states and the
actions.

## Credits

The core code for the (mu,lambda) evolutionalry strategy, the `Nim` class, the
optimal player, the random player and the `match` function has been adapted from
the code provided by Professor Squillero in his course repository
([link](https://github.com/squillero/computational-intelligence/blob/master/2023-24/rastrigin.ipynb)).
Several chenges have been made, most remarkably the optimal player was not truly
optimal in its original version and the search for the optimal strategy was too
slow to efficiently serve as an adversary during the training.