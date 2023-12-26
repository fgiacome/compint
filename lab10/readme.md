# Lab 10: tic tac toe
In this lab, I created a computer program which is able to play tic-tac-toe.
The algorithm employed to teach the program how to play is value iteration.
The value iteration algorithm, compared to other algorithms in the reinforcement learning literature, employs a full-breadth lookahead rather than episode sampling.
It is possible to use it in this case thanks to the limited number of states and the knowledge of the state transitions.

However, the vanilla value iteration update is as follows:

$$
    v_{k+1}(s) = \max_{s' \in S(s)} v_k(s')
$$

where $S(s)$ is the set of states that can be reached from $s$ in one move.
Note that I don't consider a set of actions since the state transitions are deterministic.
Since this is a two-player game, however, it is necessary to take into account the moves of the opponent in order to find the set of successor states.
By the argument that playing the opponent is a completely symmetrical problem, the modified update of the value iteration algorithm performs a two-step lookahead:

$$
    s^* = \arg\max_{s' \in S(s)}v_k(s')\\
$$

$$
    v_{k+1}(s) = \min_{s' \in S(s^*)}v_k(s')
$$

where, once again, $S(s)$ is the set of states reachable in one move from $s$.

In practice, I do not consider the states were it is not the agent's turn in the value function (ie, their value is undefined), and rather perform the first step lookahead as follows:

$$
    s^* = \arg\min_{s' \in S(s)}v_k(\neg s')
$$

where $\neg s$ is obtained from state $s$ by swapping the agent and the opponent, thus falling back to a state where it is the agent's turn.
However, since the roles are swapped, rather than maximizing and minimizing, I minimize twice.
This can be seen as minimizing the return from the point of view of the opponent.

The algorithm can be run by directly executing the file `main.py`.
The program will find the optimal value function and then offer to play an interactive game.
By switchin the flag `start`, one can control who begins the game.

## Acknowledgements
The idea of using value iteration was taken from applications seen in the course _Robot learning_, given at Politecnico di Torino in the academic year 2023/24.

As a reference, I used

> R. S. Sutton, A. G. Barto, _Reinforcement learning_, 2nd edition, The MIT Press, 2018.