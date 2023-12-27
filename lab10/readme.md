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

## Update (important!)
The procedure described above is implemented in the commit with tag `lab10-original` (it was my original submission).
I changed it for two reasons:
1. The procedure converges to the optimal value function against an optimal player.
This makes the procedure blind to the possibility of the opponent being non-optimal: the algorithm considers as equivalent a move which leads to a situation where a win is impossible and a move which leads to a situation where a win is possible provided that the opponent makes a mistake.
2. To address the previous point, I changed the policy of the second step of the lookahead to choose a random action with small probability, so that the agent would account for the possibility of the opponent making mistakes.
With this update, I realized an even deeper problem with my original approach: the algorithm does converge (within a small tolerance), but it assigns different values to symmetrical positions, which means that the convergence value is arbitrary (that is, it depends on the initial conditions).

I suppose that the reason for the latter point is that the opponent was modeled according to the agent's own value function.
In the reinforcement learning language, this means that the model of the environment depends on the policy of the agent.
This is indeed the only significant difference with the vanilla value iteration.
So, I switched to a simpler model of the opponent: a random player.
I did not go this way at first because I wanted to try to model a "smart" opponent;
however, even if the opponent is a random player, the optimal value function is higher for the states where a win is guaranteed compared to the other states, only the difference will be smaller.
In turn, this means that the convergence tolerance should be set to a higher value. 
In practice, though, for such a simple game the convergence is instantaneous in either case.
Finally, while I did not check this result with any of my classmates, the optimal value function against a random player found by the value iteration algorithm should be the same found by the Monte Carlo algorithm employing episode sampling.

## Acknowledgements
The idea of using value iteration was taken from applications seen in the course _Robot learning_, given at Politecnico di Torino in the academic year 2023/24.

As a reference, I used

> R. S. Sutton, A. G. Barto, _Reinforcement learning_, 2nd edition, The MIT Press, 2018.