# Quixo player

In this end-of-course project, I developed an intelligent player that is able to play the game Quixo.

The final player can be found in the file `mcts.py`, where it is named `MctsPlayer`.
The player can be used in conjunction with the classes and methods found in the files `main.py` and `game.py`, which were provided as part of the assignment.
The files `utils.py` and `mcts_node.py` implement helper classes and methods for the `MctsPlayer`.
The file `test_mcts.py` can be used to make test runs of the `MctsPlayer` running against an opponent playing randomly.
The file `train_policygradient.py` was used to train an earlier version of the player (found in `agent.py`, named `NeuralPlayer`) using vanilla REINFORCE, a policy gradient algorithm.
The trained checkpoint can be found in `policy_training_30000.mdl`, and can be tested using the file `test_policygradient.py`, however I have later switched to a more traditional method since policy gradient by itself proved unsatisfactory.
A natural continuation of this project would be to merge the two methods to make a stronger player.

The `MctsPlayer` uses Monte Carlo Tree Search in combination with a Minimax search to determine the best move.
Monte Carlo Tree Search is an algorithm famously used to build a strong player in the game of Go.
It employes upper confidence bounds to the results of rollouts (simulations) conducted from the leaves of the grame tree, in order to choose the best move at each level of the tree.
The resulting tree is deepened towards the more promising moves, and it can be discarded or retained for further evaluation at the next turn.
The Minimax search is used to check whether a move is deterministically going to lead to a defeat (assuming optimal play from the adversary), thus saving the available simulations of Mcts to investigate the other moves.
No heuristic is backed up by Minimax, but only the existance of winning or losing paths; together with the pruning strategy, this allows for rapidly looking 4 moves ahead in the game tree.
Without a Minimax search, the Mcts algorithm with 300 rollouts is able to defeat the Random Player every time;
however, it is easily defeated by even a shallow Minimax employing an informative heuristic.
Such adversary was taken with his permission from the repository of my classmate at https://github.com/lorenzofezza00/CI_LABS/tree/main/quixo.
Nonetheless, it is interesting to notice that plain Mcts is still able to defeat the "heuristic Minimax" when some uncommon cases arise, where the heuristic employed by the Minimax is not as effective.
The combination of Mcts and Minimax provides a much stronger player, but it must be noted that this strategy is computationally expensive and it is still unable to win consistently against a the heuristic Minimax: the two players most commonly reach a draw where the same moves are repeated in a loop.

In conclusion, the Mcts player informed by a Minimax search can be a strong Quixo player, however a simpler Minimax with a good heuristic seems to be comparably strong and much more computationally efficient.

An afterword on policy gradient: the policy gradient method used in this experiment used a pool of opponents made by a random player (initially) and previous iterations of the same policy.
This method was employed to train Google's AlphaGo policy networks.
In the case of this project, my network was made of 4 fully connected layers, including input and output, where the input was the numerical representation of the game board, the output was divided in two groups of logits: four indicating the side to place the removed cell, and 16 indicating the cell to take;
there was a residual connection between the input and output layer.
The hidden layers were made of 64 neurons each, and at each iteration the probabilities were normalized after discarding the illegal moves.
The result was that the network learned how to win against the random player *most of the times*, using a very simple strategy:
play always the same move.
With a bit of luck, this results in the whole row getting filled with the random player never interfering.
After 30000 episodes (approximately one hour) of training, it did not seem that the network had learned any more sophisticated strategies, and it was easily beaten by the Mcts and sometimes even by the random player.
This result, in my opinion, emphasizes that a good policy network for a modestly complex game such as this one must be much larger and trained with a lot more samples.
It is also possible that the reason for this unsatisfactory result lies in the inherently high variance of the observations drawn with this training procedure, and that an algorithm such as Actor Critic (in any of its many forms) could be more effective.