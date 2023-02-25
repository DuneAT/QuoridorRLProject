# Quoridor_RL_Project


The primary objective of this deep learning project is to train an artificial intelligence (AI) system to play the board game Quoridor using reinforcement learning. The project utilizes the Pytorch, Pygames, and gym libraries to implement a deep learning algorithm that can analyze the game board and make strategic decisions based on the current state of the game. 

To launch project, use the Jupter Notebook file "Project" after excctracting the whole project in one same folder.

It is possible to change the number of epochs and the training mode in the last celle of the Notebook, in the function :

trained_agent = agentVSdummy(test_env,
             test_agent,
             dummy_agent_V1,
             10,
             True)

The AI system is trained using reinforcement learning, which involves rewarding the system for making good moves and penalizing it for making poor ones, allowing the system to learn and improve its gameplay strategy over time. The ultimate goal is to develop an intelligent agent that can compete with human players in the game of Quoridor. 

We used GYM library in order to create a Quoridor environment where agents could move their pawns and place fences, and managed to train an AI agent on it. The strategy we opted for was a model-free training (no initial guidance), and DQN resolution for neural network implementation and training. The first choice we made was about the form of the reward function. We decided to penalize the loss of the game, reward winning the game, but also positive or negative reward in function of the Manhattan distance to the goal (for both opponent and AI agent). After that, because Quoridor is a 2 player game, we had to chose an opponent to train our AI agent, we proceeded by steps : random dummy agents, which was the worse result of all. The learning rate was encouraging, but the strategy was to much overfitting the random behaviour of the opponent. Then, we decided to try semi-random dummy agent, by reducing the probability of placing fences (no efficacity), then by increasing the probability to reach the winning goal domain of the board. This last strategy managed to give interesting results of training, as our trained AI agent were now following a more original and adaptative pattern that allowed it to avoid obstacles. It also preferentially moved toward the direction of the goal, which was satisfying compared to the 2 previous ones. Finally, we decided to train 2 AI agents simultaneously, but this method didn't get the results we hoped : the speed of training was very low compared to the ones with dummy agents, and then this method wasn't efficient at all. Their behaviour was overfitted between themselves and with no strategic interest for players and no adaptability from the 2 output agents.

Further work could be done on the reward function, which at the moment does not reward the action itself, but rather the raw position of the player. It would be interesting to reward the evolution of the situation due to the action.
