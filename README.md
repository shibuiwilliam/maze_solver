# Maze_solver
This project solves self-made maze in a variety of ways: A-star, Q-learning and Deep Q-network.

## The project consists of mainly four parts.

1. Maze generator consists of maze class and field class. It generates a square shaped maze. The maze has route tiles, wall and block tiles, starting and goal point. The route tiles have -1 or 0 on it, which is the point you can get by stepping it. Apparently you will get 1 point subtracted if you step on -1 tile. The wall and block tiles, in #, are where you cannot interude. You have to bypass #. The starting point, namely S, is where you start the maze and goal point, which is shown as 50, is where you head to. You will earn 50 points when you made to the goal.

2. A-star maze solver has node class, node_list class and astar-solver class. The A-star algorithm tries to find a path from start point to goal point with the process of plotting an efficiently directed path between multiple points, called nodes. It calculates Euclidean distance between where you to where you head to, and chooses the point which gives shortest distance to the goal. For precise explanation of A-star algorithm, refer to wikipedia.
https://en.wikipedia.org/wiki/A*_search_algorithm

3. Q-learning has Q-learning_solver class. This class solves the maze in reinforcement learning manner. The class samples state, action and its reward randomly, and figures out the path from start to goal to maximize the earning point. Q-learning estimates the optimal future value of reward from present state and action. Q-learning uses the following equation to update arbitrary fixed value of expected reward.

Q(s,a)←Q(s,a)+α(r+γ maxa′Q(s′,a′)−Q(s,a))

```math
Q(s,a) \leftarrow Q(s,a) + \alpha \Bigl(r + \gamma\ \max_{a'} Q(s',a') - Q(s,a)\Bigr)
```

https://en.wikipedia.org/wiki/Q-learning

4. Deep Q-network has DQN_solver class. The class uses deep q-network technique, the neural network enhanced style of q-learning. While Q-learning estimates optimal future value with the previous equation, Deep Q-network aims to approximate the optimal future value of reward with this equation and neural network. For the neural network, input variables are state and action, while target variable is calculated through the equation.

targett=rt+γmaxa′Qθk(s′t,a′)

$ {\rm{target}}^t = r_t + \gamma \max_{a'} Q_{\theta_k} (s_t',a') $

https://deepmind.com/research/dqn/


## How to use.
Just run the notebook.
The neural network is modelled with Keras, so you need Keras pip installed.
https://keras.io/
