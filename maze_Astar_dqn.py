
# coding: utf-8

# # Solving Maze with A-star algorithm, Q-learning and Deep Q-network

# ### Objective of this notebook is to solve self-made maze with A-star algorithm, Q-learning and Deep Q-network.
# ### The maze is in square shape, consists of start point, goal point and tiles in the mid of them.
# ### Each tile has numericals as its point. In other words, if you step on to the tile with -1, you get 1 point subtracted.
# ### The maze has blocks to prevent you from taking the route.

# In[1]:

import numpy as np
import pandas as pds
import random
import copy
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam, RMSprop
from collections import deque
from keras import backend as K


# In[ ]:




# # Maze Class

# In[2]:

class Maze(object):
    def __init__(self, size=10, blocks_rate=0.1):
        self.size = size if size > 3 else 10
        self.blocks = int((size ** 2) * blocks_rate) 
        self.s_list = []
        self.maze_list = []
        self.e_list = []

    def create_mid_lines(self, k):
        if k == 0: self.maze_list.append(self.s_list)
        elif k == self.size - 1: self.maze_list.append(self.e_list)
        else:
            tmp_list = []
            for l in range(0,self.size):
                if l == 0: tmp_list.extend("#")
                elif l == self.size-1: tmp_list.extend("#")
                else:
                    a = random.randint(-1, 0)
                    tmp_list.extend([a])
            self.maze_list.append(tmp_list)

    def insert_blocks(self, k, s_r, e_r):
        b_y = random.randint(1, self.size-2)
        b_x = random.randint(1, self.size-2)
        if [b_y, b_x] == [1, s_r] or [b_y, b_x] == [self.size - 2, e_r]: k = k-1
        else: self.maze_list[b_y][b_x] = "#"
            
    def generate_maze(self): 
        s_r = random.randint(1, (self.size / 2) - 1)
        for i in range(0, self.size):
            if i == s_r: self.s_list.extend("S")
            else: self.s_list.extend("#")
        start_point = [0, s_r]

        e_r = random.randint((self.size / 2) + 1, self.size - 2)
        for j in range(0, self.size):
            if j == e_r: self.e_list.extend([50])
            else: self.e_list.extend("#")
        goal_point = [self.size - 1, e_r]

        for k in range(0, self.size):
            self.create_mid_lines(k)
        
        for k in range(self.blocks):
            self.insert_blocks(k, s_r, e_r)

        return self.maze_list, start_point, goal_point


# # Maze functions

# In[3]:

class Field(object):
    def __init__(self, maze, start_point, goal_point):
        self.maze = maze
        self.start_point = start_point
        self.goal_point = goal_point
        self.movable_vec = [[1,0],[-1,0],[0,1],[0,-1]]

    def display(self, point=None):
        field_data = copy.deepcopy(self.maze)
        if not point is None:
                y, x = point
                field_data[y][x] = "@@"
        else:
                point = ""
        for line in field_data:
                print ("\t" + "%3s " * len(line) % tuple(line))

    def get_actions(self, state):
        movables = []
        if state == self.start_point:
            y = state[0] + 1
            x = state[1]
            a = [[y, x]]
            return a
        else:
            for v in self.movable_vec:
                y = state[0] + v[0]
                x = state[1] + v[1]
                if not(0 < x < len(self.maze) and
                       0 <= y <= len(self.maze) - 1 and
                       maze[y][x] != "#" and
                       maze[y][x] != "S"):
                    continue
                movables.append([y,x])
            if len(movables) != 0:
                return movables
            else:
                return None

    def get_val(self, state):
        y, x = state
        if state == self.start_point: return 0, False
        else:
            v = float(self.maze[y][x])
            if state == self.goal_point: 
                return v, True
            else: 
                return v, False


# # Generate a maze

# In[7]:

size = 10
barriar_rate = 0.1

maze_1 = Maze(size, barriar_rate)
maze, start_point, goal_point = maze_1.generate_maze()
maze_field = Field(maze, start_point, goal_point)

maze_field.display()


# In[ ]:




# # Solving the maze with A-star algorithm
# ### https://en.wikipedia.org/wiki/A*_search_algorithm

# In[8]:

class Node(object):    
    def __init__(self, state, start_point, goal_point):
        self.state = state
        self.start_point = start_point
        self.goal_point = goal_point
        self.hs = (self.state[0] - self.goal_point[0]) ** 2 + (self.state[1] - self.goal_point[1]) ** 2
        self.fs = 0
        self.parent_node = None
    
    def confirm_goal(self):
        if self.goal_point == self.state: return True
        else: return False


# In[9]:

class NodeList(list):
    def find_nodelist(self, state):
        node_list = [t for t in self if t.state==state]
        return node_list[0] if node_list != [] else None
    def remove_from_nodelist(self, node):
        del self[self.index(node)]


# In[48]:

class Aster_Solver(object):
    def __init__(self, maze, start_point, goal_point, display=False):
        self.Field = maze
        self.start_point = start_point
        self.goal_point = goal_point
        self.open_list = NodeList()
        self.close_list = NodeList()
        self.steps = 0
        self.score = 0
        self.display = display
        
    def set_initial_node(self):
        node = Node(self.start_point, self.start_point, self.goal_point)
        node.start_point = self.start_point
        node.goal_point = self.goal_point 
        return node
                
    def go_next(self, next_actions, node):
        node_gs = node.fs - node.hs
        for action in next_actions:
            open_list = self.open_list.find_nodelist(action)
            dist = (node.state[0] - action[0]) ** 2 + (node.state[1] - action[1]) ** 2
            if open_list:
                if open_list.fs > node_gs + open_list.hs + dist:
                    open_list.fs = node_gs + open_list.hs + dist
                    open_list.parent_node = node
            else:
                open_list = self.close_list.find_nodelist(action)
                if open_list:
                    if open_list.fs > node_gs + open_list.hs + dist:
                        open_list.fs = node_gs + open_list.hs + dist
                        open_list.parent_node = node
                        self.open_list.append(open_list)
                        self.close_list.remove_from_nodelist(open_list)
                else:
                    open_list = Node(action, self.start_point, self.goal_point)
                    open_list.fs = node_gs + open_list.hs + dist
                    open_list.parent_node = node
                    self.open_list.append(open_list)
    
    def solve_maze(self):
        node = self.set_initial_node()
        node.fs = node.hs
        self.open_list.append(node)
        
        while True:            
            node = min(self.open_list, key = lambda node:node.fs)
            print ("current state:  {0}".format(node.state))
            
            if self.display:
                self.Field.display(node.state)
            
            reward, tf = self.Field.get_val(node.state)
            self.score =  self.score + reward
            print("current step: {0} \t score: {1} \n".format(self.steps, self.score))
            self.steps += 1
            if tf == True:
                print("Goal!")
                break

            self.open_list.remove_from_nodelist(node)
            self.close_list.append(node)
            
            next_actions = self.Field.get_actions(node.state)   
            self.go_next(next_actions, node)


# In[49]:

astar_Solver = Aster_Solver(maze_field, start_point, goal_point, display=True)
astar_Solver.solve_maze()


# In[ ]:




# # Solving the maze in Q-learning
# ### https://en.wikipedia.org/wiki/Q-learning

# In[50]:

class QLearning_Solver(object):
    def __init__(self, maze, display=False):
        self.Qvalue = {}
        self.Field = maze
        self.alpha = 0.2
        self.gamma  = 0.9
        self.epsilon = 0.2
        self.steps = 0
        self.score = 0
        self.display = display

    def qlearn(self, greedy_flg=False):
        state = self.Field.start_point
        while True:
            if greedy_flg:
                self.steps += 1
                action = self.choose_action_greedy(state)
                print("current state: {0} -> action: {1} ".format(state, action))
                if self.display:
                    self.Field.display(action)
                reward, tf = self.Field.get_val(action)
                self.score =  self.score + reward
                print("current step: {0} \t score: {1}\n".format(self.steps, self.score))
                if tf == True:
                    print("Goal!")
                    break
            else:
                action = self.choose_action(state)    
            if self.update_Qvalue(state, action):
                break
            else:
                state = action

    def update_Qvalue(self, state, action):
        Q_s_a = self.get_Qvalue(state, action)
        mQ_s_a = max([self.get_Qvalue(action, n_action) for n_action in self.Field.get_actions(action)])
        r_s_a, finish_flg = self.Field.get_val(action)
        q_value = Q_s_a + self.alpha * ( r_s_a +  self.gamma * mQ_s_a - Q_s_a)
        self.set_Qvalue(state, action, q_value)
        return finish_flg


    def get_Qvalue(self, state, action):
        state = (state[0],state[1])
        action = (action[0],action[1])
        try:
            return self.Qvalue[state][action]
        except KeyError:
            return 0.0

    def set_Qvalue(self, state, action, q_value):
        state = (state[0],state[1])
        action = (action[0],action[1])
        self.Qvalue.setdefault(state,{})
        self.Qvalue[state][action] = q_value

    def choose_action(self, state):
        if self.epsilon < random.random():
            return random.choice(self.Field.get_actions(state))
        else:
            return self.choose_action_greedy(state)

    def choose_action_greedy(self, state):
        best_actions = []
        max_q_value = -100
        for a in self.Field.get_actions(state):
            q_value = self.get_Qvalue(state, a)
            if q_value > max_q_value:
                best_actions = [a,]
                max_q_value = q_value
            elif q_value == max_q_value:
                best_actions.append(a)
        return random.choice(best_actions)

    def dump_Qvalue(self):
        print("##### Dump Qvalue #####")
        for i, s in enumerate(self.Qvalue.keys()):
            for a in self.Qvalue[s].keys():
                print("\t\tQ(s, a): Q(%s, %s): %s" % (str(s), str(a), str(self.Qvalue[s][a])))
            if i != len(self.Qvalue.keys())-1: 
                print('\t----- next state -----')


# In[53]:

learning_count = 1000
QL_solver = QLearning_Solver(maze_field, display=True)
for i in range(learning_count):
    QL_solver.qlearn()

QL_solver.dump_Qvalue()


# In[54]:

QL_solver.qlearn(greedy_flg=True)


# In[ ]:




# In[ ]:




# # Solving the maze in Deep Q-learning
# ### https://deepmind.com/research/dqn/

# In[55]:

class DQN_Solver:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.9
        self.epsilon = 1.0
        self.e_decay = 0.9999
        self.e_min = 0.01
        self.learning_rate = 0.0001
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_shape=(2,2), activation='tanh'))
        model.add(Flatten())
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss="mse", optimizer=RMSprop(lr=self.learning_rate))
        return model

    def remember_memory(self, state, action, reward, next_state, next_movables, done):
        self.memory.append((state, action, reward, next_state, next_movables, done))

    def choose_action(self, state, movables):
        if self.epsilon >= random.random():
            return random.choice(movables)
        else:
            return self.choose_best_action(state, movables)
        
    def choose_best_action(self, state, movables):
        best_actions = []
        max_act_value = -100
        for a in movables:
            np_action = np.array([[state, a]])
            act_value = self.model.predict(np_action)
            if act_value > max_act_value:
                best_actions = [a,]
                max_act_value = act_value
            elif act_value == max_act_value:
                best_actions.append(a)
        return random.choice(best_actions)

    def replay_experience(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        X = []
        Y = []
        for i in range(batch_size):
            state, action, reward, next_state, next_movables, done = minibatch[i]
            input_action = [state, action]
            if done:
                target_f = reward
            else:
                next_rewards = []
                for i in next_movables:
                    np_next_s_a = np.array([[next_state, i]])
                    next_rewards.append(self.model.predict(np_next_s_a))
                np_n_r_max = np.amax(np.array(next_rewards))
                target_f = reward + self.gamma * np_n_r_max
            X.append(input_action)
            Y.append(target_f)
        np_X = np.array(X)
        np_Y = np.array([Y]).T
        self.model.fit(np_X, np_Y, epochs=1, verbose=0)
        if self.epsilon > self.e_min:
            self.epsilon *= self.e_decay


# In[56]:

state_size = 2
action_size = 2
dql_solver = DQN_Solver(state_size, action_size)

episodes = 20000
times = 1000

for e in range(episodes):
    state = start_point
    score = 0
    for time in range(times):
        movables = maze_field.get_actions(state)
        action = dql_solver.choose_action(state, movables)
        reward, done = maze_field.get_val(action)
        score = score + reward
        next_state = action
        next_movables = maze_field.get_actions(next_state)
        dql_solver.remember_memory(state, action, reward, next_state, next_movables, done)
        if done or time == (times - 1):
            if e % 500 == 0:
                print("episode: {}/{}, score: {}, e: {:.2} \t @ {}"
                        .format(e, episodes, score, dql_solver.epsilon, time))
            break
        state = next_state
    dql_solver.replay_experience(32)


# In[58]:

state = start_point
score = 0
steps = 0
while True:
    steps += 1
    movables = maze_field.get_actions(state)
    action = dql_solver.choose_best_action(state, movables)
    print("current state: {0} -> action: {1} ".format(state, action))
    reward, done = maze_field.get_val(action)
    maze_field.display(state)
    score = score + reward
    state = action
    print("current step: {0} \t score: {1}\n".format(steps, score))
    if done:
        maze_field.display(action)
        print("goal!")
        break


# In[ ]:




# In[ ]:



