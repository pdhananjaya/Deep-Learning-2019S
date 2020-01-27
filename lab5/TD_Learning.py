#!/usr/bin/env python
# coding: utf-8

# In[36]:


import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from IPython.display import clear_output


# # 2048 Environment implementation
# 
# from https://github.com/moporgic/2048-Demo-Python/

# In[37]:


import random

class board:
    """simple implementation of 2048 puzzle"""
    
    def __init__(self, tile = None, max_number=15):
        self.tile = tile if tile is not None else [0] * 16
        self.max_num = max_number
    
    def __str__(self):
        state = '+' + '-' * 24 + '+\n'
        for row in [self.tile[r:r + 4] for r in range(0, 16, 4)]:
            state += ('|' + ''.join('{0:6d}'.format((1 << t) & -2) for t in row) + '|\n')
        state += '+' + '-' * 24 + '+'
        return state
    
    def mirror(self):
        return board([self.tile[r + i] for r in range(0, 16, 4) for i in reversed(range(4))])
    
    def transpose(self):
        return board([self.tile[r + i] for i in range(4) for r in range(0, 16, 4)])
    
    def rotate(self):
        return board([self.tile[4*(3-(i%4)) + (i//4)] for i in range(16)])
    
    def left(self):
        move, score = [], 0
        for row in [self.tile[r:r+4] for r in range(0, 16, 4)]:
            row, buf = [], [t for t in row if t]
            while buf:
                if len(buf) >= 2 and buf[0] is buf[1]:
                    buf = buf[1:]
                    buf[0] += 1
                    score += 1 << buf[0]
                row += [buf[0]]
                buf = buf[1:]
            move += row + [0] * (4 - len(row))
        return board(move), score if move != self.tile else -1
    
    def right(self):
        move, score = self.mirror().left()
        return move.mirror(), score
    
    def up(self):
        move, score = self.transpose().left()
        return move.transpose(), score
    
    def down(self):
        move, score = self.transpose().right()
        return move.transpose(), score
    
    def popup(self):
        tile = self.tile[:]
        empty = [i for i, t in enumerate(tile) if not t]
        tile[random.choice(empty)] = random.choice([1] * 9 + [2])
        return board(tile)
    
    def end(self):
        tile = self.tile[:]
        empty = [i for i, t in enumerate(tile) if not t]
        
        count_max_num = np.count_nonzero(self.max_num == np.array(tile))
        return len(empty) == 0 or count_max_num > 0


# In[38]:


def gamestatus(game, maxnum=12):
    counter = [0]*maxnum
    for i in game.tile:
        counter[i]+=1
    return np.array(counter) / len(game.tile)

def showstatus(game):
    s = ""
    for i, p in enumerate(gamestatus(game)):
        s += "{:4d}:[{:3.1f}] ".format(1<<i & -2, p*100.0)
    return s


# # n-Tuples Pattern

# In[39]:


def find_isomorphic_pattern(pattern):
    a = board(list(range(16)))

    isomorphic_pattern = []
    for i in range(8):
        if (i >= 4):
            b = board( a.mirror().tile )
        else:
            b = board( a.tile )
        for _ in range(i%4):
            b = b.rotate()
        isomorphic_pattern.append(np.array(b.tile)[pattern])
        
    return isomorphic_pattern

class TuplesNet():
    def __init__(self, pattern, maxnum):
        self.V = np.zeros(([maxnum]*len(pattern)))
        self.pattern = pattern
        self.isomorphic_pattern = find_isomorphic_pattern(self.pattern)
        
    def getState(self, tile):
        return [tuple(np.array(tile)[p]) for p in self.isomorphic_pattern]
    
    def getValue(self, tile):
        S = self.getState(tile)
        V = [self.V[s] for s in S]
        V = sum(V)

        
        return V
    
    def setValue(self, tile, v, reset=False):
        S = self.getState(tile)
        
        v /= len(self.isomorphic_pattern)
        V = 0.0
        for s in S:
            self.V[s] += v              
            V += self.V[s]
        return V


# # TD learning

# In[41]:


class Agent():
    def __init__(self, patterns, maxnum):
        self.Tuples = []
        for p in patterns:
            self.Tuples.append(TuplesNet(p, maxnum))
        self.metrics = []
        
    def getValue(self, tile):
        V = [t.getValue(tile) for t in self.Tuples]
        V = sum(V)
        return V
    
    def setValue(self, tile, v, reset=False):
        v /= len(self.Tuples)
        V = 0.0
        for t in self.Tuples:
            V += t.setValue(tile, v, reset)
        return V
    
    def evaulate(self, next_games):
        return [ng[1] + self.getValue(ng[0].tile) for ng in next_games]
    
    def learn(self, records, lr):
        exact = 0.0
        for s, a, r, s_, s__ in records: 
            error = exact - self.getValue(s_)
            exact = r + self.setValue(s_, lr*error)
            
    def showStattistic(self, epoch, unit, show=True):
        metrics = np.array(self.metrics[epoch-unit:epoch])
        score_mean = np.mean(metrics[:, 0])
        score_max = np.max(metrics[:, 0])
        
        if show:
            print('{:<8d}mean = {:<8.0f} max = {:<8.0f}'.format(epoch, score_mean, score_max))
        
        if (metrics.shape[1] < 3):
            return score_mean, score_max
        
        end_games = metrics[:, 2]
        reach_nums = np.array([1<<max(end) & -2 for end in end_games])
                  
        if show:
            print('\n')
        
        score_stat = []
        
        for num in np.sort(np.unique(reach_nums)):
            reachs = np.count_nonzero(reach_nums >= num)
            reachs = (reachs*100)/len(metrics)
            ends = np.count_nonzero(reach_nums == num)
            ends = (ends*100)/len(metrics)
            
            if show:
                print('{:<5d}  {:3.1f} % ({:3.1f} %)'.format(num, reachs, ends) )
            
            score_stat.append( (num, reachs, ends) )
        
        score_stat = np.array(score_stat)
        return score_mean, score_max, score_stat
    
    def train(self, epoch_size, lr=0.1, showsize=1000):
        start_epoch = len(self.metrics)
        for epoch in range(start_epoch, epoch_size):
            # init score and env (2048)
            score = 0.0
            game = board().popup().popup()
            records = []
            while True:
                next_games = [game.up(), game.down(), game.left(), game.right()]
                action = np.argmax(self.evaulate(next_games))
                next_game, reward = next_games[action]

                if game.end():
                    break

                next_game_after = next_game.popup()
                
                score += reward

                records.insert(0, (game.tile, action, reward, next_game.tile, next_game_after.tile) )
                game = next_game_after
                
            self.learn(records, lr)
            self.metrics.append( (score, len(records), game.tile))
            
            if (epoch+1) % showsize == 0:
                clear_output(wait=True)
                self.showStattistic(epoch+1, showsize)
            
    def play(self, game):
        next_games = [game.up(), game.down(), game.left(), game.right()]
        action = np.argmax(self.evaulate(next_games))
                
        next_game, reward = next_games[action]
        return next_game, reward, ['up', 'down', 'left', 'right'][action]


# In[42]:


MAX_NUM = 15 
TUPLE_NUM = 6 
PATTERN_NUM = 4
ACTION_NUM = 4

PATTERNS = [
    [0,1,2,3,4,5],
    [4,5,6,7,8,9],
    [0,1,2,4,5,6],
    [4,5,6,8,9,10]
]


# In[43]:


random.seed(657835)
agent = Agent(PATTERNS, MAX_NUM)


# In[45]:


def saveAgent(agent, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(agent, f)
    return fileName
    
def loadAgent(fileName):
    with open(fileName, 'rb') as f:
        agent = pickle.load(f)
    return agent


# In[46]:


get_ipython().run_cell_magic('time', '', "# agent.train(100000)\nagent = loadAgent('agent-Copy1.pkl')")


# In[47]:


agent.showStattistic(100000, 1000)


# In[48]:


def showCurve(metrics):
    metrics = np.array(metrics).reshape(len(metrics), -1)

    plt.figure(figsize=(12,4))
    plt.plot(metrics[:,0], label='score')


# In[49]:


showCurve(agent.metrics)


# In[21]:


#saveAgent(agent, 'agent.pkl')


# In[50]:


def migration(agent):
    newagent = Agent([], 0)
    newagent.Tuples = agent.Tuples
    newagent.metrics = agent.metrics
    return newagent


# In[51]:


agent = migration(agent)


# agent.showStattistic(100000, 1000)

# In[53]:


def playWithAgent(agent, step_per_seconds=0.5, show=True):
    game = board().popup().popup()
    score = 0.0
    step = 0.0
    while not game.end():
        if show:
            clear_output(wait=True)
            print('Score : {:10.0f} Step : {:10.0f}'.format(score, step))
            print(game)
        
        start = time.time()
        next_game, reward, action = agent.play(game)
        while time.time() - start < step_per_seconds:
            pass
        game = next_game.popup()
        if reward < 0.0:
            reward = 0.0
        score += reward
        step += 1.0
    
    return score, step, game.tile


# In[54]:


get_ipython().run_cell_magic('time', '', 'playWithAgent(agent, step_per_seconds=0.0, show=False)')

