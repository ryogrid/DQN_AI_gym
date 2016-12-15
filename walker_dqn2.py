# coding: utf-8
import numpy as np
import time

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import gym

np.random.seed(7)

STATE_NUM = 6

class Q(Chain):
    def __init__(self,state_num=STATE_NUM):
        super(Q,self).__init__(
             l1=L.Linear(state_num, 16),
             l2=L.Linear(16, 32),
             l3=L.Linear(32, 64),
             l4=L.Linear(64, 128),
             l5=L.Linear(128, 2*2*2*2),            
        )

    def __call__(self,x,t):
        return F.mean_squared_error(self.predict(x,train=True),t)

    def  predict(self,x,train=False):
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.leaky_relu(self.l2(h1))
        h3 = F.leaky_relu(self.l3(h2))
        h4 =  F.leaky_relu(self.l4(h3))
        y = F.leaky_relu(self.l5(h4))
        return y

class DQNAgent():
    def __init__(self, epsilon=0.99):
        self.model = Q()
        self.optimizer = optimizers.Adam(alpha=0.001)
        self.optimizer.setup(self.model)
        self.epsilon = epsilon
        self.actions=[-1,1]
        self.experienceMemory = []
        self.memSize = 300*100
        self.experienceMemory_local=[]
        self.memPos = 0
        self.batch_num = 32
        self.gamma = 0.9
        self.loss=0
        self.total_reward_award=np.ones(100)*-1000
#        self.random_exp = 128
        self.idx = 0

    def index_to_list(self, index):
        ret_arr = []
        a = int(index / 8)
        if a == 0:
            a = -1
        rest = index - 8*int(index / 8)
        ret_arr.append(a)
        a = int(rest / 4)
        if a == 0:
            a = -1
        rest = rest - 4*int(rest / 4)
        ret_arr.append(a)
        a = int(rest / 2)
        if a == 0:
            a = -1
        rest = rest - 2*int(rest / 2)
        ret_arr.append(a)
        if rest == 0:
            rest = -1
        ret_arr.append(rest)
        
        return ret_arr

    def list_to_index(self, lst):
        ret = 0

        a = lst[0]
        if a == -1:
            a = 0
        ret += a*8
        a = lst[1]
        if a == -1:
            a = 0        
        ret += a*4
        a = lst[2]
        if a == -1:
            a = 0        
        ret += a*2
        a = lst[3]
        if a == -1:
            a = 0        
        ret += a
        
        return ret
    
    def get_action_value(self, seq):
        x = Variable(np.hstack([seq]).astype(np.float32).reshape((1,-1)))
        return self.model.predict(x).data[0]

    def get_greedy_action(self, seq):
        action_index = np.argmax(self.get_action_value(seq))
        return self.index_to_list(action_index)

    def reduce_epsilon(self):
        self.epsilon-=1.0/10000

    def get_epsilon(self):
        return self.epsilon

    def get_action(self,seq,train):
#        self.epsilon = min(1.0, 0.02 + self.random_exp / (self.idx + 1.0))
        action=[]
        if train==True and np.random.random()<self.epsilon:
            # random
            action.append(np.random.choice(self.actions))
            action.append(np.random.choice(self.actions))
            action.append(np.random.choice(self.actions))
            action.append(np.random.choice(self.actions))
        else:
            # greedy
            action= self.get_greedy_action(seq)
        return action

    def experience_local(self,old_seq, action, reward, new_seq):
        self.experienceMemory_local.append( np.hstack([old_seq,action,reward,new_seq]) )

    def experience_global(self,total_reward):
        if np.min(self.total_reward_award)<total_reward:
            i=np.argmin(self.total_reward_award)
            self.total_reward_award[i]=total_reward

            # GOOD EXPERIENCE REPLAY
            for x in self.experienceMemory_local:
                self.experience( x )

        if np.random.random()<0.01:
            # # NORMAL EXPERIENCE REPLAY
            for x in self.experienceMemory_local:
                self.experience( x )

        self.experienceMemory_local=[]

    def experience(self,x):
        if len(self.experienceMemory)>self.memSize:
            self.experienceMemory[int(self.memPos%self.memSize)]=x
            self.memPos+=1
        else:
            self.experienceMemory.append( x )

    def update_model(self,old_seq, action, reward, new_seq):
        if len(self.experienceMemory)<self.batch_num:
            return

        memsize=len(self.experienceMemory)
        batch_index = list(np.random.randint(0,memsize,(self.batch_num)))
        batch =np.array( [self.experienceMemory[i] for i in batch_index ])
        x = Variable(batch[:,0:STATE_NUM].reshape( (self.batch_num,-1)).astype(np.float32))
        targets=self.model.predict(x).data.copy()

        for i in range(self.batch_num):
            #[ seq..., action, reward, seq_new]
            a = batch[i,STATE_NUM]
            r = batch[i, STATE_NUM+1]
            ai=a
            new_seq= batch[i,(STATE_NUM+2):(STATE_NUM*2+2)]
            targets[i,ai]=( r+ self.gamma * np.max(self.get_action_value(new_seq)))
        t = Variable(np.array(targets).reshape((self.batch_num,-1)).astype(np.float32)) 

        self.model.zerograds()
        loss=self.model(x ,t)
        self.loss = loss.data
        loss.backward()
        self.optimizer.update()

class walkerEnvironment():
    def __init__(self):
        self.env = gym.make('BipedalWalkerHardcore-v2')
        self.env.monitor.start('./walker-experiment')

    def reset(self):
        self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def monitor_close(self):
        self.env.monitor.close()
        
class simulator:
    def __init__(self, environment, agent):
        self.agent = agent
        self.env = environment

        self.num_seq=STATE_NUM
        self.reset_seq()
        self.learning_rate=1.0
        self.highscore=0
        self.log=[]

    def reset_seq(self):
        self.seq=np.zeros(self.num_seq)

    def push_seq(self, state):
        self.seq[1:self.num_seq]=self.seq[0:self.num_seq-1]
        self.seq[0]=state

    def run(self, train=True):

        self.env.reset()

        self.reset_seq()
        total_reward=0

        for i in range(10000):
            old_seq = self.seq.copy()

            action = self.agent.get_action(old_seq,train)

            observation, reward, done, info =  self.env.step(action)
            total_reward +=reward

            state = observation[0]
            self.push_seq(state)
            new_seq = self.seq.copy()

            action_idx = self.agent.list_to_index(action)
            self.agent.experience_local(old_seq, action_idx, reward, new_seq)

            self.agent.idx += 1
            if done:
                print("Episode finished after {} timesteps".format(i+1))
                break
        
        self.agent.experience_global(total_reward)

        if train:
            action_idx = self.agent.list_to_index(action)
            self.agent.update_model(old_seq, action_idx, reward, new_seq)
            self.agent.reduce_epsilon()

        return total_reward

if __name__ == '__main__':
    agent=DQNAgent()
    env=walkerEnvironment()
    sim=simulator(env,agent)

    best_reword = -200
    for i in range(10000):
        total_reword = sim.run(train=True)
        if best_reword < total_reword:
            best_reword = total_reword

        print(str(i) + " " + str(total_reword) + " " + str(best_reword))            
        env.reset()

        if best_reword > 200:
            break

    env.monitor_close()
    gym.upload('./walker-experiment', api_key='sk_oOcEXAWRgKM6bBJjtTcTw')
