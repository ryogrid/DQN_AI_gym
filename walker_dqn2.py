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

STATE_NUM = 1

# DQN内部で使われるニューラルネット
class Q(Chain):
    def __init__(self,state_num=STATE_NUM):
        super(Q,self).__init__(
             l1=L.Linear(state_num, 16),  # stateがインプット
             l2=L.Linear(16, 32),
             l3=L.Linear(32, 64),
             l4=L.Linear(64, 2*2*2*2),            
#             l5=L.Linear(128, 2*2*2*2), # 出力2チャネル(Qvalue)がアウトプット            
        )

    def __call__(self,x,t):
        return F.mean_squared_error(self.predict(x,train=True),t)

    def  predict(self,x,train=False):
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.leaky_relu(self.l2(h1))
        h3 = F.leaky_relu(self.l3(h2))
#        h4 =  F.leaky_relu(self.l4(h3))
        y = F.leaky_relu(self.l4(h3))
        return y

# DQNアルゴリズムにしたがって動作するエージェント
class DQNAgent():
    def __init__(self, epsilon=0.99):
        self.model = Q()
        self.optimizer = optimizers.Adam(alpha=0.001)
        self.optimizer.setup(self.model)
        self.epsilon = epsilon # ランダムアクションを選ぶ確率
        self.actions=[-1,1] #　行動の選択肢
        self.experienceMemory = [] # 経験メモリ
        self.memSize = 300*100  # 経験メモリのサイズ(300サンプリングx100エピソード)
        self.experienceMemory_local=[] # 経験メモリ（エピソードローカル）
        self.memPos = 0 #メモリのインデックス
        self.batch_num = 32 # 学習に使うバッチサイズ
        self.gamma = 0.9       # 割引率
        self.loss=0
        self.total_reward_award=np.ones(100)*-1000 #100エピソード
        self.random_exp = 128
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
        # seq後の行動価値を返す
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
        '''
        seq (theta, old_theta)に対して
        アクション（左に動くか右に動くか）を返す。
        '''
        self.epsilon = min(1.0, 0.02 + self.random_exp / (self.idx + 1.0))
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
        #エピソードローカルな記憶
        self.experienceMemory_local.append( np.hstack([old_seq,action,reward,new_seq]) )

    def experience_global(self,total_reward):
        #グローバルな記憶
        #ベスト100に入る経験を取り込む
        if np.min(self.total_reward_award)<total_reward:
            i=np.argmin(self.total_reward_award)
            self.total_reward_award[i]=total_reward

            # GOOD EXPERIENCE REPLAY
            for x in self.experienceMemory_local:
                self.experience( x )

        #一定確率で優秀でないものも取り込む
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
        '''
        モデルを更新する
        '''
        # 経験メモリにたまってない場合は更新しない
        if len(self.experienceMemory)<self.batch_num:
            return

        # 経験メモリからバッチを作成
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

        # ネットの更新
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
        
# 環境とエージェントを渡すとシミュレーションするシミュレータ。
# ここにシーケンスを持たせるのはなんか変な気もするけどまあいいか。。
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
        self.seq = state

    def run(self, train=True):

        self.env.reset()

        self.reset_seq()
        total_reward=0

        for i in range(10000):
            # 現在のstateからなるシーケンスを保存
            old_seq = self.seq.copy()

            # エージェントの行動を決める
            action = self.agent.get_action(old_seq,train)

            # 環境に行動を入力する
            observation, reward, done, info =  self.env.step(action)
            total_reward +=reward

            # 結果を観測してstateとシーケンスを更新する
            state = np.array([observation[0]])
            self.push_seq(state)
            new_seq = self.seq.copy()

            # エピソードローカルなメモリに記憶する
            action_idx = self.agent.list_to_index(action)
            self.agent.experience_local(old_seq, action_idx, reward, new_seq)

            self.agent.idx += 1
            if done:
                print("Episode finished after {} timesteps".format(i+1))
                break
        
        # エピソードローカルなメモリ内容をグローバルなメモリに移す
        self.agent.experience_global(total_reward)

        if train:
            # 学習用メモリを使ってモデルを更新する
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
