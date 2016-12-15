import gym
import random

class AgentQL:
    def __init__(self, e=0.99,alpha=0.3):
        self.q={} #set of s,a
        self.epsilon=e
        self.alpha=alpha
        self.gamma=0.9
        self.last_act=None
        self.last_state=None

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

    def conv_to_int_state(self, state):
        return int(state * 10)
    
    def reduce_epsilon(self):
        self.epsilon-=1.0/1000

    def getQ(self, state, act):
        # encourage exploration; "optimistic" 1.0 initial values
        if self.q.get((state, act)) is None:
            self.q[(state, act)] = 1
        return self.q.get((state, act))
    
    def get_action(self, state):
        self.last_state=state
        
        #Explore sometimes
        if random.random() < self.epsilon:
            act_num = random.randint(0,15)
            return self.index_to_list(act_num)

        qs = [self.getQ(self.last_state, a) for a in xrange(16)]
        maxQ= max(qs)

        if qs.count(maxQ) > 1:
            # more than 1 best option; choose among them randomly
            best_options = [i for i in range(len(qs)) if qs[i] == maxQ]
            act_num = random.choice(best_options)
        else:
            act_num = qs.index(maxQ)

        self.last_act = act_num
        
        return self.index_to_list(act_num)

    def learn(self,s,r):
        pQ=self.getQ(s,self.last_act)
        
        maxQnew=max([self.getQ(s,a) for a in xrange(16)])
        self.q[(s,a)]=pQ+self.alpha*((r+self.gamma*maxQnew)-pQ)



if __name__ == '__main__':
    env = gym.make('BipedalWalkerHardcore-v2')
    env.monitor.start('./walker-experiment')
    agent=AgentQL()
    
    best_reward = -200
    for i in range(1000):
        total_reward = 0
        observation = env.reset()

        for i in range(2100):
            state = agent.conv_to_int_state(observation[0])
            
            action = agent.get_action(state)

            observation, reward, done, info =  env.step(action)
            total_reward +=reward


            if done:
                print("Episode finished after {} timesteps".format(i+1))
                break
        
            agent.learn(state, reward)
            agent.reduce_epsilon()
        
        if best_reward < total_reward:
            best_reward = total_reward

        print(str(i) + " " + str(total_reward) + " " + str(best_reward))            
        env.reset()

        if best_reward > 200:
            break

    env.monitor_close()
    gym.upload('./walker-experiment', api_key='sk_oOcEXAWRgKM6bBJjtTcTw')
