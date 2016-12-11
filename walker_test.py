import gym
env = gym.make('BipedalWalkerHardcore-v2')
env.monitor.start('./walker-experiment')
for i_episode in range(2):
    observation = env.reset()
    for t in range(1000):
        env.render()
#        print(observation)
        print env.action_space.high
        print env.action_space.low        
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
            
env.monitor.close()
gym.upload('./walker-experiment', api_key='sk_oOcEXAWRgKM6bBJjtTcTw')
