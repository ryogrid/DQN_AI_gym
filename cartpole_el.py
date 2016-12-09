import gym
env = gym.make('CartPole-v0')
env.monitor.start('./cartpole-experiment-2')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
            
env.monitor.close()
gym.upload('./cartpole-experiment-2', api_key='sk_oOcEXAWRgKM6bBJjtTcTw')
