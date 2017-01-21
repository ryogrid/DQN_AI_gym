# Organism shown in the OpenAI Evaluation was found in the 58th generation
# A total of 116000 episodes were used for training (400 * 5 * 58)
# Though the highest fitness organism had a fitness score of 303 in training, it did not solve the environment at test time.



from __future__ import print_function

import gym
import numpy as np
import itertools
import os

from neat import nn, population, statistics

np.set_printoptions(threshold=np.inf)
env = gym.make('MountainCar-v0')

# run through the population


def eval_fitness(genomes):
    for g in genomes:
        observation = env.reset()
        # env.render()
        net = nn.create_feed_forward_phenotype(g)
        fitness = 0
        reward = 0
        total_fitness = 0

        for k in range(5):
            fitness = -100
            frames = 0
            while 1:
                inputs = observation

                # active neurons
                output = net.serial_activate(inputs)

                output = np.clip(output, 0, 2)
                #output = np.round(output)
                # print(output)
                observation, reward, done, info = env.step(int(np.array(output)[0]))

                if fitness < observation[1]:
                    fitness = observation[1]
                # env.render()
                frames += 1
                if done or frames > 2000:
                    total_fitness += fitness
                    # print(fitness)
                    env.reset()
                    break
        # evaluate the fitness
        g.fitness = total_fitness / 5
        print(g.fitness)

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'neat_mountain.config')

pop = population.Population(config_path)
pop.run(eval_fitness, 1000)
winner = pop.statistics.best_genome()
del pop

winningnet = nn.create_feed_forward_phenotype(winner)

env.monitor.start('./mountain-experiment', force=True)


streak = 0
episode = 0
best_reward = -9999
while streak < 100:
    fitness = 0
    reward = 0
    observation = env.reset()
    frames = 0
    while 1:
        inputs = observation

        # active neurons
        output = winningnet.serial_activate(inputs)
        output = np.clip(output, 0, 2)
        #output = np.round(output)
        
        # print(output)
        observation, reward, done, info = env.step(np.array(int(np.array(output)[0])))

        fitness += reward

        frames += 1
        if done or frames > 2000:
            if fitness >= -1900 :
                    print(fitness)
                    print ('streak: ', streak)
                    streak += 1
            else:
                print(fitness)
                print('streak: ', streak)
            break                
        
    episode += 1        
    if fitness > best_reward:
        best_reward = fitness
    print(str(episode) + " " + str(fitness) + " " + str(best_reward))                    
print("completed!")
env.monitor.close()
gym.upload('./walker-experiment', api_key='sk_oOcEXAWRgKM6bBJjtTcTw')
