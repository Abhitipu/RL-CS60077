import gym
import numpy as np
import os

env = gym.make('CartPole-v1')

attributes = ["gravity", "masscart", "masspole", "length", "force_mag"]

for attr in attributes:
    # obtain original value
    orig_value = getattr(env, attr)
    
    all_test_rewards = []
    
    new_values = np.linspace(0, 2 * orig_value, 5)
    new_values = np.round(new_values, decimals = 2)

    with open('cartpole-values.txt', 'a') as f:
        for value in new_values:
            f.write(attr + ' ' + str(value) + '\n')

with open('cartpole-values.txt', 'r') as f:
    for line in f:
        if line != "":
            attr, val = line.split(' ')
            os.system(f'python cleanrl/dqn.py --env-id CartPole-v1 --attribute {attr} --value {val}')