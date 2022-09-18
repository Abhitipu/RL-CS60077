#!/usr/bin/env python
# coding: utf-8

# ## Q learning with the CartPole-v1 using Monte Carlo

# For a detailed description of the cartpole env, refer to [this](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) site

# In[1]:


import numpy as np
import gym
import sys
from tqdm import tqdm


# ### Configuring the display using matplotlib

# In[2]:


from IPython import display
import matplotlib
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from sklearn.preprocessing import KBinsDiscretizer
# from scipy.stats import norm
import math
from typing import Tuple
import pickle


# ### Setting seed for reproducibility

# In[4]:

np.random.seed(42)



# #### Define the policy
# <!-- $\epsilon$ / $|A|$ + 1 - $\epsilon$ --- $max_{a}Q(s, a)$ -->

# In[5]:


def policy(Q_table, state , eps = 0.05):
    """
    Epsilon greedy policy
    Choose the next action as follows
    eps / |A| + (1 - eps) --> greedy best
    eps / |A|             --> random
    """
    return np.argmax(Q_table[state]) if np.random.random() <= 1 - eps else np.random.randint(Q_table.shape[-1])


# ### The runner

# In[6]:


env_rgb = lambda rendered_list: np.array(rendered_list).squeeze()


# In[7]:


def Q_learning(environment, discretizer, Q_table, n_episodes, render = False, discount = 1, learning_rate = 0.95):
    
    prev = 0
    rewards = np.zeros(n_episodes)
    for ep in tqdm(range(n_episodes)):
        start_state = environment.reset()[0]
        # print(start_state)
        discrete_state = discretizer.discretize(*start_state)

        if render:
            img = plt.imshow(env_rgb(environment.render()))
        
        done = False
        iters = 1
        episode_reward = 0
        while not done:
            # Step 1: Choose the first action using an e-greedy policy
            action = policy(Q_table, discrete_state, 1 / (ep + 1))
            new_state, reward, done, info, _ = environment.step(action)

            if render:
                plt.title(f"Episode no: {ep+1} Iteration no: {iters}")       
                img.set_data(env_rgb(environment.render()))
                display.display(plt.gcf())
                display.clear_output(wait=True)

            # Step 2: Now we greedily choose the next best action and find the cumulative reward
            new_discrete_state = discretizer.discretize(*new_state)
            next_action = np.argmax(Q_table[new_discrete_state])
            net_return = reward + discount * Q_table[new_discrete_state][next_action]
            episode_reward += reward
            if episode_reward >= 475:
                done = True

            # Step 3: Modify Q(s, a)
            Q_table[discrete_state][action] = (1 - learning_rate) * Q_table[discrete_state][action] + learning_rate * net_return
            
            # Step 4: Update state
            discrete_state = new_discrete_state
            state = new_state
            iters += 1

        rewards[ep] = iters - 1
        if prev < iters - 1:
            prev = iters - 1
#             print(f"Episode no: {ep+1} Iterations: {iters - 1}")

    return Q_table, rewards


# ### Testing

# In[8]:


def play_episodes(environment, Q_res, n_episodes, discretizer, render = False):
    
    rewards = np.zeros(n_episodes)
    for episode in tqdm(range(n_episodes)):
        terminated = False
        state = environment.reset()[0]
        state = discretizer.discretize(*state)
        if render:
            img = plt.imshow(env_rgb(environment.render()))

        episode_reward = 0
        while not terminated:
            # Select best action to perform in a current state
            action = np.argmax(Q_res[state])
            # Perform an action an observe how environment acted in response
            next_state, reward, terminated, info, _ = environment.step(action)
            
            episode_reward += reward
            
            if episode_reward >= 475:
                break
            
            if render:
                plt.title(f"Episode no: {episode+1} Reward: {episode_reward}")       
                img.set_data(env_rgb(environment.render()))
                display.display(plt.gcf())
                display.clear_output(wait=True)

            # Update current state
            next_state = discretizer.discretize(*next_state)
            state = next_state

        rewards[episode] = episode_reward

    return rewards


# ### Description
# 
# #### Action space
# 
# The action is an `ndarray` with shape `(1,)` which can take values {0, 1} indicating the direction of the fixed force the cart is pushed with.
# 
# #### State Space
# 
# The observation is an `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:
# 
# | Num |      Observation      |         Min         |        Max        |   |
# |:---:|:---------------------:|:-------------------:|:-----------------:|---|
# | 0   | Cart Position         | -4.8                | 4.8               |   |
# | 1   | Cart Velocity         | -Inf                | Inf               |   |
# | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |   |
# | 3   | Pole Angular Velocity | -Inf                | Inf               |   |
# 
# #### Rewards
# 
# Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken, including the termination step, is allotted. The threshold for rewards is `475` for v1.
# 
# #### Starting State
# 
# All observations are assigned a uniformly random value in (-0.05, 0.05)
# 
# #### Episode
# 
# The episode ends if any one of the following occurs:
# 
# - `Termination`: Pole Angle is greater than ±12°</li>
# - `Termination`: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)</li>
# - `Truncation`: Episode length is greater than 500 (200 for v0)</li>
# 

# ### Discretizer

# In[9]:


class Discretizer:
    def __init__(self, n_bins, lower_bounds, upper_bounds):
        self.n_bins = n_bins
        self.discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        self.discretizer.fit([lower_bounds, upper_bounds])
    
    def discretize(self, position, velocity, angle, angular_velocity):
        return tuple(map(int, self.discretizer.transform([[position, velocity, angle, angular_velocity]])[0]))


# ### More utilities

# In[10]:


def analyze_and_plot(train_rewards, test_rewards, attr_name, attr_values, save_fig = False):
    attr_values = list(map(str, attr_values))
    
    plt.cla()
    plt.title(f"Rewards from training with tweaks in {attr_name}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    _ = plt.plot(np.arange(train_rewards.shape[-1]) + 1, train_rewards.T)
    plt.legend(attr_values)
    train_fig = plt.gca()
    plt.show()
    if save_fig:
        with open(f"rewards/train_rewards_{attr_name}.pkl", "wb") as f:
            pickle.dump(train_fig, f)
    plt.close()
    
    # Basically we print the correlations with the original value    
    train_correlations = np.corrcoef(train_rewards)
    print(f"Training time correlations: {train_correlations[2,:]}")
    
    mean_rewards = np.mean(test_rewards, axis=1)
    deviations = np.std(test_rewards, axis=1)
    print(f"Mean: {mean_rewards}, Standard deviation: {deviations}")
    
    plt.cla()
    _ = plt.plot(np.arange(test_rewards.shape[-1]) + 1, test_rewards.T)
    plt.title(f"Rewards from testing with tweaks in {attr_name}")
    plt.ylabel("Reward")
    plt.xlabel("New value")
    test_fig = plt.gca()
    plt.show()
    if save_fig:
        with open(f"rewards/test_rewards_{attr_name}.pkl", "wb") as f:
            pickle.dump(test_fig, f)
    plt.close()    


# In[11]:


def run_model(env, train_episodes, test_episodes, save_data = False):
    print("Training model")
    Q_table = np.zeros((*N_BINS, env.action_space.n))
    Q_result, train_rewards = Q_learning(env, discretizer, Q_table, train_episodes, render=False)
    
    print("Testing model")
    test_rewards = play_episodes(env, Q_result, test_episodes, discretizer)
    print(f'Average reward over {test_episodes} episodes = {np.mean(test_rewards)} \n\n')
    
    return train_rewards, test_rewards


# ### The setup

# In[12]:


N_BINS = (6, 2, 12, 12)

environment = gym.make('CartPole-v1', render_mode="rgb_array")#, new_step_api=True)
low_vals = environment.observation_space.low
high_vals = environment.observation_space.high

lower_bounds = [ low_vals[0], -3.5, low_vals[2], -3.5 ]
upper_bounds = [ high_vals[0], 3.5, high_vals[2], 3.5 ]

discretizer = Discretizer(N_BINS, lower_bounds, upper_bounds)


# ### Configure

# |       Parameter       |  Value  |               Info              |   |   |
# |:---------------------:|:-------:|:-------------------------------:|:-:|---|
# |        gravity        |   9.8   |                                 |   |   |
# |        masscart       |    1    |                                 |   |   |
# |        masspole       |   0.1   |                                 |   |   |
# |       total_mass      |   1.1   |       masspole + masscart       |   |   |
# |         length        |   0.5   | actually half the pole's length |   |   |
# |    polemass_length    |   0.05  |        masspole * length        |   |   |
# |       force_mag       |    10   |                                 |   |   |
# |          tau          |   0.02  |  seconds between state updates  |   |   |
# | kinematics_integrator | "euler" |                                 |   |   |

# In[ ]:


N_TRAIN_EPISODES = 50000
N_TEST_EPISODES = 1000

PLOT = True
SAVE = True

attributes = ["gravity", "masscart", "masspole", "length", "force_mag"]

for attr in attributes:
    print(f"Tweaking {attr}")
    print(f"Tweaking {attr}", file=sys.stderr)
    
    # obtain original value
    orig_value = getattr(environment, attr)
    
    all_train_rewards = np.zeros((5, N_TRAIN_EPISODES))
    all_test_rewards = np.zeros((5, N_TEST_EPISODES))
    
    new_values = np.linspace(0, 2 * orig_value, 5)
    new_values = np.round(new_values, decimals = 2)
    
    for idx, value in enumerate(new_values):
        setattr(environment, attr, value)
        train_rewards, test_rewards = run_model(environment, N_TRAIN_EPISODES, N_TEST_EPISODES, save_data = True) 
        print(f"Mean {np.mean(test_rewards)} and deviation {np.std(test_rewards)}")
        all_train_rewards[idx] = train_rewards
        all_test_rewards[idx] = test_rewards
    
    if PLOT:
        analyze_and_plot(all_train_rewards, all_test_rewards, attr, new_values, save_fig = SAVE)
        
    # reset to original value
    setattr(environment, attr, orig_value)
