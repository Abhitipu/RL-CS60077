import pickle
import numpy as np
import gym
import matplotlib.pyplot as plt

attributes = ["gravity", "masscart", "masspole", "length", "force_mag"]
WINDOW_SIZE = 2000
N_TRAIN_EPISODES = 100000

X = np.arange(0, N_TRAIN_EPISODES, WINDOW_SIZE)
environment = gym.make("CartPole-v1", render_mode="rgb_array")

def save_train(attr, train_rewards, new_values, X):
    plt.cla()
    plt.title(f"Rewards from training with tweaks in {attr}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    _ = plt.plot(X, np.array(train_rewards).T)
    _ = plt.legend(new_values)
    plt.savefig(f"graphs/train_rewards_{attr}.jpg")
    plt.close()
    plt.show()

def save_test(attr, new_values, mu):
    plt.cla()
    plt.title(f"Mean rewards from testing with tweaks in {attr}")
    plt.xlabel(f"New values of {attr}")
    plt.ylabel("Mean reward")
    _ = plt.bar(new_values, mu)
    plt.savefig(f"graphs/test_rewards_{attr}.jpg")
    plt.show()
    plt.close()


for attr in attributes:
    with open(f"rewards/train_rewards_{attr}.pkl", "rb") as f:
        axes = pickle.load(f)

    orig_value = getattr(environment, attr)
    new_values = np.linspace(0, 2 * orig_value, 5)
    new_values = np.round(new_values, decimals = 2)
    new_values = list(map(str, new_values))
    
    train_rewards = []
    
    for line in axes.lines:
        my_rewards = np.array(line.get_ydata())
        train_rewards.append(np.convolve(my_rewards, np.ones(WINDOW_SIZE))[X] / WINDOW_SIZE)
        
    save_train(attr, train_rewards, new_values, X)   

    with open(f"rewards/test_rewards_{attr}.pkl", "rb") as f:
        axes = pickle.load(f)
    
    test_rewards = []
    for line in axes.lines:
        my_rewards = np.array(line.get_ydata())
        test_rewards.append(my_rewards)
    test_rewards = np.array(test_rewards)
    mu = np.mean(test_rewards, axis=1)

    save_test(attr, new_values, mu)
    # plt.show()

