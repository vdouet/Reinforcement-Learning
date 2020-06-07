# Code from the course: Advanced AI: Deep Reinforcement Learning in Python
# by the Lazyprogrammer

import gym
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor


class FeatureTransformer:
    def __init__(self, env):
        # Take sample from the state space
        observation_examples = np.array(
            [env.observation_space.sample() for x in range(10000)])
        # Standardize the observation. Mean = 0, variance = 1.
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        # Used to convert a state to a featurizes representation
        # We use RBF kernels with differents variances to cover different parts
        # of the space.
        # The number of components is the number of exemplars.
        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=500)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=500)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=500)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=500))
        ])
        example_features = featurizer.fit_transform(
            scaler.transform(observation_examples))

        self.dimensions = example_features.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)


class Model:
    def __init__(self, env, feature_transformer, learning_rate):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer

        # One model for each action
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate=learning_rate)
            # First partial fit with dummy data because of how it works
            # Also enables optimistic initial values for exploration
            model.partial_fit(feature_transformer.transform([env.reset()]),
                              [0])
            self.models.append(model)

    def predict(self, s):
        '''
        Transform the state into a feature vector and makes a prediction of
        values, one for each action
        '''
        # We put s into a list because sklearn requires a 2D input.
        X = self.feature_transformer.transform([s])
        result = np.stack([m.predict(X) for m in self.models]).T
        assert (len(result.shape) == 2)
        return result

    def update(self, s, a, G):
        X = self.feature_transformer.transform([s])
        assert (len(X.shape) == 2)
        # We put G into a list because sklearn require target to be 1D object
        # whereas G is just a scalar
        self.models[a].partial_fit(X, [G])

    def sample_action(self, s, eps):
        '''
        Technically, we don't need to do epsilon-greedy because SGDRegressor
        predicts 0 for all states until they are updated. This works as the
        "Optimistic Initial Values" method since all the rewards for Mountain
        Car are -1.
        '''
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))


# Returns a list of states and rewards, and the total reward
def play_one(model, eps, gamma):
    observation = env.reset()
    done = False
    total_reward = 0

    while not done:
        # e-greedy action selection
        action = model.sample_action(observation, eps)
        observation_, reward, done, _ = env.step(action)

        # Update the model
        G = reward + gamma * np.max(model.predict(observation_)[0])
        model.update(observation, action, G)
        observation = observation_

        total_reward += reward

    return total_reward


def plot_cost_to_go(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0],
                    env.observation_space.high[0],
                    num=num_tiles)
    y = np.linspace(env.observation_space.low[1],
                    env.observation_space.high[1],
                    num=num_tiles)
    X, Y = np.meshgrid(x, y)
    # both X and Y will be of shape (num_tiles, num_tiles)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2,
                            np.dstack([X, Y]))
    # Z will also be of shape (num_tiles, num_tiles)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X,
                           Y,
                           Z,
                           rstride=1,
                           cstride=1,
                           cmap=matplotlib.cm.coolwarm,
                           vmin=-1.0,
                           vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-To-Go == -V(s)')
    ax.set_title("Cost-To-Go Function")
    fig.colorbar(surf)
    plt.show()


def plot_running_avg(total_reward):
    N = len(total_reward)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = total_reward[max(0, t - 100):(t + 1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft, "constant")
    gamma = 0.99

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 300
    total_rewards = np.empty(N)
    for n in range(N):
        eps = 0.1 * (0.97**n)
        total_reward = play_one(model, eps, gamma)
        total_rewards[n] = total_reward
        print("Episode: ", n, "Total reward: ", total_reward)
    print("Avg reward for last 100 episodes: ", total_rewards[-100:].mean())
    print("Total steps: ", -total_rewards.sum())

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(total_rewards)

    # Plot the optimal state-value function
    plot_cost_to_go(env, model)
