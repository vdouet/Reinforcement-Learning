# Code from the book "Deep Reinforcement Learning - Hands On" by Maxim Lapan

import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)

            # We do not include a softmax layer here because later on we use
            # nn.CrossEntropyLoss which combines both softmax and cross-entropy
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    observation = env.reset()
    softmax = nn.Softmax(dim=1)

    while True:
        observation_t = torch.FloatTensor([observation])
        action_prob_t = softmax(net(observation_t))
        # Replaced .data by .detach() here
        action_prob = action_prob_t.detach().numpy()[0]
        action = np.random.choice(len(action_prob), p=action_prob)
        obervation_, reward, done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=observation,
                                         action=action))

        if done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            observation_ = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []

        observation = obervation_


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_observations = []
    train_actions = []
    for episode in batch:
        if episode.reward < reward_bound:
            continue
        train_observations.extend(map(lambda step: step.observation,
                                      episode.steps))
        train_actions.extend(map(lambda step: step.action, episode.steps))

    train_observations_t = torch.FloatTensor(train_observations)
    train_actions_t = torch.LongTensor(train_actions)

    return train_observations_t, train_actions_t, reward_bound, reward_mean

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_t, acts_t, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_t = net(obs_t)
        loss_t = objective(action_scores_t, acts_t)
        loss_t.backward()
        optimizer.step()
        print("%d: loss%.3f, reward_mean=%.1f, \
               reward_bound=%.1f" % (iter_no, loss_t.item(), reward_m,
                                     reward_b))
        writer.add_scalar("loss", loss_t.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 199:
            print('Solved!')
            break
    writer.close()
