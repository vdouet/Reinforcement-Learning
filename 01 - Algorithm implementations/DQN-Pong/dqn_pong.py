from lib import wrappers
from lib import dqn_model

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.0

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 10000
LEARNING_RATE = 1e-4
# How frequently we sync model weights from the training model to the target
# model which is used to get the value of the next state in the Bellman approx.
SYNC_TARGET_FRAMES = 1000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
'''
We define our experience replay buffer to keep the transitions obtained from
the environment each time we take a step. For training we randomly sample the
batch of transitions from the replay buffer, which allows us to break the
correlation between subsequent steps in the environment.
'''

Experience = collections.namedtuple(
    'Experience',
    field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in indices])

        return np.array(states), np.array(actions), \
            np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.uint8), \
            np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_t = torch.tensor(state_a).to(device)
            q_vals_t = net(state_t)
            _, act_t = torch.max(q_vals_t, dim=1)
            action = int(act_t.item())

        state_, reward, done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, done, state_)
        self.exp_buffer.append(exp)
        self.state = state_

        if done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu"):
    '''
    The first model (net) is used to calculate gradients; the second model
    (tgt_net) is used to calculate values for the next states, and this
    calculation shouldn't affect gradients.
    '''

    states, actions, rewards, dones, states_ = batch

    states_t = torch.tensor(np.array(states, copy=False)).to(device)
    states_t_ = torch.tensor(np.array(states_, copy=False)).to(device)
    actions_t = torch.LongTensor(actions).to(device)
    rewards_t = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    # We pass observations to the first model and extract the specific Q-values
    # for the taken actions.
    state_action_values = net(states_t).gather(
        1, actions_t.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_state_values = tgt_net(states_t_).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values.detach()

    expected_state_action_value = GAMMA * next_state_values + rewards_t
    return nn.MSELoss()(state_action_values, expected_state_action_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda",
                        default=False,
                        action="store_true",
                        help="Enable cuda")
    parser.add_argument("--env",
                        default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" +
                        DEFAULT_ENV_NAME)

    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = wrappers.make_env(args.env)
    net = dqn_model.DQN(env.observation_space.shape,
                        env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape,
                            env.action_space.n).to(device)

    writer = SummaryWriter(comment="-" + args.env)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL,
                      EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, reward %.3f, "
                  "eps %.2f, speed %.2f f/s" %
                  (frame_idx, len(total_rewards), m_reward, epsilon, speed))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(),
                           args.env + "-best_%.0f.dat" % m_reward)
                if best_m_reward is not None:
                    print("Best reward updated %.3f -> %.3f" %
                          (best_m_reward, m_reward))
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
    writer.close()
