# Original code by Machine Learning with Phil
# https://www.youtube.com/watch?v=e8ofon3sg8E&t=608s

import numpy as np
import matplotlib.pyplot as plt
import gym

if __name__ == '__main__':

    env = gym.make('Blackjack-v0')
    EPS = 0.05  # Espsilon greedy
    GAMMA = 1.0  # Undiscounted task

    Q = {}  # Agent estimate of futur reward
    agent_total_space = [i for i in range(4, 22)]
    dealer_total_space = [i + 1 for i in range(10)]  # Only one card showed
    agent_ace_space = [False, True]
    action_space = [0, 1]  # Stick or hit

    np.random.rand()
    state_space = []
    returns = {}
    visited_pairs = {}  # State-action pairs visited

    # Construct and initiate tuples and list
    for agent_total in agent_total_space:
        for dealer_card in dealer_total_space:
            for ace in agent_ace_space:
                for action in action_space:
                    Q[((agent_total, dealer_card, ace), action)] = 0  # q(s,a)
                    returns[((agent_total, dealer_card, ace), action)] = 0
                    visited_pairs[((agent_total, dealer_card, ace),
                                   action)] = 0
                state_space.append((agent_total, dealer_card, ace))

    # Create a random equiprobable policy
    policy = {}
    for state in state_space:
        policy[state] = np.random.choice(action_space)

    # TRAINING PHASE #
    num_episodes = 1000000
    for i in range(num_episodes):

        # Keep track of states, actions and returns and agent memory
        states_actions_returns = []
        memory = []

        if i % 100000 == 0:
            print('Starting episode', i)

        # Reset env and done flag at the start of each episodes
        # Observation containes (agent_total, dealer_card, ace)
        observation = env.reset()
        done = False

        while not done:
            # Give an action for a state
            action = policy[observation]
            # Take a step and register new env information
            observation_, reward, done, info = env.step(action)
            memory.append((observation[0], observation[1], observation[2],
                           action, reward))
            observation = observation_
        memory.append(
            (observation[0], observation[1], observation[2], action, reward))

        G = 0
        last = True  # We ignore t = T (except for the reward)
        # We start from T-1, T-2, ..., 0
        for player_total, dealer_card, usable_ace, action, reward in \
                reversed(memory):

            if last:
                last = False
            else:
                states_actions_returns.append(
                    (player_total, dealer_card, usable_ace, action, G))
            G = GAMMA * G + reward

        # Put the list in chronological order
        states_actions_returns.reverse()
        states_actions_visited = []

        for player_total, dealer_card, usable_ace, action, G in \
                states_actions_returns:

            state_action = ((player_total, dealer_card, usable_ace), action)
            if state_action not in states_actions_visited:
                visited_pairs[state_action] += 1  # We use first visit here
                # Incremental implementation:
                # New_estimate = old_estimate + 1/N * [sample - old_estimate]
                # Equivalent to New_estimate += 1/N * [sample - old_estimate]
                returns[(state_action)] += (1/visited_pairs[(state_action)]) \
                                            * (G - returns[(state_action)])
                Q[state_action] = returns[(state_action)]
                rand = np.random.random()
                if rand < 1 - EPS:
                    state = (player_total, dealer_card, usable_ace)
                    values = np.array([Q[state, a] for a in action_space])
                    best = np.random.choice(
                        np.where(values == values.max())[0])
                    policy[state] = action_space[best]
                else:
                    policy[state] = np.random.choice(action_space)

                states_actions_visited.append(state_action)

        # Decrement epsilon over time, until fully greedy
        if EPS - 1e-7 > 0:
            EPS -= 1e-7
        else:
            EPS = 0

    # TESTING PHASE #
    num_episodes = 1000
    rewards = np.zeros(num_episodes)
    total_rewards = 0
    wins = 0
    losses = 0
    draws = 0
    print('Testing policy')
    for i in range(num_episodes):
        observation = env.reset()
        done = False

        while not done:
            action = policy[observation]
            observation_, reward, done, info = env.step(action)
            observation = observation_
        total_rewards += reward
        rewards[i] = total_rewards

        if reward >= 1:
            wins += 1
        elif reward == 0:
            draws += 1
        elif reward == -1:
            losses += 1

    wins /= num_episodes
    losses /= num_episodes
    draws /= num_episodes
    print('win rate', wins, 'loss rate', losses, 'draw rate', draws)
    plt.plot(rewards)
    plt.show()
