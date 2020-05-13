# Original code from "Reinforcement Learning in Motion" by Phil Tabor.

import numpy as np
import matplotlib.pyplot as plt
import gym

# Gym - CartPole
#
# State space:
# Cart position -> -2.4 < x < 2.4
# Cart velocity -> -inf < dx/dt < +inf
# Pole angle (rad) -> -0.73 < Θ < +0.73
# Pole velocity at the tip -> -inf < dx/dt < +inf
#
# Action space:
# Push left (0)
# Push right (1)


def action_argmax_q1q2(Q1, Q2, state):
    """Return the action with the maximum value for the current state using
        Q1 + Q2.

    Arguments:
        Q1 {dict} -- state-action function dict Q1
        Q2 {dict} -- state-action function dict Q2
        state {array} -- Current state

    Returns:
        int -- Action to take
    """
    values = np.array([Q1[state, a] + Q2[state, a] for a in range(N_ACTIONS)])
    # Random tie breaking
    action = np.random.choice(np.where(values == values.max())[0])
    return action


def action_argmax(Q, state):
    """Return the action with the maximum value for the current state.

    Arguments:
        Q {dict} -- state-action function dict
        state {array} -- Current state

    Returns:
        int -- Action to take
    """
    values = np.array([Q[state, a] for a in range(N_ACTIONS)])
    # Random tie breaking
    action = np.random.choice(np.where(values == values.max())[0])
    return action


# Discretize the state spaces
pole_theta_space = np.linspace(-0.209, 0.209, 10)
pole_theta_vel_space = np.linspace(-4, 4, 10)
cart_pos_space = np.linspace(-2.4, 2.4, 10)
cart_vel_space = np.linspace(-4, 4, 10)


def get_state(observation):
    """Return the state based on the observation from the env

    Arguments:
        observation {array} -- states from the env.

    Returns:
        array -- current state
    """
    cart_x, cart_vel, pole_theta, pole_vel = observation
    cart_x = int(np.digitize(cart_x, cart_pos_space))
    cart_vel = int(np.digitize(cart_vel, cart_vel_space))
    pole_theta = int(np.digitize(pole_theta, pole_theta_space))
    pole_vel = int(np.digitize(pole_vel, pole_theta_vel_space))

    return (cart_x, cart_vel, pole_theta, pole_vel)


def plot_running_avg(total_rewards):
    """Plot the running average of rewards

    Arguments:
        total_rewards {array} -- Array of reward for each episodes
    """
    n = len(total_rewards)
    running_avg = np.empty(n)
    for t in range(n):
        running_avg[t] = np.mean(total_rewards[max(0, t - 100):(t + 1)])
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    ALPHA = 0.1
    GAMMA = 1.0
    EPS = 1.0
    N_ACTIONS = env.action_space.n

    # Construct the state space
    states = []
    for i in range(len(cart_pos_space) + 1):
        for j in range(len(cart_vel_space) + 1):
            for k in range(len(pole_theta_space) + 1):
                for l in range(len(pole_theta_vel_space) + 1):
                    states.append((i, j, k, l))

    # initialise Q(s,a) to 0
    Q1, Q2 = {}, {}
    for state in states:
        for action in range(N_ACTIONS):
            Q1[state, action] = 0
            Q2[state, action] = 0

    number_games = 25000
    total_rewards = np.zeros(number_games)

    for i in range(number_games):

        if i % 5000 == 0:
            print('Starting game', i)

        observation = env.reset()
        state = get_state(observation)

        done = False
        ep_rewards = 0

        while not done:

            # e-greedy action selection
            rand = np.random.random()
            random_action = env.action_space.sample()
            action = action_argmax_q1q2(
                Q1, Q2, state) if rand < (1 - EPS) else random_action

            observation_, reward, done, info = env.step(action)
            ep_rewards += reward
            state_ = get_state(observation_)
            rand = np.random.random()

            # Update Q1 or Q2 based on a 50% probability
            if rand <= 0.5:
                action_ = action_argmax(Q1, state_)
                Q1[state, action] = Q1[state, action] + ALPHA * (
                    reward + GAMMA * Q2[state_, action_] - Q1[state, action])
            elif rand > 0.5:
                action_ = action_argmax(Q2, state_)
                Q2[state, action] = Q2[state, action] + ALPHA * (
                    reward + GAMMA * Q1[state_, action_] - Q2[state, action])
            state = state_

        # At the end of the episode decrease epsilon by a small amount such as
        # it converges to a greedy strategy through the series of ep.
        if EPS - 1.5 / number_games > 0:
            EPS -= 1.5 / number_games
        else:
            EPS = 0

        total_rewards[i] = ep_rewards

    # Save the Q dictionnary to be played back.
    np.save('Q1_values.npy', Q1)
    np.save('Q2_values.npy', Q2)

    # Print the running average.
    plot_running_avg(total_rewards)
