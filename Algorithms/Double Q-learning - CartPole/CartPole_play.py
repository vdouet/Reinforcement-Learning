import numpy as np
import gym


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


if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    N_ACTIONS = env.action_space.n

    # Load the trained Q function
    Q1 = np.load('Q1_values.npy', allow_pickle='TRUE').item()
    Q2 = np.load('Q2_values.npy', allow_pickle='TRUE').item()

    # Play a game of Cart Pole
    observation = env.reset()
    state = get_state(observation)
    action = action_argmax_q1q2(Q1, Q2, state)

    done = 0
    ep_rewards = 0

    while not done:
        observation, reward, done, info = env.step(action)
        ep_rewards += reward
        state = get_state(observation)
        action = action_argmax_q1q2(Q1, Q2, state)
        env.render()

    print(ep_rewards)
