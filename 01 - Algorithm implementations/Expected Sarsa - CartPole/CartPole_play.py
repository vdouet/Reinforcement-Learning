import numpy as np
import gym


def action_argmax(Q, state):
    """Return the action with the maximum value for the current state.

    Arguments:
        Q {dict} -- state-action function array
        state {array} -- Current state

    Returns:
        int -- Action to take
    """
    values = np.array([Q[state, a] for a in range(N_ACTIONS)])
    action = np.argmax(values)
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
    Q = np.load('Q_values.npy', allow_pickle='TRUE').item()

    # Play a game of Cart Pole
    observation = env.reset()
    state = get_state(observation)
    action = action_argmax(Q, state)

    done = 0
    ep_rewards = 0

    while not done:
        observation, reward, done, info = env.step(action)
        ep_rewards += reward
        state = get_state(observation)
        action = action_argmax(Q, state)
        env.render()

    print(ep_rewards)
