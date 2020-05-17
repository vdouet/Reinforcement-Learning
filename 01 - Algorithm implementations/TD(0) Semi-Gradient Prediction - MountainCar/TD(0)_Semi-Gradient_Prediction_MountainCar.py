# From the course "Reinforcement Learning in Motion" by Phil Tabor.

import matplotlib.pyplot as plt
import numpy as np
import argparse
import gym
"""
We will use asymmetric tile coding for the Moutain Car problem here.
- 2x2 tiles with 2 layers
- Rectangular tiles with 0.5 offset in position and 1.5 in velocity

Cart position:      -1.2 < x < 0.5
Cart velocity:      -0.7 < v < 0.7
Initial position:   -0.6 < x0 < 0.4

if x = -1.2 and v < 0 -> v = 0
xt+1 = xt + vt+1
vt+1 = vt + 0.001 * (action - 1) - 0.0025 * cos(3xt)
Actions: Push left (0), push right (2), no push (1)
"""

BACKWARD = 0
FORWARD = 2


def get_bins(n_bins=8, n_layers=8):

    # Construct the asymmetric bins
    tiling_offset = 3
    pos_tile_width = (0.5 + 1.2) / n_bins * 0.5
    vel_tile_width = (0.07 + 0.07) / n_bins * 0.5
    pos_bins = np.zeros((n_layers, n_bins))  # x-axis
    vel_bins = np.zeros((n_layers, n_bins))  # y-axis
    for i in range(n_layers):
        pos_bins[i] = np.linspace(-1.2 + i * pos_tile_width,
                                  0.5 + i * pos_tile_width / 2, n_bins)
        vel_bins[i] = np.linspace(
            -0.07 + tiling_offset * i * vel_tile_width,
            0.07 + tiling_offset * i * pos_tile_width / 2, n_bins)
    return pos_bins, vel_bins


def tile_state(pos_bins, vel_bins, obs, n_bins=8, n_layers=8):
    """Function to do Tile Coding. Return a binary representation of the
    current velocity and position based on the position bins and velocity bins.

    Arguments:
        pos_bins {array} -- Bins for the position values.
        vel_bins {array} -- Bins for the velocity values.
        obs {list} -- Observation for the current state returned by the env.

    Keyword Arguments:
        n_bins {int} -- Number of bins for Tile Coding. (default: {8})
        n_layers {int} -- Number of layers for Tile Coding. (default: {8})

    Returns:
        array -- Return the binary representation of the current state.
    """

    position, velocity = obs

    # The number of tiles per axis is the number of bins per axis - 1
    n_tiles = n_bins - 1

    # The tiles here should be:
    # tiled_state = np.zeros((n_bins - 1) * (n_bins - 1) * n_layers)
    # Because we have 8 bins per axis = 7 tiles per axis.
    # There is also some error in the original code where the number of bins
    # was used 3 times. If n_layers was not equal to n_bins tiled_state will
    # not have the right dimensions.
    # original: tiled_state = np.zeros(n_bins * n_bins * n_bins)
    tiled_state = np.zeros(n_tiles * n_tiles * n_layers)

    for row in range(n_layers):

        if position > pos_bins[row][0] and \
                position < pos_bins[row][n_bins - 1]:

            if velocity > vel_bins[row][0] and \
                    velocity < vel_bins[row][n_bins - 1]:

                x = np.digitize(position, pos_bins[row])
                y = np.digitize(velocity, vel_bins[row])

                # There is also a problem here with the original code
                # the idx equation is wrong because we are wasting space
                # in tiled_state with index values that will never be used (for
                # example it starts at index 3 with the original idx equation).
                # idx should be: (x * y) + (row * n_tiles**2) - 1
                # If we do that we are not wasting space in the array.
                # This is because in the course, the first possible value for
                # (x, y) in the example was (0, 0). But here the first possible
                # value with "np.digitize" is (x, y) = (1, 1).
                # We need to either change the idx equation (as below) or
                # substract 1 to x and y. Either way should work.
                # original: idx = (x + 1) * (y + 1) + row * n_bins**2 - 1
                idx = (x * y) + (row * n_tiles**2) - 1

                tiled_state[idx] = 1.0

            else:
                break
        else:
            break

    return tiled_state


class Model:
    def __init__(self, model_parameters={}):
        """Initialise model parameters.

        Keyword Arguments:
            model_parameters {dict} -- Parameters for the model.
            ("n_states": Number of states with tile coding. {int},
             "alpha": Step-size / Learning rate {float},
             "alpha_decay": Decaying value for alpha {float},
             "gamma": Discount rate {float})
        """
        self.ALPHA = model_parameters.get('alpha')
        self.GAMMA = model_parameters.get('gamma')
        self.n_states = model_parameters.get('n_states')
        self.alpha_decay = model_parameters.get('alpha_decay')
        self.weights = np.zeros(self.n_states)

    def calculate_v(self, state):
        """Calculate the value function for the current state.

        Arguments:
            state {array} -- Binary representation of the states.

        Returns:
            float -- Calculated weights for the current state.
        """
        return self.weights.dot(state)

    def update_weights(self, R, state, state_):
        """Update the weight of the given state. Multiply the array of weights
        by a binary representation of the states.

        Arguments:
            R {float} -- Rewards for the state.
            state {array} -- Binary representation of states at time t.
            state_ {array} -- Binary representation of states at time t+1
        """
        value = self.calculate_v(state)
        value_ = self.calculate_v(state_)
        self.weights += (self.ALPHA / self.alpha_decay) * (
            R + self.GAMMA * value_ - value) * state


def policy(velocity):
    """Evaluated policy. Goes backward for a velocity inferior to 0 and goes
    forward for a velocity superior or equal to 0.

    Arguments:
        velocity {float} -- Current velocity.

    Returns:
        int -- Action to take.
    """
    if velocity < 0:
        return BACKWARD
    elif velocity >= 0:
        return FORWARD


def approximate_v(num_episodes, n_bins=8, n_layers=8):
    """Generate an approximation of the value function over a number of
    episodes.

    Arguments:
        num_episodes {int} -- Number of episodes to play
        n_bins {int} -- Number of bins for Tile Coding
        n_layers {int} -- Number of layers for Tile Coding

    Returns:
        lists -- Lists containing the estimated values of the value function
        for the left side of the track and the point near the exit.
    """

    env = gym.make('MountainCar-v0')

    # Generate the bins
    pos_bins, vel_bins = get_bins(n_bins, n_layers)

    # Keep track at the value estimate at near the exit and at the left side
    near_exit_coord = (0.43, 0.054)
    left_side_coord = (-1.1, 0.001)
    near_exit = np.zeros((3, int(num_episodes / 1000)))
    left_side = np.zeros((3, int(num_episodes / 1000)))
    x = [i for i in range(near_exit.shape[1])]

    # Loop over 3 learning rates
    for k, lr in enumerate([1e-1, 1e-2, 1e-3]):

        model_parameters = {
            "alpha": lr,
            "gamma": 1.0,
            "n_states": (n_bins - 1) * (n_bins - 1) * n_layers,
            "alpha_decay": 1.0
        }

        model = Model(model_parameters)

        for i in range(num_episodes):
            if i % 1000 == 0:
                print('alpha', model.ALPHA, 'start episode', i)
                idx = i // 1000
                # Update near exit and left side value every 1000 games.
                tiled_state = tile_state(pos_bins, vel_bins, near_exit_coord)
                near_exit[k][idx] = model.calculate_v(tiled_state)
                tiled_state = tile_state(pos_bins, vel_bins, left_side_coord)
                left_side[k][idx] = model.calculate_v(tiled_state)

            # Decay the learning rate each 100 episodes
            if i % 100 == 0:
                model.alpha_decay += 10

            observation = env.reset()
            done = False

            # Play the episode.
            while not done:
                state = tile_state(pos_bins, vel_bins, observation)
                action = policy(observation[1])
                observation_, reward, done, _ = env.step(action)
                state_ = tile_state(pos_bins, vel_bins, observation_)
                model.update_weights(reward, state, state_)
                observation = observation_

    return x, near_exit, left_side


def plot_episodes(x, near_exit, left_side):
    """Plot a graph comparing the estimated values of the value function for
    different learning rates.

    Arguments:
        x {list} -- Timestep for the graph.
        near_exit {numpy.ndarray} -- Estimates values for the point near the
        exit for the different learning rates.
        left_side {numpy.ndarray} -- Estimates values for the left side of the
        track for the different learning rates.
    """

    plt.subplot(221)
    plt.plot(x, near_exit[0], 'r--')
    plt.plot(x, near_exit[1], 'g--')
    plt.plot(x, near_exit[2], 'b--')
    plt.title('Near exit, moving right')
    plt.subplot(222)
    plt.plot(x, left_side[0], 'r--')
    plt.plot(x, left_side[1], 'g--')
    plt.plot(x, left_side[2], 'b--')
    plt.title('Left side, moving right')
    plt.legend(('alpha = 0.1', 'alpha = 0.01', 'alpha = 0.001'))
    plt.show()


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-e",
                    "--episodes",
                    required=True,
                    help="Number of episodes to use.")
    args = vars(ap.parse_args())

    x, near_exit, left_side = approximate_v(num_episodes=int(args["episodes"]))
    plot_episodes(x, near_exit, left_side)
