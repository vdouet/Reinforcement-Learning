# From the course "Reinforcement Learning in Motion" by Phil Tabor.

import numpy as np
import matplotlib.pyplot as plt
import argparse
import gym

BACKWARD = 0
FORWARD = 2


class Model:
    def __init__(self, model_parameters={}):
        """Initialise model parameters.

        Keyword Arguments:
            model_parameters {dict} -- Parameters for the model.
            ("state_space": List of aggregated states {list},
             "alpha": Step-size / Learning rate {float},
             "alpha_decay": Decaying value for alpha {float},
             "gamma": Discount rate {float})
        """
        self.ALPHA = model_parameters.get('alpha')
        self.GAMMA = model_parameters.get('gamma')
        self.state_space = model_parameters.get('state_space')
        self.alpha_decay = model_parameters.get('alpha_decay')
        self.weights = {}
        for state in self.state_space:
            self.weights[state] = 0

    def calculate_v(self, state):
        """Calculate the value for the state.

        Arguments:
            state {tuple} -- Tuple used as key for the weights dictionnary.

        Returns:
            dict -- Updated dictionnary of weights.
        """
        return self.weights[state]

    def update_weights(self, G, state):
        """Update the weight of the given state.

        Arguments:
            G {float} -- Expected returns for the state.
            state {tuple} -- Tuple used as key for the weights dictionnary.
        """
        value = self.calculate_v(state)
        self.weights[state] += (self.ALPHA / self.alpha_decay) * (G - value)


def aggregate_state(pos_bins, vel_bins, obs):
    """Aggregate the states space.

    Arguments:
        pos_bins {numpy.ndarray} -- Bins for the position.
        vel_bins {numpy.ndarray} -- Bins for the velocity.
        obs {list} -- List with the current position and velocity.

    Returns:
        tuple -- Current aggregated state.
    """
    pos = int(np.digitize(obs[0], pos_bins))
    vel = int(np.digitize(obs[1], vel_bins))
    state = (pos, vel)
    return state


def policy(vel):
    """Evaluated policy. Goes backward for a velocity inferior to 4 and goes
    forward for a velocity superior or equal to 4.

    Arguments:
        vel {float} -- Current velocity.

    Returns:
        int -- Action to take.
    """
    if vel < 4:
        return BACKWARD
    elif vel >= 4:
        return FORWARD


def approximate_v(num_episodes):
    """Generate an approximation of the value function over a number of
    episodes.

    Arguments:
        num_episodes {int} -- Number of episodes to play

    Returns:
        lists -- Lists containing the estimated values of the value function
        for the left side of the track and the point near the exit.
    """

    env = gym.make('MountainCar-v0')

    # Aggregate state space into 8 parts
    pos_bins = np.linspace(-1.2, 0.5, 8)
    vel_bins = np.linspace(-0.07, 0.07, 8)

    # Initialise the state space
    state_space = []
    for i in range(1, 9):
        for j in range(1, 9):
            state_space.append((i, j))

    # Keep track at the value estimate at near the exit and at the left side
    near_exit_coord = (0.43, 0.054)
    left_side_cood = (-1.1, 0.001)
    near_exit = np.zeros((3, int(num_episodes / 1000)))
    left_side = np.zeros((3, int(num_episodes / 1000)))
    x = [i for i in range(near_exit.shape[1])]

    # Loop over 3 learning rates
    for k, lr in enumerate([0.1, 0.01, 0.001]):

        model_parameters = {
            "alpha": lr,
            "gamma": 1.0,
            "state_space": state_space,
            "alpha_decay": 1.0
        }

        model = Model(model_parameters)

        for i in range(num_episodes):
            if i % 1000 == 0:
                print('Start episode', i)
                idx = i // 1000
                # Update near exit and left side value every 1000 games.
                state = aggregate_state(pos_bins, vel_bins, near_exit_coord)
                near_exit[k][idx] = model.calculate_v(state)
                state = aggregate_state(pos_bins, vel_bins, left_side_cood)
                left_side[k][idx] = model.calculate_v(state)
                # Decay the learning rate
                model.alpha_decay += 0.1

            observation = env.reset()
            done = False
            memory = []

            # Play the episode.
            while not done:
                state = aggregate_state(pos_bins, vel_bins, observation)
                action = policy(state[1])
                observation_, reward, done, _ = env.step(action)
                memory.append((state, action, reward))
                observation = observation_

            # Append the terminal state.
            state = aggregate_state(pos_bins, vel_bins, observation)
            memory.append((state, action, reward))

            G = 0
            last = True
            states_returns = []
            for state, action, reward in reversed(memory):
                if last:
                    last = False
                else:
                    states_returns.append((state, G))
                G = model.GAMMA * G + reward

            states_returns.reverse()
            states_visited = []
            for state, G in states_returns:
                if state not in states_visited:
                    model.update_weights(G, state)
                    states_visited.append(state)

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
