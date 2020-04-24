#Week 1

Purely evaluative feedback: Indicates how good the action taken was. It is
dependent of the action taken.

Purely instructive feedback: Indicates what the correct action was independently
of the action actually taken. It is independent of the action taken.

q*(a): The value of an action -> Expected reward when that action is taken.
q*(a) is estimated Qt(a)

nonassociative: Action taken in only one situation.
asssociative: Action taken in more than one situation.

Greedy actions: Actions where the estimate value is the greatest.
Choosing greedy actions: Exploiting
Choosing non-greedy actions: Exploring -> Enable us to improve the estimates
of the non-greedy action's value.

Conflict: Exploitation vs Exploration.
Exploitation can be better in short term but exploration can be better in long
term.

- The true value of an action is the mean reward when that action is selected

ε-greedy methods: behave greedily most of the time, but every once in a while,
with small probability ε, instead select randomly from among all the actions
with equal probability, independently of the action-value estimates.

Nonstationary: True values of the actions changed over time.
Nonstationarity is the case most commonly encountered in reinforcement learning

Weighted average sum: (a = constant)
Qn+1 = a*Rn + (1-a)*Qn

Optimistic Initial Values: Set all initial actions to an optimistic value.
Encourages action-value methods to explore. The reward for each action will be
less than the initial optimistics values forcing the learning to choose another
action. Whichever actions are initially selected, the reward is less than the
starting estimates; the learner switches to other actions, being “disappointed”
with the rewards it is receiving. The result is that all actions are tried
several times before the value estimates converge. The system does a fair amount
of exploration even if greedy actions are selected all the time.
This technique can be effective in stationary problem but not well suited for
nonstationary problems.

Upper confidence bound (UCB):
Select among the non-greedy actions according to their potential for actually
being optimal, taking into account both how close their estimates are to being
maximal and the uncertainties in those estimates.
All actions will eventually be selected, but actions with lower value estimates,
or that have already been selected frequently, will be selected with decreasing
frequency over time.
Performs well but more difficult than ε-greedy to extend beyond bandits to the
more general reinforcement learning settings.
Also difficulty to deal with nonstationary problems and large state spaces.
