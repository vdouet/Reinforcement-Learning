# Week 1 - Multi-armed Bandits

## Definitions

**Purely evaluative feedback**: Indicates how good the action taken was. It is
dependent of the action taken.

**Purely instructive feedback**: Indicates what the correct action was
independently of the action actually taken. It is independent of the action
taken.

**q\*(a)**: The value of an action -> Expected reward when that action is taken.
The true value of an action is the mean reward when that action is selected
q*(a) is estimated Qt(a)

**Nonassociative**: Action taken in only one situation.

**Associative**: Action taken in more than one situation.

**Greedy actions**: Actions where the estimate value is the greatest.

**Choosing greedy actions**: Exploiting

**Choosing non-greedy actions:** Exploring -> Enable us to improve the estimates
of the non-greedy action's value.

**Conflict**: Exploitation vs Exploration.  
Exploitation can be better in short term but exploration can be better in
longterm.

**ε-greedy methods**: behave greedily most of the time, but every once in a
while, with small probability ε, instead select randomly from among all the
actions with equal probability, independently of the action-value estimates.

**Nonstationary**: True values of the actions changed over time. Nonstationarity
is the case most commonly encountered in reinforcement learning

**Sample-Average Method**: Approximate q\*(a) as Qt(a) = sum of rewards when *a*
taken prior to *t* / number of times *a* taken prior to *t*  
It is the average of rewards.

**Incremental Update Rule**:
`NewEstimate <- OldEstimate + StepSize[Target - OldEstimate]`  
Sample average case:  
<img src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/IncrementalUpdateRule.png" alt="Update rule"
	title="Update rule" width="281" height="64" />  
Nonstationary case:  
<img src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/nonstationnaryRule.png" alt="Update rule"
	title="Update rule" width="303" height="76" />  

**Policy**: A mapping from situations to the actions that are best in those
situations.

**Associative search task**: Involves both trial-and-error learning to search
for the best actions, and association of these actions with the situations in
which they are best. Associative search tasks are often now called *contextual*
bandits in the literature.

## Exploration and exploitation balancing methods

**Optimistic Initial Values**: Set all initial actions to an optimistic value.
Encourages action-value methods to explore. The reward for each action will be
less than the initial optimistics values forcing the learning to choose another
action. Whichever actions are initially selected, the reward is less than the
starting estimates; the learner switches to other actions, being “disappointed”
with the rewards it is receiving. The result is that all actions are tried
several times before the value estimates converge. The system does a fair amount
of exploration even if greedy actions are selected all the time.  
This technique can be effective in stationary problem but not well suited for
nonstationary problems, also it only drive early exploration. We also may not
know what the optimistic initial value should be.

**Upper confidence bound (UCB)**:
Select among the non-greedy actions according to their potential for actually
being optimal, taking into account both how close their estimates are to being
maximal and the uncertainties in those estimates.  
It uses uncertainty in the value estimates for balancing exploration and
exploitation.  
All actions will eventually be selected, but actions with lower value estimates,
or that have already been selected frequently, will be selected with decreasing
frequency over time.  
Performs well but more difficult than ε-greedy to extend beyond bandits to the
more general reinforcement learning settings.  
Also difficulty to deal with nonstationary problems and large state spaces.

**Gradient bandit algorithms**: Estimate not action values, but action
preferences, and favor the more preferred actions in a graded, probabilistic
manner using a soft-max distribution.

**The Gittins-index approach**: is an instance of Bayesian methods, which assume
a known initial distribution over the action values and then update the
distribution exactly after each step (assuming that the true action values are
stationary).  
*Posterior sampling* or *Thompson sampling*: In the case of *conjugate priors*
distributions, select actions at each step according to their posterior
probability of being the best action

## Real-world Reinforcement Learning

Difference between simulation and reality, large simulator/reality divergence.

For a real-world RL application you may need to shift your priorities.
Something important in a simulation environment can be less important in a
real-world environment and vice versa. For example in a simulation you have more
control over the environment as in the real-world you don't. Also, in the
real-world you might want your algorithm to generalize better than in a
simulation where you might prefer greatest rewards. In the real-world you only
have the samples that it gives you, a limited amount compared to the quasi
infinite amount of samples you can compute in a simulated environment.
