# Temporal-Difference Learning

## Definitions

**Temporal-Difference Learning (TD)**: TD learning is a combination of Monte 
Carlo ideas and dynamic programming (DP) ideas. TD methods can learn directly 
from raw experience without a model of the environment’s dynamics and update 
estimates based in part on other learned estimates (boostrap). TD, DP and Monte
Carlo methods can be combined in different ways. Model free.

**Batch Updating**: For a finite amount of experience a common approach with 
incremental learning methods is to present the experience repeatedly until the
method converges upon an answer. Updates are made only after processing each 
complete batch of training data.

## TD Prediction

Both TD and Monte Carlo methods use experience to solve the prediction problem.
Monte Carlo methods must wait until the end of the episode to determine the 
increment to V(St):

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/vtMC.png"
alt="Update rule" title="Update rule" width="245" height="41" />
</p>

TD methods need to wait only until the next time step:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/vtTD0.png"
alt="Update rule" title="Update rule" width="350" height="37" />
</p>

This TD method is called *TD(0)*, or *one-step* TD, because it is a special
case of the TD(λ) and *n-step* TD methods. The target for Monte Carlo update is
*Gt*, the target for the TD update is *Rt+1 + γ\*V(St+1)*. TD methods combine
the sampling of Monte Carlo with the boostrapping of DP. TD and Monte Carlo use
*sample updates*, DP uses *expected updates*. *Sample* updates are based on a 
single sample successor rather than on a complete distribution of all possible
successors.
The quantity in bracket un TD(0) update is a sort of error, measuring the 
difference between the estimated value of *St* and the better estimate. It is
called the *TD error*:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/tderror.png"
alt="Update rule" title="Update rule" width="234" height="34" />
</p>

On the next picture an example of the update difference between Monte Carlo and
TD:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/tdmcex.png"
alt="Update rule" title="Update rule" width="682" height="330" />
</p>

For the MC methods the change in prediction (red arrow) is updated offline, at
the end of the episode (arrive home) because only at this point you know the 
actual return. For the TD methods the change in prediction is done immediately
at the next step, each error is proportional to the change over time of the 
prediction, that is, to the *temporal differences* in predictions.

## Advantages of TD Prediction Methods

+ TD methods do not require a model of the environment, of its reward and 
next-state probability distributions.
+ They are naturally implemented in an online, fully incremental fashion.
+ Can be applied to continuous tasks.
+ In practice, TD methods have usually been found to converge faster than 
constant-α MC methods on stochastic tasks.

## Optimality of TD(0)

Batch Monte Carlo methods always find the estimates that minimize mean-squared 
error on the training set, whereas batch TD(0) always finds the estimates that 
would be exactly correct for the maximum-likelihood model of the Markov
process. The *maximum-likelihood estimate* of a parameter is the parameter 
value whose probability of generating the data is greatest for example, a model
of the Markov process. Given this model, we can compute the estimate of the 
value function that would be exactly correct if the model were exactly correct.
This is called the *certainty-equivalence estimate* because it is equivalent 
to assuming that the estimate of the underlying process was known with 
certainty rather than being approximated. In general, batch TD(0) converges to 
the certainty-equivalence estimate.  
In batch form, TD(0) is faster than Monte Carlo methods because it computes the
true certainty-equivalence estimate. The relationship to the 
certainty-equivalence estimate may also explain in part the speed advantage of
nonbatch TD(0). Although the nonbatch methods do not achieve either the 
certainty-equivalence or the minimum squared-error estimates, they can be 
understood as moving roughly in these directions. Nonbatch TD(0) may be faster 
than constant-α MC because it is moving toward a better estimate, even though 
it is not getting all the way there. 

## Sarsa: On-policy TD Control

We must estimate *qπ(s,a)* for the current behavior policy *π* and for all 
states *s* and actions *a*. We consider transitions from state–action pair to 
state–action pair, and learn the values of state–action pairs.

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/sarsaq.png"
alt="Update rule" title="Update rule" width=469" height="38" />
</p>

This update is done after every transition from a nonterminal state *St*. 
If *St+1* is terminal, then *Q(St+1,At+1)* is defined as zero. The name Sarsa
is given by the quintuple of events, (*St, At, Rt+1, St+1, At+1*), that make up
a transition from one state–action pair to the next. Sarsa converges with 
probability 1 to an optimal policy and action-value function as long as all 
state–action pairs are visited an infinite number of times and the policy 
converges in the limit to the greedy policy.

## Q-learning: Off-policy TD Control

One of the early breakthroughs in reinforcement learning was the development of
an off-policy TD control algorithm known as Q-learning.

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/qlearning.png"
alt="Update rule" title="Update rule" width=469" height="39" />
</p>

The learned action-value function, *Q*, directly approximates *q\**, the 
optimal action-value function, independent of the policy being followed. The 
policy still determines which state–action pairs are visited and updated but 
all that is required for correct convergence is that all pairs continue to be 
updated.

Q-Learning algorithm directly finds the optimal action-value function (*q\**) 
without any dependency on the policy being followed. The policy only helps to 
select the next state-action pair from a current state. Hence, Q-Learning is an
off-policy method.

SARSA will approach convergence allowing for possible penalties from 
exploratory moves, whilst Q-learning will ignore them. That makes SARSA more 
conservative - if there is risk of a large negative reward close to the optimal
path, Q-learning will tend to trigger that reward whilst exploring, whilst 
SARSA will tend to avoid a dangerous optimal path and only slowly learn to use
it when the exploration parameters are reduced (see Cliff Walking p.132).

## Expected Sarsa

Expected Sarsa is just like Q-learning except that instead of the maximum over 
next state–action pairs it uses the expected value, taking into account how 
likely each action is under the current policy but otherwise follows the schema
of Q-learning.

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/expectedsarsa.png"
alt="Update rule" title="Update rule" width=539" height="80" />
</p>

Expected Sarsa is more complex computationally than Sarsa but eliminates the
variance due to the random selection of *At+1*. Given the same amount of
experience it generally perform slightly better than Sarsa. Expected Sarsa 
subsumes and generalizes Q-learning while reliably improving over Sarsa. Except
for the small additional computational cost, Expected Sarsa may completely 
dominate both of the other more-well-known TD control algorithms. Expected
Sarsa can be used as on-policy or as an off-policy.

## Maximization Bias and Double Learning

In the previous algorithms a maximum over estimated values is used implicitly 
as an estimate of the maximum value, which can lead to a significant positive 
bias. We call this *maximization bias*. Maximization bias can make an algorithm
like Q-learning favor a non-optimal action over an optimal one because of the
maximum over estimated values (see Maximization Bias Example p.134).

One way to view the problem is that it is due to using the same samples both to
determine the maximizing action and to estimate its value. With *double
learning* we can use two independant estimates *Q1(a)* and *Q2(a)*, each an 
estimate of the true value q(a). We can use *Q1* to determine the maximizing
action *A\** = argmaxa *Q1(a)*, and *Q2* to provide the estimate of its value,
*Q2(A\*) = Q2(argmaxa Q1(a))*. This estimate will then be unbiased in the sense
that E\[*Q2(A\*)*\] = *q(A\*)*. We can also repeat the process with the role of
the two estimates reversed to yield a second unbiased estimate 
*Q1(argmaxa Q2(a))*.  
Although we learn two estimates, only one estimate is updated on each play; 
double learning doubles the memory requirements, but does not increase the 
amount of computation per step. 

The double learning algorithm analogous to Q-learning, called Double Q-learning
, divides the time steps in two. For a,probability *p1* the update is:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/doubleQlearning.png"
alt="Update rule" title="Update rule" width=575" height="37" />
</p>

For 1-*p1* then the same update is done with *Q1* and *Q2* switched, so that
*Q2* is updated. The behavior policy can use both action-value estimates. For 
example, an ε-greedy policy for Double Q-learning could be based on the average
(or sum) of the two action-value estimates. There is also double version of
Sarsa and Expected Sarsa.

## Games, Afterstates, and Other Special Cases

For some case it might be useful to evaluate states *after* the agent has made
its move. These are called *afterstates*, and value functions over these, 
*afterstate value functions*. Afterstates are useful when we have knowledge of 
an initial part of the environment’s dynamics but not necessarily of the full 
dynamics. For example, in games we typically know the immediate effects of our 
moves. We know for each possible chess move what the resulting position will 
be, but not how our opponent will reply. Afterstate value functions are a 
natural way to take advantage of this kind of knowledge and thereby produce a 
more efficient learning method. Afterstates arise in many tasks, not just 
games.

However, it is impossible to describe all the possible kinds of specialized 
problems and corresponding specialized learning algorithms.