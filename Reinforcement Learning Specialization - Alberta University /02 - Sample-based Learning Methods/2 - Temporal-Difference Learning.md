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