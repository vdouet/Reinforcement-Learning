# Week 1 - Monte Carlo Methods

## Definitions

**Experience**: Sample sequences of states, actions, and rewards from actual or
simulated interaction with an environment. Learning from *actual* experience
requires no prior knowledge of the environment’s dynamics, yet can still attain
optimal behavior. Learning from simulated experience is also powerful. Although
a model is required, the model need only generate sample transitions, not the
complete probability distributions of all possible transitions that is required
for dynamic programming (DP).

**Monte Carlo methods**: Ways of solving the reinforcement learning problem
based on averaging sample returns. Here Monte Carlo is use specifically for
methods based on averaging complete returns (as opposed to methods that learn
from partial returns).

## Monte Carlo Prediction

For example, we can estimate the value of a given state by averaging the
returns observed after visits to that state. As more returns are observed, the
average should converge to the expected value. This idea underlies all Monte
Carlo methods.  
Each occurrence of state *s* in an episode is called a visit to *s*. The first
time it is visited is called *first visit* to *s*.  
The *first-visit MC method* estimates *vπ(s)* as the average of the returns
following first visits to *s*. The *every-visit MC method* averages the returns
following all visits to *s*. Both first-visit MC and every-visit MC converge to
*vπ(s)* as the number of visits (or first visits) to *s* goes to infinity.  
In backup diagram for Monte Carlo estimation of *vπ*, the root is a state node,
and below it is the entire trajectory of transitions along a particular single
episode, ending at the terminal state. Whereas the DP diagram shows all
possible transitions, the Monte Carlo diagram shows only those sampled on the
one episode. Whereas the DP diagram includes only one-step transitions, the
Monte Carlo diagram goes all the way to the end of the episode. These
differences in the diagrams accurately reflect the fundamental differences
between the algorithms.  
An important fact about Monte Carlo methods is that the estimates for each
state are independent. The estimate for one state does not build upon the
estimate of any other state, as is the case in DP. In other words, Monte Carlo
methods do not *bootstrap*.  
The computational expense of estimating the value of a single state is
independent of the number of states. This can make Monte Carlo methods
particularly attractive when one requires the value of only one or a subset of
states. One can generate many sample episodes starting from the states of
interest, averaging returns from only these states, ignoring all others. This
is a third advantage Monte Carlo methods can have over DP methods (after the
ability to learn from actual experience and from simulated experience).

## Monte Carlo Estimation of Action Values

If a model is not available, then it is particularly useful to estimate action
values rather than state values. With a model, state values alone are
sufficient to determine a policy; one simply looks ahead one step and chooses
whichever action leads to the best combination of reward and next state.
Without a model, however, state values alone are not sufficient. One must
explicitly estimate the value of each action in order for the values to be
useful in suggesting a policy. Thus, one of our primary goals for Monte Carlo
methods is to estimate *q\**. Monte Carlo methods for this are essentially the
same as just presented for state values, except now we talk about visits to a
state–action pair rather than to a state. A state– action pair *s*, *a* is said
to be visited in an episode if ever the state *s* is visited and action *a* is
taken in it. The only complication is that many state–action pairs may never be
visited. If *π* is a deterministic policy, then in following *π* one will
observe returns only for one of the actions from each state. This is a serious
problem because the purpose of learning action values is to help in choosing
among the actions available in each state. To compare alternatives we need to
estimate the value of *all* the actions from each state.  
This is the general problem of *maintaining exploration*. One way to do this is
by specifying that the episodes start in a *state–action pair*, and that every
pair has a nonzero probability of being selected as the start. We call this the
assumption of *exploring starts*. The assumption of exploring starts is
sometimes useful, but it cannot be relied upon in general, particularly when
learning directly from actual interaction with an environment. The most common
alternative approach to assuring that all state–action pairs are encountered is
to consider only policies that are stochastic with a nonzero probability of
selecting all actions in each state.

## Monte Carlo Control

Policy evaluation is done exactly as described in Monte Carlo estimation of
action values. Many episodes are experienced, with the approximate action-value
function approaching the true function asymptotically. Assuming an infinite
number of episodes and exploring starts the Monte Carlo methods will compute
each *qπk* exactly, for arbitrary *πk*.
Policy improvement is done by making the policy greedy with respect to the
current value function. In this case we have an action-value function, and
therefore no model is needed to construct the greedy policy. For any
action-value function *q*, the corresponding greedy policy is the one that, for
each *s* in *S*, deterministically chooses an action with maximal action-value.
Policy improvement then can be done by constructing each *πk+1* as the greedy
policy with respect to *qπk*.  
In this way Monte Carlo methods can be used to find optimal policies given only
sample episodes and no other knowledge of the environment’s dynamics.
We can use ways to remove the assumption that policy evaluation operates on an
infinite number of episodes:  
+ One is to hold firm to the idea of approximating *qπk* in
each policy evaluation. Measurements and assumptions are made to obtain bounds
on the magnitude and probability of error in the estimates, and then sufficient
steps are taken during each policy evaluation to assure that these bounds are
sufficient small. But it is likely to require far too many episodes to be useful
in practice on any but the smallest problems.
+ One other is to give up trying to complete policy evaluation before returning
to policy improvement. On each evaluation step we move the value function
toward *qπk*, but we do not expect to actually get close except over many
steps. One extreme form of the idea is value iteration, in which only one
iteration of iterative policy evaluation is performed between each step of
policy improvement. The in-place version of value iteration is even more
extreme; there we alternate between improvement and evaluation steps for single
states.  
For Monte Carlo policy iteration it is natural to alternate between evaluation
and improvement on an episode-by-episode basis. After each episode, the
observed returns are used for policy evaluation, and then the policy is
improved at all the states visited in the episode.

## Monte Carlo Control without Exploring Starts

There is two methods to avoid the assumption of exploring starts, resulting in
what we call on-policy methods and off-policy methods:
+ On-policy methods attempt to evaluate or improve the policy that is used to
make decisions.
+ Off-policy methods evaluate or improve a policy different from that used to
generate the data.  

In on-policy control methods the policy is generally *soft*, meaning that
*π(a|s)* > 0 for all *s* in *S* and all *a* in *A(s)*, but gradually shifted
closer and closer to a deterministic optimal policy. Many of the method for the
k-armed bandit provide mechanisms for this. The on-policy method we choose here
uses *ε-greedy* policies. The *ε-greedy* policies are examples of *ε-soft*
policies, defined as policies for which *π(a|s)* >= ε/|A(s)| for all states
and actions, for some *ε > 0*. Among *ε-soft* policies, *ε-greedy* policies are
in some sense those that are closest to greedy.  
The overall idea of on-policy Monte Carlo control is still that of GPI.
Fortunately, GPI does not require that the policy be taken all the way to a
greedy policy, only that it be moved *toward* a greedy policy. In our on-policy
method we will move it only to an ε-greedy policy. For any ε-soft policy, *π*,
any ε-greedy policy with respect to *qπ* is guaranteed to be better than or
equal to *π*.  
Using the natural notion of greedy policy for ε-soft policies, one is assured of
improvement on every step, except when the best policy has been found among the
ε-soft policies. Now we only achieve the best policy among the ε-soft policies,
but on the other hand, we have eliminated the assumption of exploring starts.
This on-policy learns action values not for the optimal policy, but for a
near-optimal policy that still explores.  

## Off-policy Prediction via Importance Sampling

For *off-policy learning* two policies are used. One that is learned about and
that becomes the optimal policy, and one that is more exploratory and is used
to generate behavior. The policy being learned about is called the *target
policy*, and the policy used to generate behavior is called the *behavior
policy*. In this case we say that learning is from data “off” the target policy.

On-policy methods are generally simpler and are considered first. Off-policy
methods require additional concepts and notation, and because the data is due
to a different policy, off-policy methods are often of greater variance and are
slower to converge. On the other hand, off-policy methods are more powerful,
general and have a variety of additional uses in applications. For example,
they can often be applied to learn from data generated by a conventional
non-learning controller, or from a human expert. Off-policy learning is also
seen by some as key to learning multi-step predictive models of the world’s
dynamics.

First we consider the *prediction* problem in which the target and behavior
policies are fixed. We want to estimate *vπ* and *qπ* but all we have are
episode from a behavior policy *b* =/= *π*. The assumption of *coverage*
require that every action taken under *π* is also taken under *b*. *π*(a|s) > 0
implies *b*(a|s). *b* must be stochastic in states where it is not identical to
*π*, but *π* may be deterministic.

Almost all off-policy methods utilize *importance sampling* for estimating
expected values under one distribution given samples from another. We apply
importance sampling to off-policy learning by weighting returns according to the
relative probability of their trajectories occurring under the target and
behavior policies, called the *importance-sampling ratio*. The
importance-sampling ration is given by:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/importancesamplingratio.png"
alt="Update rule" title="Update rule" width="446" height="74" />
</p>

The importance sampling ratio is depending only on the two policies and the
sequence, not on the MDP.  
We can use the importance-sampling ratio to transforms the returns *Gt* of the
behavior policy to have the right expected value for the target policy.

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/Gtb.png"
alt="Update rule" title="Update rule" width="145" height="22" />
</p>
<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/Gtratiopi.png"
alt="Update rule" title="Update rule" width="214" height="30" />
</p>

When importance sampling is done as a simple average it is called *ordinary
importance sampling* (see p.104 for notation).

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/ordinaryimportancesampling.png"
alt="Update rule" title="Update rule" width="214" height="67" />
</p>

An important alternative is *weighted importance sampling*, which uses a
*weighted* average, defined as (or zero if the denominator is zero):

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/weightedimportancesampling.png"
alt="Update rule" title="Update rule" width="205" height="56" />
</p>

The difference between the first-visit methods of the two kinds of importance
sampling is expressed in their biases and variances. Ordinary importance
sampling is unbiased whereas weighted importance sampling is biased (though the
bias converges asymptotically to zero). On the other hand, the variance of
ordinary importance sampling is in general unbounded because the variance of
the ratios can be unbounded, whereas in the weighted estimator the largest
weight on any single return is one. In fact, assuming bounded returns, the
variance of the weighted importance-sampling estimator converges to zero even
if the variance of the ratios themselves is infinite. In practice, the weighted
estimator usually has dramatically lower variance and is strongly preferred.

The every-visit methods for ordinary and weighed importance sampling are both
biased, though, again, the bias falls asymptotically to zero as the number of
samples increases. In practice, every-visit methods are often preferred because
they remove the need to keep track of which states have been visited and
because they are much easier to extend to approximations.

## Incremental Implementation

Monte Carlo prediction methods can be implemented incrementally, on an episode-
by-episode basis, using incremental updates. Using those methods, we previously
average rewards, wheras in Monte Carlo methods we average returns. The same
methods can be use for *on-policy* Monte Carlo methods. For *off-policy* Monte
Carlo methods using weighted importance sampling we have to form a weighted
average of the returns, and a slightly different incremental algorithm is
required.

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/offpolicyincremental.png"
alt="Update rule" title="Update rule" width="298" height="57" />
</p>

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/offpolicyincrementalc.png"
alt="Update rule" title="Update rule" width="157" height="35" />
</p>

## Discounting-aware Importance Sampling

Previous off-policy methods are using undiscounted rewards. For discounted
rewards, we can consider gamma as a *degree* of partial termination. We need
only to compute the non-terminated *Gt* (*flat partial returns*) and *V(s)* and
not the unnecessary ones that do not change the expected update and add
enormously to its variance. “*flat*” denotes the absence of discounting, and
“*partial*” denotes that these returns do not extend all the way to termination
but instead stop at *h*, called the *horizon*.

## Monte Carlo methods advantages

+ Can be used to learn optimal behavior directly from interaction with the
environment, with no model of the environment’s dynamics.
+ Can be used with simulation or sample models.
+ It is easy and efficient to focus Monte Carlo methods on a small subset of the
states. A region of special interest can be accurately evaluated without going
to the expense of accurately evaluating the rest of the state set.
+ They may be less harmed by violations of the Markov property. This is because
they do not update their value estimates on the basis of the value estimates of
successor states. In other words, it is because they do not bootstrap.
