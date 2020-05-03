# Week 4: Dynamic Programming

## Definitions

**Dynamic Programming**: Refers to a collection of algorithms that can be used
to compute optimal policies given a perfect model of the environment as a
Markov decision process (MDP). Classical DP algorithms are of limited utility
in reinforcement learning both because of their assumption of a perfect model
and because of their great computational expense, but they are still important
theoretically. Although DP ideas can be applied to problems with continuous
state and action spaces, exact solutions are possible only in special cases.

**Policy Evaluation/Prediction Problem**: How to compute the state-value
function *vπ* for an arbitrary policy *π*.

**Policy Improvement**: Process of making a new policy that improves on an
original policy, by making it greedy with respect to the value function of
the original policy

## Policy Evaluation (Prediction)

### Iterative Policy Evaluation
Consider a sequence of approximate value functions *v0, v1, v2, ...,* each
mapping S+ to R (the real numbers). The initial approximation, *v0*, is chosen
arbitrarily (except that the terminal state, if any, must be given value 0),
and each successive approximation is obtained by using the Bellman equation for
*vπ* as an update rule:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/vpiupdatefunction.png"
alt="Update rule" title="Update rule" width="374" height="83" />
</p>

the sequence {*vk*} can be shown in general to converge to *vπ* as k->inf under
the same conditions that guarantee the existence of *vπ*.
To produce each successive approximation, *vk+1* from *vk*, *iterative policy
evaluation* applies the same operation to each state *s*: it replaces the old
value of *s* with a new value obtained from the old values of the successor
states of *s*, and the expected immediate rewards, along all the one-step
transitions possible under the policy being evaluated. We call this kind of
operation an *expected update*. Each iteration of iterative policy evaluation
updates the value of every state once to produce the new approximate value
function *vk+1*.

All the different kind of updates done in DP algorithms are called *expected
updates* because they are based on an expectation over all possible next states
rather than on a sample next state.

## Policy Improvement

For some state *s* we would like to know whether or not we should change the
policy to deterministically choose an action *a =/= π(s)*. We can consider
selecting *a* in *s* and thereafter following the existing policy, π.

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/policyimprovement1.png"
alt="Update rule" title="Update rule" width="359" height="75" />
</p>

The key criterion is whether this is greater than or less than *vπ(s)*. If it is
greater—that is, if it is better to select *a* once in *s* and thereafter follow
*π* than it would be to follow *π* all the time—then one would expect it to be
better still to select *a* every time *s* is encountered, and that the new
policy would in fact be a better one overall. This is a special case of the
*policy improvement theorem*:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/policyimprovmenttheorem.png"
alt="Update rule" title="Update rule" width="158" height="34" />
</p>

It is also possible to consider changes at all states and to all possible
actions, selecting at each state the action that appears best according to
*qπ(s,a)*. In other words, to consider the new greedy policy, *π'*, given by:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/policyimprovement2.png"
alt="Update rule" title="Update rule" width="405" height="122" />
</p>

where *argmax of a* denotes the value of *a* at which the expression that
follows is maximized (with ties broken arbitrarily). The greedy policy takes
the action that looks best in the short term—after one step of
lookahead—according to *vπ*. The process of making a new policy that improves
on an original policy, by making it greedy with respect to the value function of
the original policy, is called *policy improvement*.

If a new greedy policy *π'* is as good as but not better than the old policy
*π* then *vπ'* = *vπ* = *v\** and both *π'* and *π* are optimal policy (Bellman
optimality equation).

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/policyimprovement2.png"
alt="Update rule" title="Update rule" width="387" height="85" />
</p>

Policy improvement must give us a strictly better policy except when the
original policy is already optimal.

## Policy Iteration
