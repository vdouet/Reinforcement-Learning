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

**Policy Improvement/Control**: Process of making a new policy that improves on
an original policy, by making it greedy with respect to the value function of
the original policy.

**Policy Iteration**: Evaluate and improve policies until reaching an optimal
one.

**Value Iteration**: During policy iteration, when policy evaluation is stopped
after just one sweep (one update of each state).

**generalized policy iteration (GPI)**: Refer to the general
idea of letting policy-evaluation and policy-improvement processes interact,
independent of the granularity and other details of the two processes.

## Policy Evaluation (Prediction)

### Iterative Policy Evaluation
Consider a sequence of approximate value functions *v0, v1, v2, ...,* each
mapping S+ to R (the real numbers). The initial approximation, *v0*, is chosen
arbitrarily (except that the terminal state, if any, must be given value 0),
and each successive approximation is obtained by using the Bellman equation
for *vπ* as an update rule:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/vpiupdatefunction.png"
alt="Update rule" title="Update rule" width="374" height="83" />
</p>

the sequence {*vk*} can be shown in general to converge to *vπ* as k->inf
under the same conditions that guarantee the existence of *vπ*. To produce
each successive approximation, *vk+1* from *vk*, *iterative policy evaluation*
applies the same operation to each state *s*: it replaces the old value of *s*
with a new value obtained from the old values of the successor states of *s*,
and the expected immediate rewards, along all the one-step transitions possible
under the policy being evaluated. We call this kind of operation an *expected
update*. Each iteration of iterative policy evaluation updates the value of
every state once to produce the new approximate value function *vk+1*.

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
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/policyimprovement3.png"
alt="Update rule" title="Update rule" width="387" height="85" />
</p>

Policy improvement must give us a strictly better policy except when the
original policy is already optimal.

## Policy Iteration

Once a policy, *π*, has been improved using *vπ* to yield a better policy,
*π'*, we can then compute *vπ'* and improve it again to yield an even better
*π''*.

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/policyiteration.png"
alt="Update rule" title="Update rule" width="410" height="46" />
</p>

Where ->E denotes a policy *evaluation* and ->I denotes a policy *improvement*.
Because a finite MDP has only a finite number of policies, this process must
converge to an optimal policy and optimal value function in a finite number of
iterations. This is called *policy iteration*

One drawback to policy iteration is that each of its iterations involves policy
evaluation, which may itself be a prolonged iterative computation requiring
multiple sweeps through the state set.

Policy iteration consists of two simultaneous, interacting processes, one
making the value function consistent with the current policy (policy
evaluation), and the other making the policy greedy with respect to the current
value function (policy improvement). In policy iteration, these two processes
alternate, each completing before the other begins.

## Value Iteration

The policy evaluation step of policy iteration can be truncated in several ways
without losing the convergence guarantees of policy iteration. One important
special case is when policy evaluation is stopped after just one sweep (one
update of each state). This algorithm is called *value iteration*.
It can be written as a particularly simple update operation that combines the
policy improvement and truncated policy evaluation steps:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/valueiteration.png"
alt="Update rule" title="Update rule" width="402" height="81" />
</p>

Note that value iteration is obtained simply by turning the Bellman optimality
equation into an update rule. Also note how the value iteration update is
identical to the policy evaluation update except that it requires the
maximum to be taken over all actions.

Value iteration effectively combines, in each of its sweeps, one sweep of policy
evaluation and one sweep of policy improvement. Faster convergence is often
achieved by interposing multiple policy evaluation sweeps between each policy
improvement sweep.

## Asynchronous Dynamic Programming

A major drawback to the previous DP methods is that they involve operations
over the entire state set of the MDP, that is, they require sweeps of the state
set.  
Asynchronous DP algorithms are in-place iterative DP algorithms that are not
organized in terms of systematic sweeps of the state set. These algorithms
update the values of states in any order whatsoever, using whatever values of
other states happen to be available. The values of some states may be updated
several times before the values of others are updated once. To converge
correctly, however, an asynchronous algorithm must continue to update the
values of all the states: it can’t ignore any state after some point in the
computation. Asynchronous DP algorithms allow great flexibility in selecting
states to update. We can try to order the updates to let value information
propagate from state to state in an efficient way. Some states may not need
their values updated as often as others. We might even try to skip updating some
states entirely if they are not relevant to optimal behavior. Similarly to
other DP methods, it is possible to intermix policy evaluation and value
iteration updates to produce a kind of asynchronous truncated policy iteration.
Avoiding sweeps does not necessarily mean that we can get away with less
computation. It just means that an algorithm does not need to get locked into
any hopelessly long sweep before it can make progress improving a policy.

Asynchronous algorithms also make it easier to intermix computation with
real-time interaction. To solve a given MDP, we can run an iterative DP
algorithm at the same time that an agent is actually experiencing the MDP. The
agent’s experience can be used to determine the states to which the DP
algorithm applies its updates. At the same time, the latest value and policy
information from the DP algorithm can guide the agent’s decision making. For
example, we can apply updates to states as the agent visits them. This makes it
possible to focus the DP algorithm’s updates onto parts of the state set that
are most relevant to the agent. This kind of focusing is a repeated theme in
reinforcement learning.

## Generalized Policy Iteration

We use the term *generalized policy iteration* (GPI) to refer to the general
idea of letting policy-evaluation and policy-improvement processes interact,
independent of the granularity and other details of the two processes.
Almost all reinforcement learning methods are well described as GPI. That is,
all have identifiable policies and value functions, with the policy always being
improved with respect to the value function and the value function always being
driven toward the value function for the policy, as suggested by the diagram:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/GPIdiagram.png"
alt="Update rule" title="Update rule" width="154" height="244" />
</p>

If both the evaluation process and the improvement process stabilize, that is,
no longer produce changes, then the value function and policy must be optimal.
The value function stabilizes only when it is consistent with the current
policy, and the policy stabilizes only when it is greedy with respect to the
current value function. Thus, both processes stabilize only when a policy has
been found that is greedy with respect to its own evaluation function. This
implies that the Bellman optimality equation holds, and thus that the policy and
the value function are optimal.

## Efficiency of Dynamic Programming


DP may not be practical for very large problems, but compared with other
methods for solving MDPs, DP methods are actually quite efficient. DP is
exponentially faster than any direct search in policy space could be, because
direct search would have to exhaustively examine each policy to provide the
same guarantee. Linear programming methods can also be used to solve MDPs, and
in some cases their worst-case convergence guarantees are better than those of
DP methods. But linear programming methods become impractical at a much smaller
number of states than do DP methods (by a factor of about 100). For the largest
problems, only DP methods are feasible.  
DP is sometimes thought to be of limited applicability because of the *curse of
dimensionality*, the fact that the number of states often grows exponentially
with the number of state variables. Large state sets do create difficulties, but
these are inherent difficulties of the problem, not of DP as a solution method.
In fact, DP is comparatively better suited to handling large state spaces than
competing methods such as direct search and linear programming.  
In practice, DP methods can be used with today’s computers to solve MDPs with
millions of states. Both policy iteration and value iteration are widely used,
and it is not clear which, if either, is better in general. In practice, these
methods usually converge much faster than their theoretical worst-case run
times, particularly if they are started with good initial value functions or
policies.  
On problems with large state spaces, asynchronous DP methods are often
preferred. To complete even one sweep of a synchronous method requires
computation and memory  for every state. For some problems, even this much
memory and computation is impractical, yet the problem is still potentially
solvable because relatively few states occur along optimal solution
trajectories. Asynchronous methods and other variations of GPI can be applied
in such cases and may find good or optimal policies much faster than synchronous
methods can.
