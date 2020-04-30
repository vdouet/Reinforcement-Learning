# Week 3

## Definitions

**Value functions**: Functions of states (or of state–action pairs) that
estimate how good it is for the agent to be in a given state (or how good it is
to perform a given action in a given state). The notion of “how good” here is
defined in terms of future rewards that can be expected, or, to be precise, in
terms of expected return.

**Policy**: Mapping from states to probabilities of selecting each possible
action.

**Optimal Policy**: Policy that is better than or equal to all other policies.

## Policies and Value Functions

Value functions are defined with respect to particular way of acting, called
policies. If the agent is following policy *π* at time *t*, then *π(a|s)* is the
probability that *At* = *a* if *St* = *s*. Like *p*, *π* is an ordinary
function.

The value function of a state *s* under a policy *π*, denoted *vπ(s)* is the
expected return when starting in *s* and following *π* thereafter.

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/valuefunction1.png"
alt="Update rule" title="Update rule" width="510" height="62" />
</p>

*E[.]* is the expected value of a random variable given that the agent follows
policy *π*. The value of the terminal state is always 0. *vπ* is called the
*state-value function for policy π*

*qπ(s,a)* is the expected return starting from *s*, taking the action *a* and
thereafter following policy *π*

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/actionvaluefunction.png"
alt="Update rule" title="Update rule" width="546" height="68" />
</p>

*qπ* is called the *action-value function for policy π*

*vπ* and *qπ* can be estimated from experience. For example, if an agent
follows policy π and maintains an average, for each state encountered, of the
actual returns that have followed that state, then the average will converge to
the state’s value, *vπ(s)*, as the number of times that state is encountered
approaches infinity. If separate averages are kept for each action taken in
each state, then these averages will similarly converge to the action values,
*qπ(s,a)*. We call estimation methods of this kind *Monte Carlo methods* because
they involve averaging over many random samples of actual returns.

A fundamental property of value functions used throughout reinforcement
learning and dynamic programming is that they satisfy recursive relationships.
For any policy *π* and any state *s*, the following consistency condition holds
between the value of *s* and the value of its possible successor states:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/valuefunctionrecursive.png"
alt="Update rule" title="Update rule" width="472" height="150" />
</p>

Equation (3.14) is the *Bellman equation* for *vπ*. It expresses a relationship
between the value of a state and the values of its successor states. On the
following diagram, each open circle represents a state and each solid circle
represents a state–action pair. Starting from state *s*, the root node at the
top, the agent could take any of some set of actions based on its policy *π*.
From each of these, the environment could respond with one of several next
states, *s'*, along with a reward *r*, depending on its dynamics given by the
function *p*. The Bellman equation (3.14) averages over all the possibilities,
weighting each by its probability of occurring. It states that the value of the
start state must equal the (discounted) value of the expected next state, plus
the reward expected along the way. The value function *vπ* is the unique
solution to its Bellman equation.

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/backupdiagram.png"
alt="Update rule" title="Update rule" width="158" height="122" />
</p>

We call diagrams like that above *backup diagrams* because they diagram
relationships that form the basis of the update or *backup* operations that are
at the heart of RL methods. These operations transfer value information *back*
to a state (or state-action pair) from its successor states (or state-action
pairs)

The value of a state depends on the values of the actions possible in that state
and on how likely each action is to be taken under the current policy:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/backupdiagramv.png"
alt="Update rule" title="Update rule" width="424" height="88" />
</p>

The value of an action, *qπ(s,a)*, depends on the expected next reward and the
expected sum of the remaining rewards.

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/backupdiagramq.png"
alt="Update rule" title="Update rule" width="338" height="98" />
</p>

## Optimal Policies and Optimal Value Functions

Solving a reinforcement learning task means, roughly, finding a policy that
achieves a lot of reward over the long run. A policy *π* is defined to be better
than or equal to a policy *π'* if its expected return is greater than or equal
to that of *π'* for all states. *π* >= *π'* if *vπ(s)* >= *vπ'(s)*. This is
called an *optimal policy* and is denoted *π\**. They may be more than one
optimal policy and all share the same state-value function and action-value
function called *optimal state-value function* and *optimal action-value
function* denoted *v\** and *q\**

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/vstar.png"
alt="Update rule" title="Update rule" width="134" height="31" />
</p>
<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/qstar.png"
alt="Update rule" title="Update rule" width="163" height="34" />
</p>

For the state–action pair *(s,a)*, the *q\** function gives the expected return
for taking action *a* in state *s* and thereafter following an optimal policy.

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/qstarvstar.png"
alt="Update rule" title="Update rule" width="328" height="31" />
</p>

The optimal action-value function gives the values after committing to a
particular first action, but afterward using whichever actions are best.

The Bellman equation for *v\** and *q\** can be written in special form without
reference to any specific policy. This is the *Bellman optimality equation*

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/bellmanv.png"
alt="Update rule" title="Update rule" width="354" height="178" />
</p>
<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/bellmanq.png"
alt="Update rule" title="Update rule" width="438" height="95" />
</p>

The backup diagram for *v\** and *q\** are the same as for *vπ* and *qπ* except
that arcs have been added at the agent’s choice points to represent that the
maximum over that choice is taken rather than the expected value given some
policy.

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/backupdiagramqvstar.png"
alt="Update rule" title="Update rule" width="398" height="134" />
</p>

Solving the Bellman optimality equation provides one route to finding an
optimal policy, and thus to solving the reinforcement learning problem.
However, this solution is rarely directly useful. This solution relies on at
least three assumptions that are rarely true in practice: (1) we accurately
know the dynamics of the environment; (2) we have enough computational
resources to complete the computation of the solution; and (3) the Markov
property. For the kinds of tasks in which we are interested, one is generally
not able to implement this solution exactly because various combinations of
these assumptions are violated. Many reinforcement learning methods can be
clearly understood as approximately solving the Bellman optimality equation,
using actual experienced transitions in place of knowledge of the expected
transitions.

The online nature of reinforcement learning makes it possible to approximate
optimal policies in ways that put more e↵ort into learning to make good
decisions for frequently encountered states, at the expense of less e↵ort for
infrequently encountered states. This is one key property that distinguishes
reinforcement learning from other approaches to approximately solving MDPs.
