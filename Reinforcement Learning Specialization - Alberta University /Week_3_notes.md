# Week 3

## Definitions

**Value functions**: Functions of states (or of state–action pairs) that
estimate how good it is for the agent to be in a given state (or how good it is
to perform a given action in a given state). The notion of “how good” here is
defined in terms of future rewards that can be expected, or, to be precise, in
terms of expected return.

**Policy**: Mapping from states to probabilities of selecting each possible
action.

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
