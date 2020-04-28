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
