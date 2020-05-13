# Planning and Learning with Tabular Methods

**Model-based**: Reinforcement learning methods that require a model of the 
environment, such as dynamic programming and heuristic search. Model-based 
methods rely on *planning* as their primary component.

**Model-free**: Methods that can be used without a model, such as Monte Carlo 
and temporal-di↵erence methods. Model-free methods primarily rely on *learning*.

**Model of the environment**: Anything that an agent can use to predict how the
environment will respond to its actions.

## Models and Planning

Given a state and an action, a model produces a prediction of the resultant 
next state and next reward. Some models produce a description of all 
possibilities and their probabilities; these we call *distribution models*.
Other models produce just one of the possibilities, sampled according to the 
probabilities; these we call *sample models*.