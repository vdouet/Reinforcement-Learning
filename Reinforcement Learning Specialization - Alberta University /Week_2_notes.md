# Week 2 - Finite Markov Decision Processes (MDP)

## Definitions

**MDP**: Classical formalization of sequential decision making, where actions
influence not just immediate rewards, but also subsequent situations, or states,
and through those future rewards. MDPs involve delayed reward and the need to
tradeoff immediate and delayed reward.  
Straightforward framing of the problem of learning from interaction to achieve a
goal.  
In MDPs we estimate the value q\*(s,a) of each action *a* in each state *s*, or
we estimate the value v\*(s) of each state given optimal action selections.

**Finite MDP**: Finite number of elements for the sets of states, actions and
rewards (*S*, *A*, and *R*).

**Agent**: Learner and decision maker.

**Agent & Environment**: The agent is the learner and decision maker. The
environment is everything the agent interact with. Anything that cannot be
changed arbitrarily by the agent is considered to be outside of it and thus
part of its environment. We do not assume that everything in the environment is
unknown to the agent. In some cases the agent may know everything about how its
environment works.
The agent–environment boundary represents the limit of the agent’s absolute
control, not of its knowledge and can be located at different places for
different purposes. In practice, the agent–environment boundary is determined
once one has selected particular states, actions, and rewards, and thus has
identified a specific decision making task of interest.

## Agent–Environment Interface

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/agentenvironmentinterface.png"
alt="Update rule" title="Update rule" width="377" height="136" />
</p>

The agent select actions and the environment responds to these actions and
presents new situations to the agent. At each time step *t* the agent receives
some representation of the environment's *state* *St* and selects an *action*
*At* on that basis. One time step later the agent receives a numerical *reward*
*Rt+1* as a consequence of its action and finds itself in a new state *St+1*.  
The MDP and agent give rise to a sequence or *trajectory*: *S0, A0, R1, S1,
A1, R2, S2, A2, R3,* ...  


**_Attention_**: We use *Rt+1* instead of *Rt* to denote the reward due to
*At* because it emphasizes that the next reward and next state, *Rt+1* and
*St+1*, are jointly determined. Unfortunately, both conventions are widely used
in the literature.

In *finite* MDP the random variables *St* and *Rt* have well defined discrete
probability distribution dependent only on the preceding state and action.

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/MDPdynamics.png"
alt="Update rule" title="Update rule" width="341" height="28" />
</p>

The function *p* defines the *dynamics* of the MDP. This function is an
ordinary deterministic function of four arguments. It reminds us that *p*
specifies a probability distribution for each choice of *s* and *a*.

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/pfunction.png"
alt="Update rule" title="Update rule" width="313" height="41" />
</p>

In MDP the probabilities given by *p* completely characterize the environment’s
dynamics. That is, the probability of each possible value for *St* and *Rt*
depends only on the immediately preceding state and action, *St-1* and *At-1*,
and, given them, not at all on earlier states and actions.  
This is viewed as a restriction on the *state*, it must include information
about all aspects of the past agent–environment interaction that make a
difference for the future. The state is then said to have the *Markov property*.

From the dynamic function *p* we can compute everything we want to know about
the environment such as the *state-transition probabilities*, the expected
rewards for state-action pairs and the expected rewards for
state-action-next-state triples.

The MDP framework can be applied to many different problems. For example the
time-step doesn't need to refer to fix intervals of real-time but can refer to
arbitrary successive stages of decision making and acting. The actions can be
voltages, whether or not to have lunch or can control what an agent choose to
think about. The states can be determined by direct sensor readings, symbolic
description of objects in a room or can be be based on memory of past
sensations.  
In general, actions can be any decisions we want to learn how to make, and the
states can be anything we can know that might be useful in making them.
 
The MDP framework is a considerable abstraction of the problem of goal-directed
learning from interaction. It proposes that any problem of learning
goal-directed behavior can be reduced to three signals passing back and forth
between an agent and its environment: one signal to represent the choices made
by the agent (the actions), one signal to represent the basis on which the
choices are made (the states), and one signal to define the agent’s goal (the
rewards).
