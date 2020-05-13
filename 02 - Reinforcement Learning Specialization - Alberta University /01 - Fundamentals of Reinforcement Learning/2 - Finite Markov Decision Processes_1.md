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

**Expected return**: Cumulative rewards we want to maximize.

**Discounted expected return**: Cumulative discounted rewards we want to
maximize.

**Episodes/Trials**: Subsequences of an agent-environment interaction.

**Episodic tasks**: Tasks that can be broken into identifiable episodes.

**Continuous tasks**: Tasks that cannot be broken into identifiable episodes
and goes continually without limit.

## Agent–Environment Interface

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/agentenvironmentinterface.png"
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
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/MDPdynamics.png"
alt="Update rule" title="Update rule" width="341" height="28" />
</p>

The function *p* defines the *dynamics* of the MDP. This function is an
ordinary deterministic function of four arguments. It reminds us that *p*
specifies a probability distribution for each choice of *s* and *a*.

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/pfunction.png"
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

## Goals and Rewards

In reinforcement learning the goal of the agent is to maximize the total
amount of reward it receives. This means maximizing not immediate reward, but
cumulative reward in the long run.
The *reward hypothesis*:  
*"All of what we mean by goals and purposes can be well thought of as the
maximization of the expected value of the cumulative sum of a received scalar
signal (called reward)."*  
We must provide rewards to the agent in such a way that in maximizing them it
will also achieve our goals.  It is thus critical that the rewards we set up
truly indicate what we want accomplished. For example, a chess-playing agent
should be rewarded only for actually winning, not for achieving subgoals such
as taking its opponent’s pieces or gaining control of the center of the board.  
The reward is for communicating what to achieve not how to achieve it.

There is still some exception where some intermediate small rewards can make a
big difference in helping the agent to the right direction. Even if we accept
the reward hypothesis there is still work to do to define the right rewards.
The hypothesis should not be taken too litterally and we should be able to
reject the hypothesis when it has outlived its usefulness. For example when the
target is something other than expected cumulative reward. Or is the
maximization of cumulative rewards the best match for high-level human-like
behavior?  
Maximizing rewards might be an excellent **approximation** of what motivates
intelligent agents.

Examples of rewards:  
To make a robot learn to walk, researchers have provided reward on each time
step proportional to the robot’s forward motion. In making a robot learn how to
escape from a maze, the reward is often -1 for every time step that passes prior
to escape; this encourages the agent escape as quickly as possible.  
To make a robot learn to find and collect empty soda cans for recycling, one
might give it a reward of zero most of the time, and then a reward of +1 for
each can collected. One might also want to give the robot negative rewards when
it bumps into things or when somebody yells at it.  
For an agent to learn to play checkers or chess, the natural rewards are +1 for
winning, -1 for losing, and 0 for drawing and for all nonterminal positions.

Standard RL algorithms don't respond well to nonstationary rewards. For
example, an user who gives rewards in real-time, the rewards will not always be
the same for the same states depending on how the user reacts.

### Where do rewards come from ?

+ Programming  
\- Coding: Coding the rewards from states once and for all.  
\- Human-in-the-loop: An user can give rewards in real-time to the algorithm.

+ Examples  
\- Mimic reward: An agent learning to copy the reward that a person gives.  
\- Inverse reinforcement learning: A trainer demonstrate an example of the
desired behavior and the learner figures out what reward the trainer must have
been maximizing that makes this behavior optimal.

+ Optimization  
\- Evolutionary optimization  
\- Meta RL


## Returns and Episodes

Episode begins independently of how the previous one ended and can all be
considered to end in the same terminal state, with different rewards for the
different outcomes. Tasks with episodes of this kind are called *episodic
tasks.*  
In episodic tasks we sometimes need to distinguish the set of all nonterminal
states, denoted S, from the set of all states plus the terminal state,
denoted S\+. The time of termination, T, is a random variable that normally
varies from episode to episode.  
In many cases the agent–environment interaction does not break naturally into
identifiable episodes, but goes on continually without limit. Those are called
*continuous tasks*

Due to the infinite nature of continuous tasks we want to maximize the expected
*discounted return*:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/discountedreturn.png"
alt="Update rule" title="Update rule" width="360" height="45" />
</p>

Gamma is a parameter between 0 and 1 called the *discount rate*. It determines
the present value of future rewards. If gamma = 0 the agent is considered
"*myopic*" in being concerned only to maximize immediate rewards. As gamma
approches 1, the return objective takes future rewards into account more
strongly; the agent becomes more farsighted.

Returns at successive time steps are related to each other in a way that is
important for the theory and algorithms of reinforcement learning:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/discountedreturn2.png"
alt="Update rule" title="Update rule" width="330" height="74" />
</p>
