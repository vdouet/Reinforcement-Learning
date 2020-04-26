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

**Agent**: Learner and decision maker.

**Environment**: Everything the *agent* interact with.

## Agent–Environment Interface

The agent select actions and the environment responds to these actions and
presents new situations to the agent. At each time step *t* the agent receives
some representation of the environment's *state* *St* and selects an *action*
*At* on that basis. One time step later the agent receives a numerical *reward*
*Rt+1* as a consequence of its action and finds itself in a new state *St+1*.  
The MDP and agent give rise to a sequence or *trajectory*:  
*S0 ,A0 ,R1 ,S1 ,A1 ,R2 ,S2 ,A2 ,R3 ,*...  


__**Attention**__: We use *Rt+1* instead of *Rt* to denote the reward due to
*At* because it emphasizes that the next reward and next state, *Rt+1* and
*St+1*, are jointly determined. Unfortunately, both conventions are widely used
in the literature.

<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/agentenvironmentinterface.png"
alt="Update rule"
	title="Update rule" width="377" height="136" />  
