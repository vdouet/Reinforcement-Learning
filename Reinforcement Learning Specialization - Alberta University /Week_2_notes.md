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

**Agent–Environment Interface**: The agent select actions and the environment
responds to these actions and presents new situations to the agent.
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/agentenvironmentinterface.png"
alt="Update rule"
	title="Update rule" width="503" height="181" />  
