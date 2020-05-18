# On-policy Control with Approximation

## Episodic Semi-gradient Control

For semi-gradient prediction with action-value the random training examples is
now of the form *St, At -> Ut*. The update target *Ut* can be any approximation
of *qπ(St, At)*. The general gradient-descent update for action-value 
prediction is:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/actionvaluesemigradient.png"
alt="Update rule" title="Update rule" width="363" height="38" />
</p>

For example, the update for the one-step Sarsa method is:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/semigradientonestepsarsa.png"
alt="Update rule" title="Update rule" width="538" height="39" />
</p>

We call this method *episodic semi-gradient one-step Sarsa*. For a constant 
policy, this method converges in the same way that TD(0) does. To form control 
methods, we need to couple such action-value prediction methods with techniques
for policy improvement and action selection.

We can approximate the action-value function the same way as for the value 
function: 

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/linearactionvalue.png"
alt="Update rule" title="Update rule" width="303" height="64" />
</p>

## Semi-gradient n-step Sarsa

We can obtain an *n*-step version of episodic semi-gradient Sarsa by using an 
*n*-step return as the update target in the semi-gradient Sarsa update 
equation:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/nstepsarsagt.png"
alt="Update rule" title="Update rule" width="490" height="31" />
</p>

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/nstepsarsasgdupdate.png"
alt="Update rule" title="Update rule" width="507" height="26" />
</p>

## Average Reward: A New Problem Setting for Continuing Tasks

Like the discounted setting, the *average reward* setting applies to continuing
problems. However, there is no discounting the agent cares just as much about 
delayed rewards as it does about immediate reward. The discounted setting is 
problematic with function approximation, and thus the average-reward setting is
needed to replace it.

In the average-reward setting, the quality of a policy π is defined as the 
average rate of reward, or simply *average reward*, while following that 
policy, which we denote as *r(π)*:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/averagereward.png"
alt="Update rule" title="Update rule" width="339" height="140" />
</p>

*µπ(s)* needs to be independant of *S0*. This assumption about the MDP is known
as *ergodicity*. It means that where the MDP starts or any early decision made 
by the agent can have only a temporary effect; in the long run the expectation 
of being in a state depends only on the policy and the MDP transition 
probabilities.

We consider all policies that attain the maximal value of *r(π)* to be optimal.
In the average-reward setting, returns are defined in terms of differences
between rewards and the average reward:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/averagerewardreturns.png"
alt="Update rule" title="Update rule" width="448" height="38" />
</p>

This is known as the *differential return*, and the corresponding value 
functions are known as *differential* value functions. They are defined in the 
same way and we will use the same notation for them as we have all along.  
Differential vlue functions have slightly different Bellman equations:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/differentialbellman.png"
alt="Update rule" title="Update rule" width="426" height="234" />
</p>

There is also a differential form of the two TD errors where *R¯t* is an 
estimate at time *t* of the average reward *r(π)*:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/differentialtd.png"
alt="Update rule" title="Update rule" width="402" height="74" />
</p>

## Deprecating the Discounted Setting