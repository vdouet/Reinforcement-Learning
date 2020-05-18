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