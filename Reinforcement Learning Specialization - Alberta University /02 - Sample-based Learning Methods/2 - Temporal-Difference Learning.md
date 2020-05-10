# Temporal-Difference Learning

## Definitions

**Temporal-Difference Learning (TD)**: TD learning is a combination of Monte 
Carlo ideas and dynamic programming (DP) ideas. TD methods can learn directly 
from raw experience without a model of the environment’s dynamics and update 
estimates based in part on other learned estimates (boostrap). TD, DP and Monte
Carlo methods can be combined in different ways.

## TD Prediction

Both TD and Monte Carlo methods use experience to solve the prediction problem.
Monte Carlo methods must wait until the end of the episode to determine the 
increment to V(St):

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/vtMC.png"
alt="Update rule" title="Update rule" width="245" height="41" />
</p>

TD methods need to wait only until the next time step:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/vtTD0.png"
alt="Update rule" title="Update rule" width="350" height="37" />
</p>

This TD method is called *TD(0)*, or *one-step* TD, because it is a special
case of the TD(λ) and *n-step* TD methods. The target for Monte Carlo update is
*Gt*, the target for the TD update is *Rt+1 + γ\*V(St+1)*. TD methods combine
the sampling of Monte Carlo with the boostrapping of DP. TD and Monte Carlo use
*sample updates*, DP uses *expected updates*. *Sample* updates are based on a 
single sample successor rather than on a complete distribution of all possible
successors.
The quantity in bracket un TD(0) update is a sort of error, measuring the 
difference between the estimated value of *St* and the better estimate. It is
called the *TD error*:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/tderror.png"
alt="Update rule" title="Update rule" width="234" height="34" />
</p>

On the next picture an example of the update difference between Monte Carlo and
TD:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/tdmcex.png"
alt="Update rule" title="Update rule" width="682" height="330" />
</p>

For the MC methods the change in prediction (red arrow) is updated offline, at
the end of the episode (arrive home) because only at this point you know the 
actual return. For the TD methods the change in prediction is done immediately
at the next step, each error is proportional to the change over time of the 
prediction, that is, to the *temporal differences* in predictions.

## Advantages of TD Prediction Methods
