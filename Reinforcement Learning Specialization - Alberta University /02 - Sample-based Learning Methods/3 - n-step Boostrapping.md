# n-step Bootstrapping

Neither MC methods nor one-step TD methods are always the best. We see *n-step
TD methods* that generalize both methods so that one can shift from one to the 
other smoothly as needed to meet the demands of a particular task. *n*-step 
methods span a spectrum with MC methods at one end and one-step TD methods at 
the other. The best methods are often intermediate between the two extremes.
*n*-step methods enable bootstrapping to occur over multiple steps, freeing us
from the tyranny of the single time step.

## n-step TD Prediction

The methods that use *n*-step updates are still TD methods because they still 
change an earlier estimate based on how it differs from a later estimate. Now 
the later estimate is not one step later, but *n* steps later. It can be used
with any number of timestep, the target for a two-step update is the *two-step
return*:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/2stepreturn.png"
alt="Update rule" title="Update rule" width="294" height="34" />
</p>

Similarly, the target for an arbitrary *n*-step update is the *n-step return*:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/nstepreturn.png"
alt="Update rule" title="Update rule" width="442" height="34" />
</p>

Note that *n*-step returns for *n* > 1 involve future rewards and states that 
are not available at the time of transition from *t* to *t+1*. The natural 
state-value learning algorithm for using *n*-step returns is:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/vnstep.png"
alt="Update rule" title="Update rule" width="481" height="31" />
</p>

Note that no changes at all are made during the first *n-1* steps of each 
episode. To make up for that, an equal number of additional updates are made at
the end of the episode, after termination and before starting the next episode.
An important property of the *n*-step returns is that the worst error of the 
expected *n*-step return is guaranteed to be less than or equal to γ^n times 
the worst error under *Vt+n-1*. This is called the *error reduction property*

## n-step Sarsa

For *n*-step Sarsa we redefine *n*-step returns (update targets) in terms of 
estimated action values:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/nsarsaupdate.png"
alt="Update rule" title="Update rule" width="623" height="30" />
</p>

The natural algorithm is then:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/nstepsarsaq.png"
alt="Update rule" title="Update rule" width="449" height="33" />
</p>

While the values of all other states remain unchanged: *Qt+n(s, a)* = 
*Qt+n-1(s, a)*, for all *s*, *a* such that *s* =/= *St* or *a* =/= *At*.
Note, *inf*-step Sarsa is equal to Monte Carlo.

For *n*-step Expected Sarsa, the *n*-step return is redifined as:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/nstepexpectedsarsaq.png"
alt="Update rule" title="Update rule" width="379" height="30" />
</p>

*¯V* is the *expected approximate value* of state *s*, using the estimated 
action value a time *t*, under the target policy.

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/vnexpectedsarsa.png"
alt="Update rule" title="Update rule" width="189" height="42" />
</p>

If *s* is terminal, then its expected approximate value is defined to be 0.

## n-step Off-policy Learning

For *n*-step off-policy methods the *importance sampling ratio* is calculated 
only over the *n* actions. For example, to make a simple off-policy version of 
*n*-step TD:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/offpolicynsteptd.png"
alt="Update rule" title="Update rule" width="421" height="27" />
</p>
<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/offpolicynsteptd2.png"
alt="Update rule" title="Update rule" width="206" height="63" />
</p>

If any one of the actions would never be taken by *π*, then the n-step return 
should be given zero weight and be totally ignored. 

Our previous *n*-step Sarsa update can be completely replaced by a simple 
off-policy form:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/nstepsarsaoffpolicy.png"
alt="Update rule" title="Update rule" width="506" height="30" />
</p>

