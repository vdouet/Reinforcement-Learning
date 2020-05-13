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

## Per-decision Methods with Control Variates

The multi-step off-policy methods presented in the previous section are simple
and conceptually clear, but are probably not the most efficient. A more 
sophisticated approach would use per-decision importance sampling ideas seen
previously.  
For the *n* steps ending at horizon *h*, the *n*-step return can be written (h
was previously denoted *t+n*):

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/nstepreturnh.png"
alt="Update rule" title="Update rule" width="176" height="24" />
</p>

In this more sophisticated approach, we can use an alternate, off-policy 
definition of the *n*-step return ending at horizon *h*, as:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/nstepperdecision.png"
alt="Update rule" title="Update rule" width="340" height="31" />
</p>

In this approach, if *pt* is zero, then instead of the target being zero and 
causing the estimate to shrink, the target is the same as the estimate and 
causes no change. The importance sampling ratio being zero means we should 
ignore the sample, so leaving the estimate unchanged seems appropriate. The
additionnal term on the right side is called a *control variate*.  
For a conventional *n*-step method, the learning rule to use in conjunction
with the previous *n*-step return is the *n*-step TD update, which has no 
explicit importance sampling ratios other than those embedded in the return.

For action-value, the *n*-step *off-policy* return can be written with control
variates as:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/offpolicyreturncontrolvariates.png"
alt="Update rule" title="Update rule" width="573" height="70" />
</p>

## Off-policy Learning Without Importance Sampling: The n-step Tree Backup Algorithm

The *n-step tree backup algorithm* estimate the value of the node at the top of
the diagram toward a target combining the rewards along the way (appropriately 
discounted) and the estimated values of the nodes at the bottom, as previously
done. But it also add the estimated values of the action not selected at all 
level. It is an update from the entire tree of estimated action values. More 
precisely, the update is from the estimated action values of the leaf nodes of 
the tree. The action nodes in the interior, corresponding to the actual actions
taken, do not participate. Each leaf node contributes to the target with a 
weight proportional to its probability of occurring under the target policy
*π*.

The general recursive definition of the tree-backup *n*-step return:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/nsteptreereturn.png"
alt="Update rule" title="Update rule" width="556" height="49" />
</p>

This target is then used with the usual action-value update rule from *n*-step
Sarsa:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/nsteptreeupdate.png"
alt="Update rule" title="Update rule" width="451" height="30" />
</p>

While the values of all other state–action pairs remain unchanged: *Qt+n(s, a)*
= *Qt+n-1(s, a)*, for all *s*, *a* such that *s =/= St* or *a =/= At*.

## A Unifying Algorithm: n-step Q(σ)

*n*-step Sarsa has all sample transitions, the tree-backup algorithm has all 
state-to-action transitions fully branched without sampling, and *n*-step 
Expected Sarsa has all sample transitions except for the last state-to-action 
one, which is fully branched with an expected value. 

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/nstepdiagram.png"
alt="Update rule" title="Update rule" width="450" height="332" />
</p>

*ρ* indicates half transitions on which importance sampling is required in the
off-policy case. *σ* is if we choose to sample (*σt* = 1) or not (*σt* = 0)

To unifiy all of those algorithm, we can decide on a step-by-step basis whether 
to take the action as a sample, as in Sarsa, or consider the expectation over 
all actions instead, as in the tree-backup update. If we chose always to 
sample, we would obtain Sarsa, whereas if we chose never to sample, we would 
get the tree-backup algorithm. Expected Sarsa would be the case where we chose 
to sample for all steps except for the last one.

We can consider a continuous variation between sampling and expectation. The 
random variable *σt* might be set as a function of the state, action, or 
state–action pair at time *t*.

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/Qsigma.png"
alt="Update rule" title="Update rule" width="624" height="66" />
</p>

Then we use the general (off-policy) update for *n*-step Sarsa.