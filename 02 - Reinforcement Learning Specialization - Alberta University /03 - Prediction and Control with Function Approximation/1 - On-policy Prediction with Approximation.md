# Approximate Solution Methods

In many of the tasks to which we would like to apply reinforcement learning the
state space is combinatorial and enormous. We cannot expect to find an optimal
policy or the optimal value function even in the limit of infinite time and 
data.  
It is necessary to generalize from previous encounters with different states 
that are in some sense similar to the current one.  
*Generalization*: How can experience with a limited subset of the state space 
be usefully generalized to produce a good approximation over a much larger 
subset?  
We need to combine reinforcement learning methods with existing generalization 
methods called *function approximation*.

# On-policy Prediction with Approximation

Here the approximate value function is represented not as a table but as a 
parameterized functional form with weight vector **w**. *v̂(s,**w**) ≈ vπ(s)*.
More generally, v̂ might be a linear function or the function computed by a 
multi-layer artificial neural network, with **w** the vector of connection 
weights in all the layers. v̂ might also be the function computed by a decision 
tree, where **w** is all the numbers defining the split points and leaf values 
of the tree.  
The number of weights is called the dimensionality of **w**. Typically the
dimensionality of **w** is much less than the number of states (d << |*S*|),
and changing one weight changes the estimated value of many states.  
Extending reinforcement learning to function approximation also makes it 
applicable to partially observable problems. What function approximation can’t 
do, however, is augment the state representation with memories of past 
observations.

## Value-function Approximation

Up to now, for the updates, the table entry of state *s*'s estimated value has 
simply been shifted a fraction of the way toward *u* (*update target*): 
*s -> u*, and the estimated values of all other states were left unchanged. 
Now we permit arbitrarily complex and sophisticated methods to implement the 
update, and updating at *s* generalizes so that the estimated values of many 
other states are changed as well.

Machine learning methods that learn to mimic input–output examples in this way
are called *supervised learning methods*, and when the outputs are numbers, 
like *u*, the process is often called *function approximation*.
Function approximation methods expect to receive examples of the desired 
input–output behavior of the function they are trying to approximate. We use 
these methods for value prediction simply by passing to them the *s -> g* of 
each update as a training example. We then interpret the approximate function 
they produce as an estimated value function. In principle, we can use any 
method for supervised learning from examples. However in RL it is important 
that learning be able to occur online, while the agent interacts with its 
environment or with a model of its environment. It also requires function 
approximation methods able to handle nonstationary target functions.

## The Prediction Objective (VE)

With genuine approximation we must specify a state distribution µ representing
how much we care about the error in each state *s*.  
VE is the MSVE (*Mean Square Value Error*) 

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/VE.png"
alt="Update rule" title="Update rule" width="274" height="50" />
</p>

The square root of VE gives a rough measure of how much the approximate values
differ from the true values and is often used in plots. Often µ is chosen to be
the fraction of time spent in *s*. Under on-policy training this is called the
*on-policy distribution*.

An ideal goal in terms of VE would be to find a *global optimum* sometime 
possible for simple function approximators such as linear ones. Complex 
function approximators such as artificial neural networks and decision tree may
seek to converge instead to a *local optimum*. It is typically the best for 
nonlinear function approximators, and often it is enough. Both convergence to
an optimum are not guaranteed for many cases of interest in RL.

## Stochastic-gradient and Semi-gradient Methods

Stochastic gradient descent (SGD) methods are among the most widely used of all
function approximation methods and are particularly well suited to online 
reinforcement learning.  
In gradient-descent methods, the weight vector is a column vector with a fixed 
number of real valued components and the approximate value function is a 
differentiable function of **w**. **w** is updated at each of a series of 
discrete time steps, we use **w**t for the weight vector at each step. We 
assume that states appear in examples with the same distribution, μ, over which
we are trying to minimize the VE. Here we assume that, on each step, we observe
a new example *St -> vπ(St)* consisting of a state *St* and its true value 
under the policy.  
A good strategy in this case is to try to minimize error on the observed 
examples, SGD methods do this by adjusting the weight vector after each example
by a small amount in the direction that would most reduce the error on that 
example.

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/sgd.png"
alt="Update rule" title="Update rule" width="336" height="74" />
</p>

*∇f(**w**)* for a function of the vector **w** denotes the column vector of 
partial derivatives of the expression with respect to the components of the 
vector. This is the *gradient* of *f* with respect to **w**:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/nablaf.png"
alt="Update rule" title="Update rule" width="325" height="65" />
</p>

The convergence results for SGD methods assume that *α* (the timestep - learning
rate) decreases over time. If it satisfy the standard stochastic approximation
conditions then the SGD method is guaranted to converge to a local optimum.

Now we look at the case in which the target output *Ut* of the *t*th training 
example, *St -> Ut* is not the true value, *vπ(St)*, but some, possibly random,
approximation to it. In these cases we cannot perform the exact update because
*vπ(St)* is unknown, but we can approximate it by substituting *Ut*

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/sgd2.png"
alt="Update rule" title="Update rule" width="310" height="43" />
</p>

If Ut is an *unbiased* estimate then *wt* is guaranteed to converge to a local 
optimum under the usual stochastic approximation conditions for decreasing *α*.

Because the true value of a state is the expected value of the return following
it, the Monte Carlo target *U -> Gt* is by definition an unbiased estimate of 
*vπ(St)*. Bootstrapping targets such as n-step returns or DP target all depend 
on the current value of the weight vector **w***t*, which implies that they 
will be biased and that they will not produce a true gradient-descent method.
Bootstrapping methods are not in fact instances of true gradient descent, they 
include only a part of the gradient and, accordingly, we call them 
*semi-gradient methods*. Although semi-gradient (bootstrapping) methods do not 
converge as robustly as gradient methods, they do converge reliably in 
important cases such as the linear case. They also offer important 
advantages that make them often clearly preferred. They typically enable 
significantly faster learning and enable learning to be continual and online, 
without waiting for the end of an episode. This enables them to be used on 
continuing problems and provides computational advantages.

*State aggregation* is a simple form of generalizing function approximation in 
which states are grouped together, with one estimated value for each group. It
is a special case of SGD.

## Linear Methods