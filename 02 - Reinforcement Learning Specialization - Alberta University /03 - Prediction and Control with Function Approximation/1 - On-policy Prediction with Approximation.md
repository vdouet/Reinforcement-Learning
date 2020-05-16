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
*Discrimination*: The ability to make the value of two states different.  
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

One of the most important special cases of function approximation is that in 
which the approximate function, *v̂(·,**w**)*, is a linear function of the 
weight vector, **w**. Corresponding to every state *s*, there is a real-valued 
vector __x__*(s) = (x1(s), x2(s), ..., xd(s))* with the same number of 
components as *w*. Linear methods approximate state-value function by the inner
product between *w* and *x*(s):

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/linearv.png"
alt="Update rule" title="Update rule" width="238" height="58" />
</p>

In this case the approximate value function is said to be *linear in the 
weights*, or simply *linear*. The vector **x**(s) is called a *feature vector* 
representing state *s*. For linear methods, features are *basis functions* 
because they form a linear basis for the set of approximate functions. Features
may be defined in many different ways.

The gradient of the approximate value function with respect to **w** using SGD
updates with linear function approximation is:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/gradientlinear.png"
alt="Update rule" title="Update rule" width="127" height="30" />
</p>

In the linear case the SGD update is:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/linearsgdupdate.png"
alt="Update rule" title="Update rule" width="270" height="39" />
</p>

The semi-gradient TD(0) algorithm converge to a weight vector which is a point 
near the local optimum called *TD fixed point*. The *linear semi-gradient
TD(0)* also converge to this point as well as the on-policy bootstrapping 
methods such as linear semi-gradient DP. One-step semi-gradient *action-value* 
methods, such as semi-gradient Sarsa(0) converge to an analogous fixed point.

Critical to the these convergence results is that states are updated according 
to the on-policy distribution. For other update distributions, bootstrapping 
methods using function approximation may actually diverge to infinity. 

## Feature Construction for Linear Methods

How states are represented in terms of features is critical for the convergence
and efficientness of both data and computation for linear methods. Choosing 
features appropriate to the task is an important way of adding prior domain 
knowledge to reinforcement learning systems.

A limitation of the linear form is that it cannot take into account any 
interactions between features, such as the presence of feature *i* being good 
only in the absence of feature *j*. It needs instead, or in addition, features
for combinations of these two underlying state dimensions.

### Polynomials

The states of many problems are initially expressed as numbers. In these types 
of problems, function approximation for reinforcement learning has much in 
common with the familiar tasks of interpolation and regression. Polynomials 
make up one of the simplest families of features used for interpolation and 
regression. Althought, basic polynomial features do not work as well as other
types of features in RL. It is not recommended to use polynomials for online 
learning.

### Fourier basis

Another linear function approximation method is based on the time-honored 
Fourier series, which expresses periodic functions as weighted sums of sine and
cosine basis functions (features) of different frequencies. In reinforcement 
learning, where the functions to be approximated are unknown, Fourier basis 
functions are of interest because they are easy to use and can perform well in 
a range of reinforcement learning problems. However, Fourier features have 
trouble with discontinuities because it is diffcult to avoid “ringing” around 
points of discontinuity unless very high frequency basis functions are 
included.

### Coarse Coding

In a task in which the natural representation of the state set is a continuous 
two-dimensional space. One kind of representation for this case is made up of 
features corresponding to circles in state space. If the state is inside a 
circle, then the corresponding feature has the value 1 and is said to be 
*present*; otherwise the feature is 0 and is said to be *absent*. This kind of 
1–0-valued feature is called a *binary feature*. 

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/coarsecoding.png"
alt="Update rule" title="Update rule" width="275" height="245" />
</p>

Representing a state with features that overlap in this way
(although they need not be circles or binary) is known as *coarse coding*. The 
approximate value function will be affected at all states within the union of 
the circles, with a greater effect the more circles a point has “in common” 
with the state. If the circles are small, then the generalization will be over 
a short distance, if they are large, it will be over a large distance, if the
shape of the features are not strictly circular, but are elongated in one 
direction, then generalization will be similarly affected.

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/coarsecoding2.png"
alt="Update rule" title="Update rule" width="642" height="502" />
</p>

Initial generalization from one point to another is controlled by the size and
shape of the receptive fields, but acuity, the finest discrimination ultimately
possible, is controlled more by the total number of features.

### Tile Coding

Tile coding is a form of coarse coding for multi-dimensional continuous spaces 
that is flexible and computationally efficient. It may be the most practical 
feature representation for modern sequential digital computers.

In tile coding the receptive fields of the features are grouped into partitions
of the state space. Each such partition is called a *tiling*, and each element
of the partition is called a *tile*. Generalization is done to all states 
within the same tile and nonexistent to states outside it. To get true coarse 
coding with tile coding, multiple tilings are used, each offset by a fraction 
of a tile width.

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/tilecoding.png"
alt="Update rule" title="Update rule" width="662" height="240" />
</p>

In the left of the picture, the state indicated by the white spot, falls in 
exactly one tile in each of the four tilings. These four tiles correspond to 
four features that become active when the state occurs. The feature vector 
**x**(s) has one component for each tile in each tiling. In this example there 
are 4 x 4 x 4 = 64 components, all of which will be 0 except for the four 
corresponding to the tiles that s falls within.

An advantage of tile coding is that, because it works with partitions, the 
overall number of features that are active at one time is the same for any 
state. Exactly one feature is present in each tiling, so the total number of 
features present is always the same as the number of tilings. This allows the 
step-size parameter, *α*, to be set in an easy, intuitive way. For example, 
choosing *α* = *1/n* , where *n* is the number of tilings, results in exact 
one-trial learning (*v̂(s,__w__ t+1) = v*). Usually one wishes to change more 
slowly than this, to allow for generalization and stochastic variation in 
target outputs