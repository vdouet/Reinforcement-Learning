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
alt="Update rule" title="Update rule" width="642" height="201" />
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
within the same tile, proportional to the number of tiles in common, and 
nonexistent to states outside it. To get true coarse coding with tile coding, 
multiple tilings are used, each offset by a fraction of a tile width.

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
one-trial learning (*v̂(s,*__w__*t+1) = v*). Usually one wishes to change more 
slowly than this, to allow for generalization and stochastic variation in 
target outputs. For example *α* = *1/10n* in which case the estimate for the 
trained state would move one-tenth of the way to the target in one update, and 
neighboring states will be moved less, proportional to the number of tiles they
have in common. Tile coding also gains computational advantages from its use of
binary feature vectors.

The choice of how to offset the tilings from each other affects generalization.
Asymmetrical offsets are preferred in tile coding because they are all well 
centered on the trained state with no obvious asymmetries.

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/tilecodingoffset.png"
alt="Update rule" title="Update rule" width="487" height="393" />
</p>

Shown is the strength of generalization from a trained state, indicated by the 
small black plus, to nearby states, for the case of eight tilings. If the 
tilings are uniformly offset (above), then there are diagonal artifacts and 
substantial variations in the generalization, whereas with asymmetrically 
offset tilings the generalization is more spherical and homogeneous.

For a two-dimensional space, we say that each tiling is offset by the 
*displacement vector* meaning that it is offset from the previous tiling by 
*w/n* times this vector, *w* denotes the tile width and *n* the number of 
tilings.  
For a continuous space of dimension *k*, a good *displacement vectors* choice 
is to use the first odd integers (1,3,5,7,...,2*k*-1), with *n* (the number of
tilings) set to an integer power of 2 greater than or equal to 4*k*.

In choosing a tiling strategy, one has to pick the number of the tilings and 
the shape of the tiles. The number of tilings, along with the size of the 
tiles, determines the resolution or fineness of the asymptotic approximation.
Tilings don't need to be grids. They can be arbitrarily shaped and non-uniform,
while still in many cases being computationally efficient to compute.
The shape of the tiles will determine the nature of generalization:
+ Square tiles will generalize roughly equally in each dimension.
+ Tiles that are elongated along one dimension, such as the stripe tilings, 
will promote generalization along that dimension.
+ Diagonal stripe tiling will promote generalization along one diagonal.

In practice, it is often desirable to use different shaped tiles in different 
tilings. With multiple tilings, some horizontal, some vertical, and some 
conjunctive, one can get everything: a preference for generalizing along each 
dimension, yet the ability to learn specific values for conjunctions.

Another useful trick for reducing memory requirements is *hashing*, a 
consistent pseudo-random collapsing of a large tiling into a much smaller set 
of tiles. Hashing produces tiles consisting of noncontiguous, disjoint regions 
randomly spread throughout the state space, but that still form an exhaustive 
partition. Through hashing, memory requirements are often reduced by large 
factors with little loss of performance. This is possible because high 
resolution is needed in only a small fraction of the state space. Hashing frees
us from the curse of dimensionality in the sense that memory requirements need 
not be exponential in the number of dimensions, but need merely match the real 
demands of the task.

### Radial Basis Functions

Radial basis functions (RBFs) are the natural generalization of coarse coding 
to continuous-valued features. Rather than each feature being either 0 or 1, it
can be anything in the interval [0, 1], reflecting various *degrees* to which 
the feature is present. A typical RBF feature, *xi*, has a Gaussian 
(bell-shaped) response *xi(s)* dependent only on the distance between the 
state, *s*, and the feature’s prototypical or center state, *ci*, and relative 
to the feature’s width, *σi*:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/rbf.png"
alt="Update rule" title="Update rule" width="199" height="58" />
</p>

The norm or distance metric of course can be chosen in whatever way seems most 
appropriate to the states and task at hand.

The primary advantage of RBFs over binary features is that they produce 
approximate functions that vary smoothly and are differentiable but in most 
cases it has no practical significance. It requires substantial additional 
computational complexity (over tile coding) and often reduce performance when 
there are more than two state dimensions. In high dimensions the edges of tiles
are much more important, and it has proven difficult to obtain well controlled 
graded tile activations near the edges.

An RBF *network* is a linear function approximator using RBFs for its features.
Some learning methods for RBF networks change the centers and widths of the 
features as well, bringing them into the realm of nonlinear function 
approximators. Nonlinear methods may be able to fit target functions much more 
precisely. The downside to RBF networks, and to nonlinear RBF networks 
especially, is greater computational complexity and, often, more manual tuning 
before learning is robust and efficient.

## Selecting Step-Size Parameters Manually

Most SGD methods require the designer to select an appropriate step-size 
parameter *α*. Slowly decreasing step-size sequence are sufficient to guarantee
convergence (stochastic approximation theory), but these tend to result in 
learning that is too slow.  
The classical choice *αt = 1/t*, which produces sample averages in tabular MC 
methods, is not appropriate for TD methods, for nonstationary problems, or for 
any method using function approximation.  
For linear methods, there are recursive least-squares methods that set an 
optimal *matrix* step size, and these methods can be extended to TD learning,
but these require *O(d^2)* step-size parameters, or *d* times more parameters 
than we are learning.  
For linear function approximation, if you wanted to learn in about *τ* 
experiences with substantially the same feature vector, a good  rule of thumb 
for setting the step-size parameter of linear SGD methods is:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/sgdstepsize.png"
alt="Update rule" title="Update rule" width="150" height="46" />
</p>

Where **x** is a random feature vector chosen from the same distribution as 
input vectors will be in the SGD. This method works best if the feature vectors
do not vary greatly in length; ideally **x**T**x** is a constant.

## Nonlinear Function Approximation: Artificial Neural Networks

Artificial neural networks (ANNs) are widely used for nonlinear function 
approximation.

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/ann.png"
alt="Update rule" title="Update rule" width="288" height="227" />
</p>

This feedforward neural network is composed of 1 input layer with 4 input units
2 hidden layers and 1 output layer with 2 output units. A real-valued weight is
associated with each link. A weight roughly corresponds to the efficacy of a 
synaptic connection in a real neural network. If an ANN has at least one loop 
in its connections, it is a recurrent rather than a feedforward ANN.

The units are typically semi-linear units, they compute a weighted sum of their
input signals and then apply to the result a nonlinear function, called the 
*activation function* to produce the unit’s output, or activation. The 
activation of each output unit of a feedforward ANN is a nonlinear function of 
the activation patterns over the network’s input units. The functions are 
parameterized by the network’s connection weights.

Training the hidden layers of an ANN is a way to automatically create features 
appropriate for a given problem so that hierarchical representations can be 
produced without relying exclusively on hand-crafted features.  
In common supervised learning case, the objective function is the expected 
error, or loss, over a set of labeled training examples. In RL, ANNs can use TD
errors to learn value functions, or they can aim to maximize expected reward as
in a gradient bandit or a policy-gradient algorithm.

To update weights in an ANN, backpropagation is used which consists of 
alternating forward and backward passes through the network. Each forward pass 
computes the activation of each unit given the current activations of the 
network’s input units. After each forward pass, a backward pass efficiently 
computes a partial derivative for each weight. 

We can use Deep Neural Networks as a function approximator to represent:
+ Policy
+ Value function
+ Model

We need to choose a corresponding loss function e.g.:
+ Policy gradient (for policy-based RL)
+ TD error (for value-based RL)
+ Next-step prediction error (for model-based RL)

We optimise this loss function by gradient descent.

Ex: We can use one-hot encoding of the current state number as an input to our
ANN and its output will be the estimated state value:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/02%20-%20Reinforcement%20Learning%20Specialization%20-%20Alberta%20University%20/Images/ann2.png"
alt="Update rule" title="Update rule" width="451" height="338" />
</p>

networks are powerful function approximators capable of representing a wide 
class of functions. They are also capable of producing features without 
exclusively relying on hand-crafted mechanisms. On the other hand, compared to 
a linear function approximator with tile-coding, neural networks can be less 
sample efficient.

## Least-Squares TD

The *Least-Squares TD algorithm* or *LSTD* directly compute the TD fixed point.
This algorithm is the most data efficient form of linear TD(0), but it is also 
more expensive computationally.

Whether the greater data efficiency of LSTD is worth the computational expense 
depends on how large *d* is, how important it is to learn quickly, and the 
expense of other parts of the system. LSTD does not require a step size, but it
does requires *ε*:
+ if *ε* chosen too small the sequence of inverses can vary wildly.
+ if *ε* is chosen too large then learning is slowed. 
In addition, LSTD’s lack of a step-size parameter means that it never forgets. 
This is sometimes desirable, but it is problematic if the target policy π 
changes as it does in RL and GPI. In control applications, LSTD typically has 
to be combined with some other mechanism to induce forgetting, mooting any 
initial advantage of not requiring a step-size parameter.