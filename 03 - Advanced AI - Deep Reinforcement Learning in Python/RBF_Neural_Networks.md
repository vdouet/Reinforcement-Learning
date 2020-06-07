# RBF Neural Networks

RBF = Radial Basis Function

2 perspectives:
+ Linear model with feature extraction, where the feature extraction is RBF
kernel
+ 1-hidden layer Neural Network, with RBF kernel as activation function

The equation is:

<p align="center">
<img
src="https://github.com/vdouet/Reinforcement-Learning/blob/master/03%20-%20Advanced%20AI%20-%20Deep%20Reinforcement%20Learning%20in%20Python/Images/RBF_function.png"
alt="RBF function" title="RBF function" width="204" height="60" />
</p>

Where:
+ It is a non-normalized Gaussian
+ x is the input vector
+ c is the center / exemplar vector
+ Only depends on distance between x and c, not direction, hence the term 
"radial".
+ Max is 1, when x == c, approaches O as x goes further away from c.

## From "Reinforcement Learning - An introduction" by Richard S. Sutton

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