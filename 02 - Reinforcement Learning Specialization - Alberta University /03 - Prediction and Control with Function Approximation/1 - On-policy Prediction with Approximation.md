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
weights in all the layers.