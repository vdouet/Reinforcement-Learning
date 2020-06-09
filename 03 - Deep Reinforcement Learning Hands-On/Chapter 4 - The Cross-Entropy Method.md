# The Cross-Entropy Method

The cross-entropy method is model-free, policy-based, and on-policy meaning:
+ It doesn't build any model of the environment; it just says to the agent
what to do at every step.
+ It approximates the policy of the agent.
+ It requires fresh data obtained from the environment.

As our cross-entropy method is policy-based our nonlinear function (neural
network (NN)) produces the policy.

Observaton s ==> Trainable functio (NN) ==> Policy

The core of the cross-entropy method is to throw away bad episodes and train 
on better ones:

1. Play N number of episodes using our current model and environment
2. Calculate the total reward for every episode and decide on a reward boundary.
Usually, we use some percentile of all rewards, such as 50th or 70th.
3. Throw away all episodes with a reward below the boundary.
4. Train on the remaining "elite" episodes using observations as the input and 
issued actions as the desired output.
5. Repeat from step 1 until we become satisfied with the result.

Despite the simplicity of this method, it works well on simple environments,
it is easy to implement and quite robust to hyperparameters changing.