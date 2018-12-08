# Reinforcement-Learning-Project

Implemented 4 papers Deep Q Learning, Double Deep Q Learning, Prioritized DDQN and Deep Recurrent Q-Networks. 

# Part 1 

All the above 4 algorithms are first tested on the cartpole problem

# Part 2

Importance of a recurrent layer is shown in case of POMDPs by first converting the CartPole problem into an POMDP by hiding the cart velocity and pole tip velocity. Now DDQN and DRQN are run on this POMDP which successfully satisfies our claim that DRQN is still able to learn whereas DDNQ fails.

# Part 3

All the four models are run on Pacman and Breakout atari environments. Sadly, we are unable to produce any results. However the reason for this is that we didn't get a chance to train our model for longenough. Training is required atleast for a million frames inorder to get  reasonable results on Atari.
