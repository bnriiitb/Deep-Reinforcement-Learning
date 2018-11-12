[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1 : Navigation
## Nagaraju Budigam

### Project Details

In this project, I have trained an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.


### Learning Algorithm

I used [Deep Double Q-learning Algorithm] to develop an agent to interact with the environemtn and learn. It has been shown that Deep Double Q-learning algorithm not only offers better performance but also reduces the observed overestimation. Please find the implementation in [doubledqn.py](doubledqn.py).

References:
1. https://arxiv.org/abs/1509.06461
2. https://github.com/dusty-nv/jetson-reinforcement
