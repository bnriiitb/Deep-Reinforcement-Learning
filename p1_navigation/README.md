[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

## Project Details

In this project, I have trained an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## Getting Started

Please make sure you have Python 3.6+ installed on your machine and to install the dependencies that are quuired to run this project. Please run the below command.

	conda install requirements.txt

1. Download the environment from one of the links below.  Select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

2. Please place the downloaded file in the `p1_navigation/` folder, and unzip (or decompress) the file. 



## Instructions

### Training the Agent

	python3 ddqn.py --unity_environmet_file_path=Banana.app
	

### Testing the Agent

	python3 ddqn.py --unity_environmet_file_path=Banana.app --testing_mode=1


### Report

The file [Report.md](Report.md) contains the strategies and the methodologies followed to train the agent.  

The agent was able to solve the environment in __500 episodes__
Please refer to [checkpoint_solved.pth](checkpoint_solved.pth) file for finding the saved weights.

