"""
References:
1. https://arxiv.org/abs/1509.06461
2. https://github.com/dusty-nv/jetson-reinforcement
"""

import argparse
import random
import numpy as np
import logging
import pickle
from collections import namedtuple, deque
from unityagents import UnityEnvironment

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("doubledqn")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='DobleDQN')
parser.add_argument('--unity_environmet_file_path', default='/Users/naga/GitHub/Deep-Reinforcement-Learning/p1_navigation/Banana.app',
                    help='Unity environment')
parser.add_argument('--pkl_file_name', default='scores.pkl', help='Stats output file path')
parser.add_argument('--learning_rate', type=float, default=0.001, metavar='N', help='optimizer learning rate')
parser.add_argument('--replay_mem', type=int, default=10000, metavar='N', help='replay memory')
parser.add_argument('--num_history', type=int, default=4, metavar='N', help='num history')
parser.add_argument('--num_episodes', type=int, default=600, metavar='N', help='num episodes')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size')
parser.add_argument('--update_every', type=int, default=4, metavar='N', help='update every')
parser.add_argument('--gamma', type=float, default=0.99, metavar='N',
                    help='discount factor for present rewards vs. future rewards')
parser.add_argument('--tau', type=float, default=1e-3, metavar='N', help='for soft update of target parameters')
parser.add_argument('--epsilon_start', type=float, default=1.0, metavar='N', help='epsilon_start of random actions')
parser.add_argument('--epsilon_end', type=float, default=0.01, metavar='N', help='epsilon_end of random actions')
parser.add_argument('--epsilon_decay', type=float, default=0.995, metavar='N',
                    help='exponential decay of random actions')
parser.add_argument('--allow_random', type=int, default=1, metavar='N', help='Allow DQN to select random actions')
parser.add_argument('--debug_mode', type=int, default=0, metavar='N', help='debug mode')
parser.add_argument('--training_mode', type=int, default=1, metavar='N', help='training mode')

args = parser.parse_args()


input_width = 37
num_actions = 4

unity_environmet_file_path = args.unity_environmet_file_path
pkl_file_name = args.pkl_file_name
learning_rate = args.learning_rate
replay_mem = args.replay_mem
num_history = args.num_history
num_episodes = args.num_episodes
batch_size = args.batch_size
update_every = args.update_every
gamma = args.gamma
tau = args.tau
epsilon_start = args.epsilon_start
epsilon_end = args.epsilon_end
epsilon_decay = args.epsilon_decay
allow_random = args.allow_random
debug_mode = args.debug_mode
training_mode = args.training_mode

logger.info('input_width:    ' + str(input_width))
logger.info('num_actions:    ' + str(num_actions))
logger.info('unity_environmet_file_path:' + str(unity_environmet_file_path))
logger.info('pkl_file_name:  ' + str(pkl_file_name))
logger.info('learning rate:  ' + str(learning_rate))
logger.info('replay_memory:  ' + str(replay_mem))
logger.info('num_history:    ' + str(num_history))
logger.info('num_episodes:   ' + str(num_episodes))
logger.info('batch_size:     ' + str(batch_size))
logger.info('update_every:   ' + str(update_every))
logger.info('gamma:          ' + str(gamma))
logger.info('tau:            ' + str(tau))
logger.info('epsilon_start:  ' + str(epsilon_start))
logger.info('epsilon_end:    ' + str(epsilon_end))
logger.info('epsilon_decay:  ' + str(epsilon_decay))
logger.info('allow_random:   ' + str(allow_random))
logger.info('debug_mode:     ' + str(debug_mode))
logger.info('training_mode:  ' + str(training_mode))


class ReplayBuffer:

    def __init__(self):
        self.action_size = num_actions
        self.memory = deque(maxlen=replay_mem)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([[e.state] for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([[e.next_state] for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class HistoryBuffer(object):

    def __init__(self):
        self.history = deque(maxlen=num_history)

    def reset(self, state):
        for i in range(num_history):
            self.append(state)

    def append(self, state):
        if state is None:
            raise ValueError('state should not be None')
        self.history.append(state)

    def get_state(self):
        return np.vstack([s for s in self.history]).reshape(num_history, -1)


class DQN(nn.Module):

    def __init__(self):
        logging.info('DQN::__init__()')
        super(DQN, self).__init__()
        self.conv1 = nn.Conv1d(num_history, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, 16, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, num_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent(object):

    def __init__(self):
        self.qnetwork_local = DQN().to(device)
        self.qnetwork_target = DQN().to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer()
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % update_every
        if self.t_step == 0:
            if len(self.memory) > batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, gamma)

    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(num_actions))

    def best_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.smooth_l1_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


env = UnityEnvironment(file_name=unity_environmet_file_path)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
print('Number of agents:', len(env_info.agents))
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)


def dqn():
    agent = Agent()
    scores = []
    scores_window = deque(maxlen=100)
    eps = epsilon_start
    scores_window_mean = []
    for i_episode in range(1, num_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        history = HistoryBuffer()
        history.reset(state)
        history_state = history.get_state()
        score = 0
        while True:
            action = agent.act(history_state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            history.append(next_state)
            history_next_state = history.get_state()
            agent.step(history_state, action, reward, history_next_state, done)
            history_state = history_next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(epsilon_end, epsilon_decay * eps)
        print('\rEpisode {}\tAverage Score: {:.2f}\tEps: {:.3f}'.format(i_episode, np.mean(scores_window), eps), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            scores_window_mean.append(np.mean(scores_window))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_solved.pth')
    return scores, scores_window_mean


def dqn_test():
    agent = Agent()
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint_solved.pth', map_location='cpu'))
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    history = HistoryBuffer()
    history.reset(state)
    history_state = history.get_state()
    score = 0
    while True:
        action = agent.best_action(history_state)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        history.append(next_state)
        history_state = history.get_state()
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward
        if done:
            break
    print("Score: {}".format(score))


if not debug_mode:
    if training_mode:
        logging.info('Training')
        scores, scores_window_mean = dqn()
        output_dict = {'scores': scores, 'scores_window_mean': scores_window_mean}
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(output_dict, f)
    else:
        logging.info("Testing")
        dqn_test()

if debug_mode:
    logging.info("Stats and figures")
    with open(pkl_file_name, 'rb') as f:
        input_dict = pickle.load(f)
    scores = input_dict['scores']
    scores_window_mean = input_dict['scores_window_mean']

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    plt.plot(np.arange(len(scores)), scores)
    ax1.set_ylabel('Score')
    ax1.set_xlabel('Episode #')
    ax1.grid()
    ax2 = fig.add_subplot(122)
    plt.plot(np.arange(len(scores_window_mean)), scores_window_mean)
    ax2.set_ylabel('Average Score')
    ax2.set_xlabel('Every 100th Episode')
    ax2.axis('equal')
    ax2.grid()

    plt.subplots_adjust(left=0.2, wspace=0.5, top=0.8)

    plt.savefig('scores.png')

env.close()
