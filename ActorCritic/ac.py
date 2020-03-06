import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch as th
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

DOWN = 0
UP = 1
LEFT = 2
RIGHT = 3
actions = [DOWN, UP, LEFT, RIGHT]
use_cuda = True
device = th.device("cuda") if use_cuda and th.cuda.is_available() else th.device("cpu")

GAMMA = 0.9
LR=0.001

world = "grid_world2.txt"
goal_reward = 100
start_states = [(0, 0), (0, 20), (16, 21)]
goal_states = [(9, 5)]
max_steps = 10000

from grid_world import GridWorldEnv, GridWorldWindyEnv

env = GridWorldWindyEnv(
    world,
    goal_reward=goal_reward,
    start_states=start_states,
    goal_states=goal_states,
    max_steps=max_steps,
    action_fail_prob=0.2,
)
plt.figure(figsize=(10, 10))
# Go UP
env.step(UP)
env.render(ax=plt, render_agent=True)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.encode_net = nn.Sequential(
            nn.Conv2d(1, 4, 3),
            nn.ReLU(),
            nn.Conv2d(4, 5, 3),
            nn.ReLU(),
            nn.MaxPool2d(3),
            Flatten(),
        )
        self.actor_net = nn.Sequential(
            nn.Linear(120, 100),
            nn.ReLU(),
            nn.Linear(100, 4),
            nn.Softmax(dim=-1)
        )
        self.critic_net = nn.Sequential(
            nn.Linear(120, 100),
            nn.ReLU(),
            nn.Linear(100, 4)
        )
    
    def critic_forward(self, state):
        x = self.encode_net(state)
        return self.critic_net(x)
    
    def actor_forward(self, state):
        x = self.encode_net(state)
        return self.actor_net(x)
    
    def forward(self, state):
        x = self.encode_net(state)
        return self.actor_net(x), self.critic_net(x)
    
    def sample_action(self, state):
        probs = self.actor_forward(state).squeeze()
        dist = Categorical(probs)
        return dist.sample()
    


def update_ac(net: ActorCritic, opt:optim.Adam, s1, a, r, s2, a2):
    prob1, q1 = net(s1)
    q1 = q1[th.arange(q1.size(0)), a]
    prob1 = prob1[th.arange(prob1.size(0)), a]
    q2 = net.critic_forward(s2)
    q2 = q2[th.arange(q2.size(0)), a2].detach()
    td_error = (r + GAMMA*q2 - q1).pow(2).squeeze().mean()
    actor_loss = (-th.log(prob1) * q1.detach()).squeeze().mean()
    print(f"Actor Loss: {actor_loss}, Critic Loss: {td_error}")
    loss = td_error + actor_loss
    opt.zero_grad()
    loss.backward()
    opt.step()

    writer.add_scalar("Actor_loss", actor_loss, ct)
    writer.add_scalar("Critic_loss", td_error, ct)


def coords_to_grid(state):
    grid = env.coord_to_grid(*state)
    state = th.FloatTensor(grid).unsqueeze(0).unsqueeze(0).to(device)
    return state

EXPTS = 1
EPISODES=10000
PRINT_EVERY = 100

REWARDS, STEPS = [], []
for e in range(EXPTS):
    print(f" Experiment {e+1}")
    rewards = []
    steps = []
    ct = 0
    writer = SummaryWriter(f"log_dir/exp{e+1}", flush_secs=30)
    network = ActorCritic().to(device)

    opt = optim.Adam(network.parameters(), lr=LR)

    for ep in range(EPISODES):
        tot_r, tot_s = 0, 0
        print(f"Episode {ep+1}")
        buffer = [[],[],[],[],[]]
        done = False
        env.reset()
        network.eval()
        with th.no_grad():
            s1 = coords_to_grid(env.state)
            action = int(network.sample_action(s1).unsqueeze(0).detach().cpu().numpy())
            while not done:
                
                buffer[0].append(s1.clone())
                buffer[1].append(action)
                s1, r, done = env.step(action)
                tot_r += r
                tot_s += 1
                steps
                s1 = coords_to_grid(s1)
                buffer[2].append([r])
                buffer[3].append(s1)
                if not done:
                    action = int(network.sample_action(s1).unsqueeze(0).detach().cpu().numpy())
                else:
                    action = UP
                buffer[4].append(action)
            
            buffer[0] = th.cat(buffer[0], 0).to(device)
            buffer[2] = th.FloatTensor(buffer[2]).to(device)
            buffer[3] = th.cat(buffer[3], 0).to(device)
        network.train()
        update_ac(network, opt, buffer[0], buffer[1], buffer[2], buffer[3], buffer[4])
        rewards.append(tot_r)
        steps.append(tot_s)
        print(f"Reward: {tot_r}, Steps: {tot_s}")
        writer.add_scalar("Reward", tot_r, ct)
        writer.add_scalar("Steps", tot_s, ct)
        ct+=1
    REWARDS.append(rewards)
    STEPS.append(steps)
    writer.close()

import pickle
with open("rewards.pkl", "wb") as fl:
    pickle.dump(REWARDS, fl)

with open("steps.pkl", "wb") as fl:
    pickle.dump(STEPS, fl)