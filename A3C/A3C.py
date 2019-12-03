import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
import numpy as np
import gym
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# One thread per process
os.environ["OMP_NUM_THREADS"] = "1"
# Necessary for cuda to work RUN ONLY ONCE
#device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
device = th.device('cpu')

ENV_NAME = "CartPole-v0"
env = gym.make(ENV_NAME)
ACTION_NUM = env.action_space.n
STATE_DIM = env.observation_space.shape[0]
GAMMA = 0.9
print("No. of Actions: %d Dimensions of state %d"%(ACTION_NUM, STATE_DIM))

def init_model(m):
    for p in m.parameters():
        if p.dim() > 1:
            nn.init.normal_(p, 0, 0.1)
        else:
            nn.init.constant_(p, 0.)

def to_torch(np_array, dtype=np.float32, device=device):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return th.from_numpy(np_array).to(device)

class Actor(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dims = [200, 100]):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_num = action_num
        self.first_layer = nn.Linear(state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for d in range(1, len(hidden_dims)):
            self.hidden_layers.append(nn.Linear(hidden_dims[d-1], hidden_dims[d]))
        self.last_layer = nn.Linear(hidden_dims[-1], action_num)
        init_model(self)
    
    def forward(self, x):
        x = F.relu6(self.first_layer(x))
        for layer in self.hidden_layers:
            x = F.relu6(layer(x))
        logits = self.last_layer(x)
        return logits

    
class Value(nn.Module):
    def __init__(self, state_dim, hidden_dims = [200, 100]):
        super(Value, self).__init__()
        self.state_dim = state_dim
        self.first_layer = nn.Linear(state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for d in range(1, len(hidden_dims)):
            self.hidden_layers.append(nn.Linear(hidden_dims[d-1], hidden_dims[d]))
        self.last_layer = nn.Linear(hidden_dims[-1], 1)
        init_model(self)
    
    def forward(self, x):
        x = F.relu6(self.first_layer(x))
        for layer in self.hidden_layers:
            x = F.relu6(layer(x))
        values = self.last_layer(x)
        return values


class A2C(nn.Module):
    def __init__(self, state_dim, action_num, hidden_actor_dims = [200], hidden_value_dims = [100]):
        super(A2C, self).__init__()
        self.state_dim = state_dim
        self.action_num = action_num
        self.actor = Actor(state_dim, action_num, hidden_actor_dims)
        self.value = Value(state_dim, hidden_value_dims)
        self.distribution = th.distributions.Categorical
        
    def forward(self, x):
        logits = self.actor(x)
        values = self.value(x)
        return logits, values
    
    def choose_action(self, x):
        self.eval()
        logits, _ = self.forward(x)
        prob = F.softmax(logits, dim=1).data
        return self.distribution(prob).sample().cpu().numpy()[0]
    
    # Calculation of loss:
    # v_target = r + v(s_{t+1})
    def loss_func(self, state, action, v_target):
        self.train()
        logits, values = self.forward(state)
        advantages = v_target - values
        #Mean Square error of target v and predicted v
        value_loss = advantages.pow(2)
        
        # Loss w.r.t Actor
        prob = F.softmax(logits, dim=1).data
        dist = self.distribution(prob)
        actor_loss = -(dist.log_prob(action)*advantages.detach().squeeze())
        
        return (value_loss + actor_loss).mean()


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)
class A2C(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(A2C, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 200)
        self.pi2 = nn.Linear(200, a_dim)
        self.v1 = nn.Linear(s_dim, 100)
        self.v2 = nn.Linear(100, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = th.distributions.Categorical

    def forward(self, x):
        pi1 = F.relu6(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = F.relu6(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().cpu().numpy()[0]

    def loss_func(self, state, action, v_target):
        self.train()
        logits, values = self.forward(state)
        adv = v_target - values
        c_loss = adv.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(action) * adv.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss

class SharedAdam(th.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.9), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = th.zeros_like(p.data)
                state['exp_avg_sq'] = th.zeros_like(p.data)

                # share in memory to all processes
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


LR = 0.0001
#Number of steps till update to target network
UPDATE_FREQ = 4
MAX_EPISODES = 10000
#Number of worker networks
NUM_WORKERS = mp.cpu_count()


target_net = A2C(STATE_DIM, ACTION_NUM).to(device)
# Share network object to all processes
target_net.share_memory()
optimizer = SharedAdam(target_net.parameters(), lr=LR)

#Global variables
total_episodes = mp.Value('i', 0)
reward_queue = mp.Queue()

class Worker(mp.Process):
    def __init__(self, target_net, opt, total_ep, result_queue, idx):
        super(Worker, self).__init__()
        self.idx = idx
        self.tot_eps, self.result_queue = total_ep, result_queue
        self.target_net, self.opt = target_net, opt
        self.worker_net = A2C(STATE_DIM, ACTION_NUM).to(device)           # local network
        self.env = gym.make(ENV_NAME)
        self.render  = False
    
    # Training routine
    def run(self):
        total_steps = 1
        total_episodes = 1
        while self.tot_eps.value < MAX_EPISODES:
            state = self.env.reset()
            
            # Stores the episode rollout (state,action,reward)
            buffer = ([], [], [])
            total_reward = 0.
            done = False
            while not done:
                if self.render:
                    self.env.render()
                
                # Select Action
                action = self.worker_net.choose_action(to_torch(state[None, :]))
        
                next_state, reward, done, _ = self.env.step(action)
                #if done: reward = -1
                total_reward += reward
                
                # Append to rollout buffer
                buffer[0].append(state)
                buffer[1].append(action)
                buffer[2].append(reward)
                
                # Asynchronous update
                if total_steps % UPDATE_FREQ == 0 or done:  # update global and assign to local net
                    # sync
                    update_params(self.opt, self.worker_net, self.target_net, done, next_state, buffer, GAMMA, device=device)
                    buffer = [[], [], []]

                    if done:
                        # store and print reward obtianed
                        
                        self.result_queue.put((total_episodes, total_reward))
                        with self.tot_eps.get_lock():
                            self.tot_eps.value += 1
                        print("Worker %2d: Episode %4d, Reward: %4d"%(self.idx, self.tot_eps.value, total_reward))
                        break
                state = next_state
                total_steps += 1
            total_episodes += 1
        self.result_queue.put(None)


def update_params(opt, worker_network, target_network, done, next_state, buffer, gamma, device=device):
    # Estimate V_()
    v_t = 0. if done else \
        worker_network.forward(to_torch(next_state[None, :]))[-1].data.cpu().numpy()[0, 0]
    
    # Calculate V(s_t)
    v_targets = []
    for reward in buffer[2][::-1]:
        v_t = reward + gamma*v_t
        v_targets.append(v_t)
    v_targets.reverse()
    v_targets = np.array(v_targets)
    
    # Calculate loss for the worker network
    worker_loss = worker_network.loss_func(state = to_torch(np.vstack(buffer[0])),
                                        action = to_torch(np.vstack(buffer[1]), dtype=np.int64), v_target=to_torch(v_targets[:, None]))
    
    #Get gradients
    opt.zero_grad()
    worker_loss.backward()
    
    # Copy gradients
    for wp, tp in zip(worker_network.parameters(), target_network.parameters()):
        tp.grad = wp.grad
    # Apply gradient
    opt.step()
    
    # Copy weights of global network to worker network
    worker_network.load_state_dict(target_network.state_dict())

    
worker_trainers = [Worker(target_net, optimizer, total_episodes, reward_queue, x) for x in range(NUM_WORKERS)]
worker_trainers[0].render = True
for w in worker_trainers:
    w.start()

ct = 0
results = pd.DataFrame(columns=['Episode', 'Reward'])

while ct<NUM_WORKERS:
    res = reward_queue.get()
    if res is None:
        break
    else:
        results = results.append({'Episode': res[0], 'Reward': res[1]}, ignore_index=True)

for w in worker_trainers:
    w.join()

plt.plot(results.Reward)
plt.ylabel('Reward')
plt.xlabel('Step')
plt.show()