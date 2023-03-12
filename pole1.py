import gymnasium as gym
from collections import namedtuple, deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from itertools import count

MEMORYSIZE = 3000
BATCH_SIZE = int(MEMORYSIZE/3*2)
GAMMA = 0.95
TARGET_UPDATE = 5
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    def push_bash(self,vs):
        self.memory.extend(vs)
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)
    def clear(self):
        self.memory.clear()
memory=ReplayMemory(MEMORYSIZE)
gamename="LunarLander-v2"#"LunarLander-v2" #'CartPole-v1'
USERANDACT=False
env = gym.make(gamename)
env2 = gym.make(gamename,render_mode="human")
#env._max_episode_steps=env2._max_episode_steps=10000000
print(env.action_space)
n_actions=env.action_space.n
print(env.observation_space.shape[0])
ob_shape=env.observation_space.shape[0]-2
class DQN(nn.Module):
    def __init__(self, obshape,actspace):
        super(DQN, self).__init__()
        self.proc=nn.Sequential(
            nn.Linear(obshape,20),
            nn.LeakyReLU(),
            nn.Linear(20,10),
            nn.LeakyReLU(),
            nn.Linear(10,actspace),
        )

    def forward(self, x):
        return self.proc(x)

policy_net = DQN(ob_shape, n_actions).to(device)
target_net = DQN(ob_shape, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
policy_net.eval()
#RMSprop
optimizer = optim.RMSprop(policy_net.parameters(),lr=0.01)

steps_done=0

def select_action(state):
    global steps_done
    if state is None:
        return torch.tensor([[random.randrange(n_actions)]], device=device)
    if USERANDACT==False:
        with torch.no_grad():
            return policy_net(state.unsqueeze(0)).max(1)[1]
        
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state.unsqueeze(0)).max(1)[1]
    else:
        return torch.tensor([random.randrange(n_actions)], device=device)

criterion = nn.SmoothL1Loss()
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return 1000
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    #with torch.no_grad():
    #    next_state_values[non_final_mask] = policy_net(non_final_next_states).max(1)[0].detach()
    policy_net.train()
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    expected_state_action_values = (next_state_values.unsqueeze(1) * (GAMMA)) + (reward_batch)
    optimizer.zero_grad()
    loss = criterion(state_action_values, expected_state_action_values)
    loss.backward()
    optimizer.step()
    policy_net.eval()
    return loss.item()

observation_last=None

reward0=torch.tensor([0], device=device)
for i_episode in count():
    observation_last=env.reset()[0]
    touch=observation_last[-2:]
    observation_last=torch.tensor(observation_last[:-2],device=device)
    
    go_update=False
    tmprecord=[]
    reward_sum=0
    for t in count():
        if np.any(touch):
            action=torch.tensor([0,])
        else:
            action = select_action(observation_last)
        observation, reward, done,_,_= env.step(action.item())
        reward_sum+=reward
        touch=observation[-2:]
        observation=torch.tensor(observation[:-2],device=device)
        reward = torch.tensor([reward], device=device)
        if observation_last is not None:
            if done:
                tmprecord.append(Transition(observation_last,action,None,reward0))
            else:
                tmprecord.append(Transition(observation_last,action,observation,reward))
        observation_last=observation
        #optimize_model()
        if done or reward_sum<-200:
            memory.push_bash(tmprecord)
            break
    
    loss=optimize_model()
    print(f"({reward_sum}),{t}\tloss\t{loss:.4f}")
    #if ave_t<10 and steps_done>EPS_DECAY:
    #    steps_done=EPS_DECAY
    if i_episode%TARGET_UPDATE==0:
        #pass
        target_net.load_state_dict(policy_net.state_dict())

        if reward_sum>100:
            observation_last=env2.reset()[0]
            touch=observation_last[-2:]
            observation_last=torch.tensor(observation_last[:-2],device=device)
            for t in count():
                env2.render()
                if np.any(touch):
                    action=torch.tensor([0,])
                else:
                    action = select_action(observation_last)
                observation, reward, done,_,_= env2.step(action.item())
                touch=observation[-2:]
                observation=torch.tensor(observation[:-2],device=device)
                observation_last=observation
                if done or t>1000:
                    break
env.close()