import gymnasium as gym
from collections import namedtuple, deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from itertools import count

BATCH_SIZE = 1000
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
memory=ReplayMemory(3000)
env = gym.make('Acrobot-v1')
print(env.action_space)
n_actions=env.action_space.n
print(env.observation_space.shape[0])
class DQN(nn.Module):
    def __init__(self, obshape,actspace):
        super(DQN, self).__init__()
        self.proc=nn.Sequential(
            nn.Linear(obshape,20),
            nn.Softplus(),
            nn.Linear(20,10),
            nn.Softplus(),
            nn.Linear(10,actspace),
        )

    def forward(self, x):
        return self.proc(x)

policy_net = DQN(env.observation_space.shape[0], n_actions).to(device)
target_net = DQN(env.observation_space.shape[0], n_actions).to(device)
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
        return 1.0
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

TARGET_SHOW=20
reward0=torch.tensor([0], device=device)
ave_t=0
for i_episode in count():
    observation_last=env.reset()[0]
    observation_last=torch.tensor(observation_last,device=device)
    
    go_update=False
    tmprecord=[]
    for t in count():
        if i_episode % TARGET_SHOW == 0:
            env.render()
        action = select_action(observation_last)
        observation, reward, done, _,_= env.step(action.item())
        observation=torch.tensor(observation,device=device)
        reward = torch.tensor([reward], device=device)
        if observation_last is not None:
            if done:
                tmprecord.append(Transition(observation_last,action,None,reward0))
            else:
                tmprecord.append(Transition(observation_last,action,observation,reward))
        observation_last=observation
        #optimize_model()
        if done or t>1000:
            if t>(ave_t*0.8) or t>150:
                ave_t=ave_t*0.9+t*0.1
            memory.push_bash(tmprecord)
            break
    
    loss=1.0
    if t<1000:
        loss=optimize_model()
    print(f"({ave_t:.2f}){t}\tloss\t{loss:.4f}")
    #if ave_t<10 and steps_done>EPS_DECAY:
    #    steps_done=EPS_DECAY
    if i_episode%TARGET_UPDATE==0:
        #pass
        target_net.load_state_dict(policy_net.state_dict())
env.close()