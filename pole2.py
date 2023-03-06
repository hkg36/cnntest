import gym
import numpy as np
import numpy_gen
import random
from itertools import count

env = gym.make('CartPole-v1')
env._max_episode_steps=10000000
print(env.action_space)
n_actions=env.action_space.n
print(env.observation_space.shape[0])
o_shape=env.observation_space.shape[0]

class GenNN(object):
    def __init__(self, obshape=None,actspace=None):
        if obshape is not None and actspace is not None:
            self.n1=np.random.rand(obshape,20)
            self.n2=np.random.rand(20,10)
            self.n3=np.random.rand(10,actspace)
        else:
            self.n1=None
            self.n2=None
            self.n3=None
        self.runstep=np.nan
    def forward(self, x):
        """
        nn.Linear(obshape,20),
            nn.ReLU(),
            nn.Linear(20,10),
            nn.ReLU(),
            nn.Linear(10,actspace),
        """
        tmp=np.dot(x,self.n1)
        tmp=numpy_gen.relu(tmp)
        tmp=np.dot(tmp,self.n2)
        tmp=numpy_gen.relu(tmp)
        tmp=np.dot(tmp,self.n3)
        return tmp
def Mate(na,nb):
    newnn=GenNN()
    newnn.n1=numpy_gen.Mate(na.n1,nb.n1)
    newnn.n2=numpy_gen.Mate(na.n2,nb.n2)
    newnn.n3=numpy_gen.Mate(na.n3,nb.n3)
    return newnn
def Mute(na):
    numpy_gen.Mute(na.n1,0.05)
    numpy_gen.Mute(na.n2,0.05)
    numpy_gen.Mute(na.n3,0.05)

gnn=[]
for i in range(100):
    gn=GenNN(o_shape,n_actions)
    gnn.append(gn)

def select_action(nn,state):
    res=nn.forward(state)
    return np.argmax(res)

noise=0.01
for i_episode in count():
    noise+=0.001
    if noise>0.05:
        break
    for n in gnn:
        showrun=random.randint(0,1)
        observation_last=env.reset()
        for t in count():
            if random.random()<noise: #训练抗噪音信号的网络
                action=random.randint(0,n_actions-1)
            else:
                action = select_action(n,observation_last)
            observation, reward, done,_= env.step(action)
            observation_last=observation

            if done or t>1000:
                break
        n.runstep=t
    gnn.sort(key=lambda a:a.runstep,reverse=True)

    if True:
        print("max:",gnn[0].runstep)
        observation_last=env.reset()
        for t in count():
            env.render()
            if random.random()<0.02:
                action=random.randint(0,n_actions-1)
            else:
                action = select_action(gnn[0],observation_last)
            observation, reward, done,_= env.step(action)
            observation_last=observation
            if done or t>300:
                break

    if len(gnn)>500:
        gnn=gnn[:400]
    selparent=gnn[:50]
    for i in range(100):
        pars=random.sample(selparent,2)
        newnn=Mate(*pars)
        if random.random()<0.1:
            Mute(newnn)
        gnn.append(newnn)

for n in gnn:
    print(n.runstep)