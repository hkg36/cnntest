import gymnasium as gym
import numpy as np
import numpy_gen
import random
from itertools import count

gamename="LunarLander-v2"#"LunarLander-v2" #'CartPole-v1'

env = gym.make(gamename,continuous=True,render_mode=None)
env2 = gym.make(gamename,continuous=True,render_mode="human")
env._max_episode_steps=env2._max_episode_steps=10000000
print(env.action_space)
#n_actions=env.action_space.n
print(env.observation_space.shape[0])
o_shape=env.observation_space.shape[0]-2

class MyGNN(numpy_gen.GenNNBase):
    def __init__(self) -> None:
        super().__init__()
        self.scrohis=[]
factory=numpy_gen.GenNNFactory((o_shape,20),
                               numpy_gen.leakyrelu,
                               (20,10),
                               numpy_gen.leakyrelu,
                               (10,4))
#factory.setNNClass(MyGNN)
gnn=[]
for i in range(100):
    gn=factory.NewNN()
    gnn.append(gn)

def RunOne(env,nn):
    observation_last=env.reset()[0]
    rewardall=0
    stopjet=False
    act_take=np.array([0,0])
    for t in count():
        env.render()
        if np.all(observation_last[-2:]):
            stopjet=True
        res=nn.forward(observation_last[:-2])
        resact = np.argmax(res)
        if resact==0:
            act_take=np.array([0,0])
        elif resact==1:
            act_take=np.array([0,-1])
        elif resact==2:
            act_take=np.array([1,0])
        elif resact==3:
            act_take=np.array([0,1])
        #if stopjet and action==2:
        #    action=0
        observation, reward, done,_,_= env.step(act_take)
        observation_last=observation
        rewardall+=reward
        if done or t >1000:
            break
    return rewardall
for i_episode in count():
    for n in gnn:
        n.score=RunOne(env,n)
    """    n.scrohis.append(scro)
    for n in gnn:
        n.scrohis=n.scrohis[-3:]
        n.score=sum(n.scrohis)/len(n.scrohis)"""
    gnn.sort(key=lambda a:a.score,reverse=True)
    print("max:",gnn[0].score)
    if i_episode%5==0:
        RunOne(env2,gnn[0])

    if len(gnn)>500:
        gnn=gnn[:400]
    selparent=gnn[:20]
    for i in range(100):
        pars=random.sample(selparent,2)
        newnn=factory.Mate(*pars)
        if random.random()<0.1:
            factory.Mute(newnn,0.01)
        gnn.append(newnn)
