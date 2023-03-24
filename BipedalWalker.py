import gymnasium as gym
import numpy as np
import numpy_gen
import random
from itertools import count
from functools import reduce

def Mute(a,e=0.1):
    raded=np.random.rand(*a.shape)<e
    r = np.random.rand(*a.shape)*1.2-0.1
    a[raded]=r[raded]
    return a
numpy_gen.Mute=Mute
gamename="BipedalWalker-v3"

env = gym.make(gamename,render_mode=None)
env2 = gym.make(gamename,render_mode="human")
print(env.action_space)
#n_actions=env.action_space.n
print(env.observation_space.shape[0])
o_shape=env.observation_space.shape[0]

acts=[
    [-1,0,1] for i in range(4)
    ]
acts_len=[len(a) for a in acts]
output=reduce(lambda x,y:x*y,acts_len)
print("output:",output)
factory=numpy_gen.BuildGenNNFactory(o_shape,120,numpy_gen.leakyrelu,
                                    100,numpy_gen.leakyrelu,
                                    90,numpy_gen.leakyrelu,
                                    output)

gnn=[]
for i in range(1000):
    gn=factory.NewNN()
    gnn.append(gn)
step_max=1000
def RunOne(env,nn):
    observation_last=env.reset()[0]
    rewardall=0
    act_take=np.zeros([len(acts_len),])
    for t in count():
        env.render()
        res=nn.forward(observation_last)
        resact = np.unravel_index(np.argmax(res, axis=None), acts_len)
        for i in range(len(acts_len)):
            act_take[i]=acts[i][resact[i]]
        observation, reward, done,_,_= env.step(act_take)
        observation_last=observation
        rewardall+=reward
        if done or t >step_max:
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
    if i_episode%2==0 and gnn[0].score>50:
        RunOne(env2,gnn[0])

    if len(gnn)>500:
        gnn=gnn[:400]
    selparent=gnn[:50]
    for i in range(200):
        pars=random.sample(selparent,2)
        newnn=factory.Mate(*pars)
        if random.random()<0.1:
            factory.Mute(newnn,0.01)
        gnn.append(newnn)
