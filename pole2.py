import gymnasium as gym
import numpy as np
import numpy_gen
import random
from itertools import count
from functools import reduce

gamename="LunarLander-v2"#"LunarLander-v2" #'CartPole-v1'

env = gym.make(gamename,continuous=True,render_mode=None)
env2 = gym.make(gamename,continuous=True,render_mode="human")
print(env.action_space)
#n_actions=env.action_space.n
print(env.observation_space.shape[0])
o_shape=env.observation_space.shape[0]-2

acts=[
    [0,0.3,0.7,1],
    [-1,-0.75,-0.25,0,0.25,0.75,1]
]
acts_len=[len(a) for a in acts]

"""factory=numpy_gen.GenNNFactory((o_shape,40),
                               numpy_gen.leakyrelu,
                               (40,20),
                               numpy_gen.leakyrelu,
                               (20,a1len*a2len))"""
factory=numpy_gen.BuildGenNNFactory(o_shape,40,numpy_gen.leakyrelu,20,numpy_gen.softmax,reduce(lambda x,y:x+y,acts_len))

gnn=[]
for i in range(100):
    gn=factory.NewNN()
    gnn.append(gn)

def RunOne(env,nn):
    observation_last=env.reset()[0]
    rewardall=0
    stopjet=False
    act_take=np.zeros([len(acts_len),])
    for t in count():
        env.render()
        if np.all(observation_last[-2:]):
            stopjet=True
        res=nn.forward(observation_last[:-2])
        #resact = np.unravel_index(np.argmax(res, axis=None), acts_len)
        pre=0
        for i in range(len(acts_len)):
            aft=pre+acts_len[i]
            act_take[i]=acts[i][np.argmax(res[pre:aft])]
        #    act_take[i]=acts[i][resact[i]]
        if stopjet:
            act_take[0]=0
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
    selparent=gnn[:40]
    for i in range(100):
        pars=random.sample(selparent,2)
        newnn=factory.Mate(*pars)
        if random.random()<0.1:
            factory.Mute(newnn,0.01)
        gnn.append(newnn)
