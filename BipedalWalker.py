import gymnasium as gym
import numpy as np
import numpy_gen
import random
from itertools import count
from functools import reduce

gamename="BipedalWalker-v3"

env = gym.make(gamename,render_mode=None)
env2 = gym.make(gamename,render_mode="human")
print(env.action_space)
#n_actions=env.action_space.n
print(env.observation_space.shape[0])
o_shape=env.observation_space.shape[0]

bottomgp=np.array([-3.1415927,-5.,-5.,-5.,-3.1415927,-5.,-3.1415927,-5.,-0.,-3.1415927,-5.,-3.1415927,-5.,-0.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,])
topgap=np.array([3.1415927,5.,5.,5.,3.1415927,5.,3.1415927,5.,5.,3.1415927,5.,3.1415927,5.,5.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,])
oblen=len(bottomgp)
gapp=topgap-bottomgp

acts=[
    [-1,0,1] for i in range(4)
    ]
acts_len=[len(a) for a in acts]
output=reduce(lambda x,y:x*y,acts_len)
print("output:",output)
factory=numpy_gen.BuildGenNNFactory(o_shape,120,numpy_gen.leakyrelu,
                                    100,numpy_gen.leakyrelu,
                                    output)

gnn=[]
for i in range(1000):
    gn=factory.NewNN()
    gnn.append(gn)
step_max=1000

def tranobservation(ob):
    return (ob-bottomgp)/gapp
    for i in range(oblen):
        ob[i]=(ob[i]-bottomgp[i])/gapp[i]
    return ob
def RunOne(env,nn):
    observation_last=env.reset()[0]
    rewardall=0
    act_take=np.zeros([len(acts_len),])
    for t in count():
        env.render()
        tran_observation=tranobservation(observation_last)
        res=nn.forward(tran_observation)
        resact = np.unravel_index(np.argmax(res, axis=None), acts_len)
        for i in range(len(acts_len)):
            act_take[i]=acts[i][resact[i]]
        observation, reward, done,_,_= env.step(act_take)
        observation_last=observation
        rewardall+=reward
        if done:
            break
    return rewardall
for i_episode in count():
    for n in gnn:
        if np.isnan(n.score):
            n.score=RunOne(env,n)
    """    n.scrohis.append(scro)
    for n in gnn:
        n.scrohis=n.scrohis[-3:]
        n.score=sum(n.scrohis)/len(n.scrohis)"""
    gnn.sort(key=lambda a:a.score,reverse=True)
    print("max:",gnn[0].score)
    if i_episode%2==0 and gnn[0].score>50:
        RunOne(env2,gnn[0])

    if len(gnn)>200:
        gnn=gnn[:200]
    selparent=gnn[:50]
    for i in range(200):
        pars=random.sample(selparent,2)
        newnn=factory.Mate(*pars)
        if random.random()<0.1:
            factory.Mute(newnn,0.01)
        gnn.append(newnn)
