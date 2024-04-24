import gymnasium as gym
import numpy as np
import cv2
import numpy_gen
from itertools import count
from functools import reduce
import random
# import myplot

render = True
n_episodes = 100
env = gym.make('CarRacing-v2',render_mode=None)
env2 = gym.make('CarRacing-v2',render_mode="human")

print(env.action_space)
print(env.observation_space)

acts=[
    [-1,-0.5,0,0.5,1],
    [0,0.5,1],
    [0,1]
]
acts_len=[len(a) for a in acts]
factory=numpy_gen.BuildGenNNFactory(5,40,numpy_gen.leakyrelu,20,numpy_gen.leakyrelu,reduce(lambda x,y:x+y,acts_len))

gnn=[]
for i in range(100):
    gn=factory.NewNN()
    gnn.append(gn)

def countDis(dt):
    c=0
    for o in dt[::-1]:
        if o==False:
            return c
        c+=1
    return c
def transState(observation):
    img = observation[:84, 6:90]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    forward=img[:65,42]<140
    side=img[65,:]<140
    sideleft=side[:42]
    sideright=side[42:][::-1]
    leftforward=np.diagonal(img,42-65)[:42]<140
    rightforward=np.diagonal(np.fliplr(img),42-65)[:42]<140
    return np.array((countDis(forward),countDis(sideleft),countDis(sideright),countDis(leftforward),countDis(rightforward)))

def RunOne(env,nn):
    observation = env.reset()
    sum_reward = 0
    state=None
    for t in count():
        if render:
            env.render()
        # [steering, gas, brake]
        act_take=np.zeros([len(acts_len),])
        if t>80:
            if state is not None:
                res=nn.forward(state)
                pre=0
                for i in range(len(acts_len)):
                    aft=pre+acts_len[i]
                    act_take[i]=acts[i][np.argmax(res[pre:aft])]
        # observation is 96x96x3
        observation, reward, done, info,_ = env.step(act_take)
        if t<80:
            continue
        state=transState(observation)
        sum_reward += reward
        if (state<=0).any() or sum_reward<=-10:
            done=True
        #print(state,reward)
        
        if done:
            print(f"finished after {t+1} timesteps Reward: {sum_reward}")
        if done:
            break
    return sum_reward
for i_episode in range(n_episodes):
    for n in gnn:
        n.score=RunOne(env,n)
    gnn.sort(key=lambda a:a.score,reverse=True)
    print("max:",gnn[0].score)
    if i_episode%5==0:
        RunOne(env2,gnn[0])

    if len(gnn)>200:
        gnn=gnn[:150]
    selparent=gnn[:40]
    for i in range(100):
        pars=random.sample(selparent,2)
        newnn=factory.Mate(*pars)
        if random.random()<0.1:
            factory.Mute(newnn,0.01)
        gnn.append(newnn)