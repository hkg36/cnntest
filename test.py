import gymnasium as gym
from itertools import count
from functools import reduce
import numpy as np
import cv2
import matplotlib.pyplot as plt
env = gym.make('CarRacing-v2',render_mode="human")

env.reset()
sum_reward=0
for t in count():
    # [steering, gas, brake]
    act_take=np.array([0,1,0])
    # observation is 96x96x3
    observation, reward, done, info,_ = env.step(act_take)
    if t<100:
        continue
    speedline=observation[85:95,13,0]
    img = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    plt.imshow(img, cmap='gray')
    plt.show()
    sum_reward += reward
    
    if done:
        print(f"finished after {t+1} timesteps Reward: {sum_reward}")
    if done:
        break