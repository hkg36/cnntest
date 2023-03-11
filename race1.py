import gymnasium as gym
from itertools import count
import matplotlib.pyplot as plt
# import myplot

render = True
n_episodes = 100
env = gym.make('CarRacing-v2',render_mode="human")

print(env.action_space)
print(env.observation_space)

rewards = []
for i_episode in range(n_episodes):
    observation = env.reset()
    sum_reward = 0
    for t in count():
        if render:
            env.render()
        # [steering, gas, brake]
        action = env.action_space.sample()
        # observation is 96x96x3
        observation, reward, done, info,_ = env.step(action)
        print(type(observation))
        print(len(observation[0]))
        print(len(observation[0][0]))
        # break
        sum_reward += reward
        if(t % 100 == 0):
            print(t)
        if done:
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            print("Reward: {}".format(sum_reward))
            rewards.append(sum_reward)
        if done:
            break