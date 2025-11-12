import crittr
import numpy as np
from crittr.algorithms.continuous_ppo.intelligence import Agent

# create your simulator through the factory
env = crittr.make("crittr", creature_number=10, render_mode="human")

observation, info = env.reset()

agent = Agent(env)

for _ in range(1000):
    env.render()
    # action = env.action_space.sample()

    action, log_prob, value = agent.choose_action(observation)
    observation_, rewards, dones, info = env.step(action)

    agent.memory.store_memory(observation, action, log_prob, value, rewards, dones)
        # agent.learn()

    observation = observation_


    if np.all(dones):
        observation, info = env.reset()


env.close()

