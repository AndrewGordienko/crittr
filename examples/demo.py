import crittr
import numpy as np

# create your simulator through the factory
env = crittr.make("crittr", creature_number=10, render_mode="human")

observation, info = env.reset()

actor = env.actor

for _ in range(1000):
    env.render()
    # action = env.action_space.sample()

    action = actor.act(observation)
    observation_, rewards, dones, info = env.step(action)

    # agent.memory.add(state, action, reward, state_, done)
        # agent.learn()

    observation = observation_


    if np.all(dones):
        observation, info = env.reset()


env.close()

