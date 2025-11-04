import crittr
import pygame

# create your simulator through the factory
env = crittr.make("crittr", creature_number=10, render_mode="human")

t = 0
running = True

observation, info = env.reset()
print("Observation shape:", observation.shape)
print("First few values:", observation[:8])

while running:
    action = env.action_space.sample()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    env.step(action)
    env.render()
    t += 1

env.close()