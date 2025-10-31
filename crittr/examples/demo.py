import crittr
import pygame

# create your simulator through the factory
env = crittr.make("crittr", creature_number=10, render_mode="human")

t = 0
running = True

obs, info = env.reset()
print("Observation shape:", obs.shape)
print("First few values:", obs[:8])

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    env.step(t)
    env.render()
    t += 1

pygame.quit()