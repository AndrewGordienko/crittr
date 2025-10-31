import random
import pygame
import numpy as np

from .creatures import Creature
from Box2D import (
    b2World, b2PolygonShape, b2_dynamicBody, b2_staticBody,
    b2RevoluteJointDef, b2Filter
)

class Simulator:
    def __init__(self, creature_number=2, render_mode="human", seed: int = 0, gravity=(0,-10), ppm=20.0):
        self.ppm = ppm
        self.width, self.height = 1000, 600

        # Physics world
        self.world = b2World(gravity, doSleep=True)
        self.ground = self.world.CreateStaticBody(
            position=(0, 0),
            shapes=b2PolygonShape(box=(50, 1))
        )
        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        self.render_mode = render_mode
        self.creature_number = creature_number

    def make_creature(self, NUM_CREATURES):
        self.all_bodies = []
        self.all_joints = []

        for _ in range(NUM_CREATURES):
            x = random.uniform(3, 45)
            y = random.uniform(25, 35)

            creature = Creature(self.world, pos=(x, y))
            bodies, joints = creature.bodies, creature.joints

            self.all_bodies.extend(bodies)
            self.all_joints.extend(joints)

    def step(self, t: int):
        """Advance simulation and update motors every 20 ticks."""
        if t % 20 == 0:
            for j in self.all_joints:
                j.motorSpeed = random.uniform(-3.0, 3.0)

        obs = self._get_obs()
        reward = self.reward_state()
        terminated = self.terminated_state()
        
        self.world.Step(1.0/60.0, 6, 2)

        return obs

    def reward_state(self):
        return 0
    
    def terminated_state(self):
        return False

    def render(self):
        if self.render_mode == "human":
            """Draw ground and creatures."""
            self.screen.fill((240, 240, 240))

            # Draw ground (green)
            pygame.draw.rect(self.screen, (34, 139, 34), pygame.Rect(0, self.height-20, self.width, 20))

            # Draw creatures
            for body, color in self.all_bodies:
                for fixture in body.fixtures:
                    shape = fixture.shape
                    vertices = [(body.transform * v) * self.ppm for v in shape.vertices]
                    vertices = [(v[0], self.height - v[1]) for v in vertices]
                    pygame.draw.polygon(self.screen, color, vertices)

            pygame.display.flip()
            self.clock.tick(60)
    
    def reset(self, seed: int | None = None):
        """Reset the simulation and return (obs, info)."""
        if seed is not None:
            random.seed(seed)

        # Clear world
        self.world = b2World(self.world.gravity, doSleep=True)
        self.ground = self.world.CreateStaticBody(
            position=(0, 0),
            shapes=b2PolygonShape(box=(50, 1))
        )

        # Recreate creatures
        self.make_creature(self.creature_number)

        # Observation: joint anchor positions
        obs = self._get_obs()
        info = {}

        return obs, info

    def _get_obs(self):
        """Return observation = array of joint anchor positions."""
        obs = []
        for j in self.all_joints:
            ax, ay = j.anchorA
            bx, by = j.anchorB
            obs.extend([ax, ay, bx, by])
        return np.array(obs, dtype=float)
