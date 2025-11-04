import random
import pygame
import numpy as np
from Box2D import (
    b2World, b2PolygonShape, b2_dynamicBody, b2_staticBody,
    b2RevoluteJointDef, b2Filter
)

from .creatures import Creature
from .environment_space import ActionSpace


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
        self.action_space = ActionSpace(-2.0, 2.0, (creature_number, 4, 3))

    def make_creature(self, NUM_CREATURES):
        self.all_bodies = []
        self.all_joints = []
        self.creatures = []

        for _ in range(NUM_CREATURES):
            x = random.uniform(3, 45)
            y = random.uniform(25, 35)
            c = Creature(self.world, pos=(x, y))
            self.creatures.append(c)
            self.all_bodies.extend(c.bodies)
            self.all_joints.extend(c.joints)


    def step(self, action=None):
        if action is None:
            action = self.action_space.sample()

        a = np.asarray(action, dtype=np.float32)
        a = np.clip(a, self.action_space.low, self.action_space.high)

        for ci, creature in enumerate(self.creatures):
            # shape check defensive
            ai = a[ci] if ci < a.shape[0] else 0.0
            for li in range(4):
                for si in range(3):
                    j = creature.joint_grid[li][si]
                    speed = float(ai[li, si]) if isinstance(ai, np.ndarray) else 0.0
                    if j is not None:
                        j.motorSpeed = speed  # apply
                    # else: skipped (slot consumed but not applied)

        # physics step
        self.world.Step(1.0/60.0, 6, 2)

        obs = self._get_obs()
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
    
    def close(self):
        pygame.quit()
