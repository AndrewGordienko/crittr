import random
import pygame
import numpy as np
from Box2D import (
    b2World, b2PolygonShape
)

from .creatures import Creature
from .environment_space import ActionSpace
from .policies import actor_network

class Simulator:
    def __init__(self, creature_number=2, render_mode="human", seed: int = 0, gravity=(0, -10), ppm=20.0):
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
        self.screen = pygame.display.set_mode((self.width, self.height)) if render_mode == "human" else None
        self.clock = pygame.time.Clock() if render_mode == "human" else None

        self.render_mode = render_mode
        self.creature_number = creature_number

        # Fixed, padded action space: (N creatures, 4 legs, 3 segments)
        self.action_space = ActionSpace(-2.0, 2.0, (creature_number, 4, 3))
        self.observation_space = ActionSpace(-2.0, 2.0, (creature_number, 4, 3))
        
        # Per-creature alive mask
        self.alive = np.ones(self.creature_number, dtype=bool)

        # Build creatures up-front so step() is safe before reset()
        self.make_creature(self.creature_number)

        # State for reward shaping
        self.prev_x = np.zeros(self.creature_number, dtype=np.float32)
        for ci, c in enumerate(self.creatures):
            self.prev_x[ci] = c.body.position.x

        # Store last applied action for energy penalty
        self.last_action = np.zeros((self.creature_number, 4, 3), dtype=np.float32)

        self.actor = actor_network(self)

    # ---------- World / Entities ----------

    def make_creature(self, NUM_CREATURES):
        self.all_bodies = []
        self.all_joints = []
        self.creatures = []

        for _ in range(NUM_CREATURES):
            x = random.uniform(3, 45)
            y = random.uniform(25, 35)
            c = Creature(self.world, pos=(x, y))
            self.creatures.append(c)
            self.all_bodies.extend(c.bodies)   # list of (body, color)
            self.all_joints.extend(c.joints)   # list of joints

    # ---------- Step / Reset / Close ----------

    def step(self, action=None):
        self._process_pygame_events()
        # Normalize action
        if action is None:
            a = self.action_space.sample()
        else:
            a = np.asarray(action, dtype=np.float32)
            # Try to coerce to (N,4,3) if user passes flat or mismatched shape
            expected = self.creature_number * 12
            if a.ndim == 1 and a.size == expected:
                a = a.reshape(self.creature_number, 4, 3)
            elif a.shape != (self.creature_number, 4, 3):
                a = self.action_space.sample()
        a = np.clip(a, self.action_space.low, self.action_space.high)

        # Zero last_action; fill per alive creature
        self.last_action.fill(0.0)

        # Apply actions only to alive creatures
        for ci, creature in enumerate(self.creatures):
            if not self.alive[ci]:
                continue
            ai = a[ci]
            self.last_action[ci] = ai  # remember applied actions
            for li in range(4):
                for si in range(3):
                    j = creature.joint_grid[li][si]
                    if j is not None:
                        j.motorSpeed = float(ai[li, si])

        # Advance physics
        self.world.Step(1.0 / 60.0, 6, 2)

        # Outputs
        observation = self._get_obs()
        # --- Per-creature reward (vector) ---
        r_vec = self._get_reward()            # shape (N,), make _get_reward return vector

        # --- Per-creature done; persist deaths ---
        done_vec = self._get_done()           # shape (N,), True = this creature terminated
        self.alive &= ~done_vec               # once dead, stays dead

        # --- VectorEnv API expects arrays, not a single boolean ---
        terminations = done_vec.astype(bool)                      # (N,)
        truncations  = np.zeros(self.creature_number, bool)       # (N,) or set True on time-limit

        # --- infos: one dict per creature (list of length N) ---
        infos = [
            {"alive": bool(self.alive[ci]), "reward": float(r_vec[ci])}
            for ci in range(self.creature_number)
        ]

        # Optional render
        if self.render_mode == "human":
            self.render()

        # VectorEnv return signature:
        # (obs[N,4,3], rewards[N], terminations[N], truncations[N], infos[list[N]])
        return observation, r_vec.astype(np.float32), terminations, infos


    def reset(self, seed: int | None = None):
        """Reset the simulation and return (obs, info)."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Rebuild world
        self.world = b2World(self.world.gravity, doSleep=True)
        self.ground = self.world.CreateStaticBody(
            position=(0, 0),
            shapes=b2PolygonShape(box=(50, 1))
        )

        # Recreate creatures and reset masks/state
        self.make_creature(self.creature_number)
        self.alive[:] = True
        for ci, c in enumerate(self.creatures):
            self.prev_x[ci] = c.body.position.x
        self.last_action.fill(0.0)

        obs = self._get_obs()
        info = {}
        return obs, info

    def close(self):
        pygame.quit()
    
    def _process_pygame_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit

    # ---------- Rendering ----------

    def render(self):
        if self.render_mode != "human":
            return

        self.screen.fill((240, 240, 240))

        # Draw ground (green strip)
        pygame.draw.rect(self.screen, (34, 139, 34), pygame.Rect(0, self.height - 20, self.width, 20))

        # Draw bodies
        for body, color in self.all_bodies:
            for fixture in body.fixtures:
                shape = fixture.shape
                # world -> pixels
                vertices = [(body.transform * v) * self.ppm for v in shape.vertices]
                # flip y
                vertices = [(vx, self.height - vy) for (vx, vy) in vertices]
                pygame.draw.polygon(self.screen, color, vertices)

        pygame.display.flip()
        self.clock.tick(60)

    # ---------- Observations / Reward / Termination ----------

    def _get_obs(self):
        """
        Return joint angles in fixed slot order.
        Shape: (creature_number, 4, 3)
        Missing joints → 0.0
        """
        N = self.creature_number
        obs = np.zeros((N, 4, 3), dtype=np.float32)
        for ci, creature in enumerate(self.creatures):
            for li in range(4):
                for si in range(3):
                    j = creature.joint_grid[li][si]
                    if j is not None:
                        obs[ci, li, si] = j.angle  # Box2D revolute joint angle (radians)
        return obs

    def _get_reward(self):
        """
        Bipedal-style shaping (averaged across alive creatures):
          + forward progress along +x
          - small energy penalty from motor commands
          - small instability penalty from torso tilt
          + alive bonus while torso is above threshold
        Dead creatures contribute 0 to all terms.
        """
        N = self.creature_number
        progress = np.zeros(N, dtype=np.float32)
        instability = np.zeros(N, dtype=np.float32)
        alive_bonus = np.zeros(N, dtype=np.float32)

        # Per-creature terms
        for ci, c in enumerate(self.creatures):
            if self.alive[ci]:
                x_now = c.body.position.x
                dist_before = abs(self.prev_x[ci] - c.flag_x)
                dist_now = abs(x_now - c.flag_x)
                progress[ci] = dist_before - dist_now  # positive if moving toward flag
                self.prev_x[ci] = x_now
                instability[ci] = abs(c.body.angle)
                alive_bonus[ci] = 0.05 if c.body.position.y > 1.0 else -1.0
            else:
                # keep prev_x updated even if dead, to avoid huge deltas after revive (if ever)
                self.prev_x[ci] = c.body.position.x

        # Energy from last_action (only for existing joints on alive creatures)
        e = 0.0
        for ci, creature in enumerate(self.creatures):
            if not self.alive[ci]:
                continue
            ai = self.last_action[ci]
            for li in range(4):
                for si in range(3):
                    if creature.joint_grid[li][si] is not None:
                        e += abs(float(ai[li, si]))
        energy_penalty = 0.001 * e

        # Aggregate (across creatures)
        # If all dead, means will be 0, energy_penalty will be 0.
        reward_vec = (
            5.0 * progress          # toward flag
            - 0.01 * instability    # posture penalty
            + alive_bonus           # alive bonus
        )

        # distribute energy penalty evenly among alive creatures
        if self.alive.any():
            reward_vec[self.alive] -= energy_penalty / self.alive.sum()

        return reward_vec
    
    def _get_done(self):
        """
        Returns a boolean array (creature_number,) where True means that creature is dead.
        Conditions:
          - torso y too low (hit ground)
          - torso tilt too large
        """
        done = np.zeros(self.creature_number, dtype=bool)
        for ci, c in enumerate(self.creatures):
            if c.body.position.y < 0.5:
                done[ci] = True
            if abs(c.body.angle) > 1.0:  # ~57 degrees
                done[ci] = True
        return done
