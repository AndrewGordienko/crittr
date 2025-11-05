import random
import colorsys
from Box2D import (
    b2World, b2PolygonShape, b2_dynamicBody, b2_staticBody,
    b2RevoluteJointDef, b2Filter
)

# Base palette (pastel-ish); each creature will get a slight variation
BASE_TORSO_COLOR = (170, 210, 235)   # light blue base
BASE_LEG_COLOR   = (255, 182, 193)   # light pink


class Creature:
    _color_counter = 0
    _color_phase = random.random()

    def __init__(self, world, pos=(10, 20)):
        self.bodies = []
        self.joints = []
        # Fixed slot layout: up to 4 legs × 3 segments
        self.joint_grid = [[None for _ in range(3)] for _ in range(4)]

        # Per-creature torso color (slight variation) and reuse for flag cloth
        self.torso_color = self._next_pastel_color()

        # Torso
        self.body = world.CreateDynamicBody(position=pos)
        self.body.CreatePolygonFixture(
            box=(1, 0.5), density=1, friction=0.3,
            filter=b2Filter(categoryBits=0x0002, maskBits=0x0001)  # collide only with ground (cat 0x0001)
        )
        self.bodies.append((self.body, self.torso_color))

        # Ground goal flag (visual only, no collision)
        self._add_flag(world, pos)

        # Generate random legs (colors vary slightly around base)
        num_legs = random.randint(1, 4)
        for leg_idx in range(num_legs):
            leg_color = self.vary_color(BASE_LEG_COLOR, variation=25)
            leg_anchor = (pos[0] + 1.2 * random.choice([-1, 1]), pos[1] - 0.2)
            prev = self.body

            num_parts = random.randint(2, 3)  # segments = 2 or 3
            for seg_idx in range(num_parts):
                half_w = 0.2
                half_h = random.uniform(0.3, 0.8)

                leg = world.CreateDynamicBody(position=(leg_anchor[0], leg_anchor[1] - (seg_idx + 1)))
                leg.CreatePolygonFixture(
                    box=(half_w, half_h), density=1, friction=0.3,
                    filter=b2Filter(categoryBits=0x0002, maskBits=0x0001)  # only ground
                )
                self.bodies.append((leg, leg_color))

                jd = b2RevoluteJointDef(
                    bodyA=prev,
                    bodyB=leg,
                    localAnchorA=(0, -0.5 if prev is self.body else -half_h),
                    localAnchorB=(0,  half_h),
                    enableLimit=True,
                    lowerAngle=-0.8,
                    upperAngle=0.8,
                    enableMotor=True,
                    maxMotorTorque=50.0,
                    motorSpeed=0.0  # policy controls it
                )
                j = world.CreateJoint(jd)
                self.joints.append(j)

                # Assign joint to fixed slot index
                if leg_idx < 4 and seg_idx < 3:
                    self.joint_grid[leg_idx][seg_idx] = j

                prev = leg

    # ---------- Helpers ----------

    def _add_flag(self, world, pos):
        """
        Create a larger, non-colliding (sensor + filtered) ground flag ahead of the creature:
        - Thin pole planted on y=0
        - Small mount
        - Right-pointing pennant triangle (like Gym's BipedalWalker)
        - Pennant color matches this creature's torso color
        """
        # Place each creature’s flag ahead along x, pole planted on ground (y=0)
        self.flag_x = pos[0] + random.uniform(-20, 20)

        # Bigger visuals
        pole_height = 3.0
        pole_half_w = 0.08
        pole_half_h = pole_height / 2.0

        # Use a filter that collides with nothing (maskBits=0) and a distinct category (0x0004)
        flag_filter = b2Filter(categoryBits=0x0004, maskBits=0x0000)

        # Pole centered at (flag_x, pole_half_h) so base sits on ground
        flag_pole = world.CreateStaticBody(position=(self.flag_x, pole_half_h + 0.3))
        pole_fix = flag_pole.CreatePolygonFixture(
            box=(pole_half_w, pole_half_h),
            density=0,
            friction=0,
            isSensor=True,
            filter=flag_filter
        )

        # Mount near the top of the pole
        pennant_y = pole_height * 0.95 + 0.3
        mount_half_w = 0.12
        mount_half_h = 0.04
        flag_mount = world.CreateStaticBody(position=(self.flag_x + pole_half_w + mount_half_w, pennant_y))
        mount_fix = flag_mount.CreatePolygonFixture(
            box=(mount_half_w, mount_half_h),
            density=0,
            friction=0,
            isSensor=True,
            filter=flag_filter
        )

        # Pennant triangle pointing right (larger so it's visible)
        tri_len = 1.2
        tri_ht  = 0.6
        tri_vertices = [
            (0.0,  tri_ht / 2.0),   # top at mount
            (0.0, -tri_ht / 2.0),   # bottom at mount
            (tri_len, 0.0),         # tip to the right
        ]
        flag_tri = world.CreateStaticBody(position=(self.flag_x + pole_half_w + 2 * mount_half_w, pennant_y))
        tri_fix = flag_tri.CreatePolygonFixture(
            shape=b2PolygonShape(vertices=tri_vertices),
            density=0,
            friction=0,
            isSensor=True,
            filter=flag_filter
        )

        # Add to render list:
        # - pole grey
        # - mount darker grey
        # - pennant uses this creature's torso color (so it matches)
        self.bodies.append((flag_pole, (120, 120, 120)))
        self.bodies.append((flag_mount, (90, 90, 90)))
        self.bodies.append((flag_tri, self.torso_color))

    def vary_color(self, base_color, variation=20):
        r, g, b = base_color
        return (
            min(255, max(0, r + random.randint(-variation, variation))),
            min(255, max(0, g + random.randint(-variation, variation))),
            min(255, max(0, b + random.randint(-variation, variation))),
        )
    
    def _next_pastel_color(self):
        golden_turn = 0.38196601125
        h = (Creature._color_phase + golden_turn * Creature._color_counter) % 1.0
        Creature._color_counter += 1
        s = 0.45
        l = 0.78
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


