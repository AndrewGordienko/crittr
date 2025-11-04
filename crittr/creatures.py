import random
from Box2D import (
    b2World, b2PolygonShape, b2_dynamicBody, b2_staticBody,
    b2RevoluteJointDef, b2Filter
)

TORSO_COLOR = (173, 216, 230)
BASE_LEG_COLOR = (255, 182, 193)

class Creature:
    def __init__(self, world, pos=(10, 20)):
        self.bodies = []
        self.joints = []
        # Fixed slot layout: up to 4 legs × 3 segments
        self.joint_grid = [[None for _ in range(3)] for _ in range(4)]

        self.body = world.CreateDynamicBody(position=pos)
        self.body.CreatePolygonFixture(
            box=(1, 0.5), density=1, friction=0.3,
            filter=b2Filter(categoryBits=0x0002, maskBits=0x0001)
        )
        self.bodies.append((self.body, (173, 216, 230)))

        # Deterministic slot order: legs fill slots 0..num_legs-1
        num_legs = random.randint(1, 4)
        for leg_idx in range(num_legs):
            leg_color = self.vary_color((255, 182, 193), variation=25)
            leg_anchor = (pos[0] + 1.2 * random.choice([-1, 1]), pos[1] - 0.2)
            prev = self.body

            num_parts = random.randint(2, 3)  # segments = 2 or 3
            for seg_idx in range(num_parts):
                half_w = 0.2
                half_h = random.uniform(0.3, 0.8)

                leg = world.CreateDynamicBody(position=(leg_anchor[0], leg_anchor[1] - (seg_idx+1)))
                leg.CreatePolygonFixture(
                    box=(half_w, half_h), density=1, friction=0.3,
                    filter=b2Filter(categoryBits=0x0002, maskBits=0x0001)
                )
                self.bodies.append((leg, leg_color))

                jd = b2RevoluteJointDef(
                    bodyA=prev, bodyB=leg,
                    localAnchorA=(0, -0.5 if prev is self.body else -half_h),
                    localAnchorB=(0,  half_h),
                    enableLimit=True, lowerAngle=-0.8, upperAngle=0.8,
                    enableMotor=True, maxMotorTorque=50.0,
                    motorSpeed=0.0  # start neutral; policy will drive it
                )
                j = world.CreateJoint(jd)
                self.joints.append(j)

                # Fill slot if within 4×3
                if leg_idx < 4 and seg_idx < 3:
                    self.joint_grid[leg_idx][seg_idx] = j

                prev = leg


    def vary_color(self, base_color, variation=20):
        # Helper to shift a pastel color slightly
        r, g, b = base_color
        return (
            min(255, max(0, r + random.randint(-variation, variation))),
            min(255, max(0, g + random.randint(-variation, variation))),
            min(255, max(0, b + random.randint(-variation, variation))),
        )


        



