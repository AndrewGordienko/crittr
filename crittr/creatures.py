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

        self.body = world.CreateDynamicBody(position=pos)
        self.body.CreatePolygonFixture(
            box=(1, 0.5), density=1, friction=0.3,
            filter=b2Filter(categoryBits=0x0002, maskBits=0x0001) # collide only with the ground
        )

        self.bodies.append((self.body, TORSO_COLOR))
        num_legs = random.randint(1, 4)

        for i in range(num_legs):
            leg_color = self.vary_color(BASE_LEG_COLOR, variation=25)
            leg_anchor = (pos[0] + 1.2 * random.choice([-1, 1]), pos[1] - 0.2)
            prev = self.body
            num_parts = random.randint(2, 3)

            for j in range(num_parts):
                # Random segment length
                half_w = 0.2
                half_h = random.uniform(0.3, 0.8)

                leg = world.CreateDynamicBody(position=(leg_anchor[0], leg_anchor[1] - (j+1)))
                leg.CreatePolygonFixture(
                    box=(half_w, half_h), density=1, friction=0.3,
                    filter=b2Filter(categoryBits=0x0002, maskBits=0x0001)  # collide only with ground
                )
                self.bodies.append((leg, leg_color))

                # Revolute joint with motor
                joint_def = b2RevoluteJointDef(
                    bodyA=prev,
                    bodyB=leg,
                    localAnchorA=(0, -0.5 if prev is self.body else -half_h),
                    localAnchorB=(0, half_h),
                    enableLimit=True,
                    lowerAngle=-0.8,
                    upperAngle=0.8,
                    enableMotor=True,
                    maxMotorTorque=50.0,
                    motorSpeed=random.uniform(-2.0, 2.0)
                )
                joint = world.CreateJoint(joint_def)
                self.joints.append(joint)

                prev = leg

    def vary_color(self, base_color, variation=20):
        # Helper to shift a pastel color slightly
        r, g, b = base_color
        return (
            min(255, max(0, r + random.randint(-variation, variation))),
            min(255, max(0, g + random.randint(-variation, variation))),
            min(255, max(0, b + random.randint(-variation, variation))),
        )


        



