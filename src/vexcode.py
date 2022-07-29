def turn_for(direction: str = "LEFT", angle: float = 0, units: str = "DEGREES"):
    return f"drivetrain.turn_to_heading({angle}, {units})"


def drive_for(direction: str = "FORWARD", distance: float = 0, units: str = "MM"):
    return f"drivetrain.drive_for({direction}, {distance}, {units})"


ROBOT_SIZE = 100
PLAYGROUND_SIZE = 2000 - 200

PREAMBLE = """
#region VEXcode Generated Robot Configuration
import math
import random
from vexcode_vr_enhanced_robot import *

drivetrain = Drivetrain()
magnet = Electromagnet("magnet", 0)
pen = Pen()
brain = Brain()
left_bumper = Bumper("leftBumper", 1)
right_bumper = Bumper("rightBumper", 2)
front_eye = EyeSensor("fronteye", 3)
down_eye = EyeSensor("downeye", 4)
right_eye = EyeSensor("righteye", 5)
left_eye = EyeSensor("lefteye", 6)
rear_eye = EyeSensor("reareye", 7)
front_distance = Distance("frontdistance", 8)
rear_distance = Distance("reardistance", 9)
left_distance = Distance("leftdistance", 10)
right_distance = Distance("rightdistance", 11)
location = Location()
pen.set_pen_width(THIN)
distance = front_distance
#endregion VEXcode Generated Robot Configuration
"""


POSTAMBLE = f"""
def when_started1():
    drivetrain.set_drive_velocity(100, PERCENT)
    drivetrain.set_turn_velocity(100, PERCENT)

    generate_image()

vr_thread(when_started1)
"""
