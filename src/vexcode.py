from typing import Literal

# Both of the below should mirror vexcode's API for a subset of drivetrain and pen functions


class drivetrain:
    @staticmethod
    def turn_to_heading(angle: float = 0, units: Literal["DEGREES"] = "DEGREES"):
        return f"drivetrain.turn_to_heading({angle}, {units})"

    @staticmethod
    def drive_for(
        direction: Literal["FORWARD", "REVERSE"] = "FORWARD",
        distance: float = 0,
        units: Literal["INCHES", "MM"] = "MM",
    ):
        return f"drivetrain.drive_for({direction}, {distance}, {units})"

    @staticmethod
    def set_drive_velocity(amount: int, units: Literal["PERCENT"] = "PERCENT"):
        return f"drivetrain.set_drive_velocity({amount}, {units})"

    @staticmethod
    def set_turn_velocity(amount: int, units: Literal["PERCENT"] = "PERCENT"):
        return f"drivetrain.set_turn_velocity({amount}, {units})"


class pen:
    @staticmethod
    def move(direction: Literal["DOWN", "UP"] = "DOWN"):
        return f"pen.move({direction})"

    @staticmethod
    def set_pen_width(
        width: Literal[
            "EXTRA_THIN", "THIN", "MEDIUM", "WIDE", "EXTRA_WIDE"
        ] = "EXTRA_THIN"
    ):
        return f"pen.set_pen_width({width})"

    @staticmethod
    def set_pen_color_rgb(r: int, g: int, b: int, a: int = 100):
        return f"pen.set_pen_color_rgb({r}, {g}, {b}, {a})"


ROBOT_SIZE = 100
PLAYGROUND_SIZE = 2000 - (2 * ROBOT_SIZE)  # account for size of robot; ~200

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
    {drivetrain.set_drive_velocity(100)}
    {drivetrain.set_turn_velocity(100)}
    generate_image()

vr_thread(when_started1)
"""
