PLAYGROUND_SIZE = 2000
R = 25
R2 = R**2
SCAN_GAP = 25

SCAN_COUNT = PLAYGROUND_SIZE // SCAN_GAP + 1


def euclidean_distance(x1, y1, x2, y2):
    return (x2 - x1) ** 2 + (y2 - y1) ** 2


def drive():
    while left_bumper.pressed() == False:
        x = location.position(X, MM)
        y = location.position(Y, MM)

        if any([euclidean_distance(x, y, p[0], p[1]) < R2 for p in IMAGE_POINTS]):
            pen.move(DOWN)

        drivetrain.drive(FORWARD)
        pen.move(UP)

        wait(5, MSEC)


def scan_line(direction, SCAN_GAP):
    drivetrain.turn_for(direction, 90, DEGREES)
    drivetrain.drive_for(FORWARD, SCAN_GAP, MM)
    drivetrain.turn_for(direction, 90, DEGREES)
    drive()


def when_started1():
    drivetrain.set_drive_velocity(100, PERCENT)
    drivetrain.set_turn_velocity(100, PERCENT)

    drive()

    drivetrain.turn_for(LEFT, 90, DEGREES)
    drive()

    for _ in range(SCAN_COUNT):
        scan_line(LEFT, SCAN_GAP)
        scan_line(RIGHT, SCAN_GAP)
