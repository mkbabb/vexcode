"""
Functions used to generate VEXcode (https://vr.vex.com/)
robot API calls that allow for the drawing of an arbitrary image to the screen.
"""

import colorsys
from typing import Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree, distance
from sklearn.metrics import pairwise_distances

from vexcode import PLAYGROUND_SIZE, POSTAMBLE, PREAMBLE, drivetrain, pen
import rtree


def scale_to_aspect_ratio(n: int, aspect_ratio: float):
    if aspect_ratio <= 1:
        return n, int(n * aspect_ratio)
    else:
        return n, int(n / aspect_ratio)


def make_direction_vector(angle: float):
    angle = np.deg2rad(angle)
    return np.array([np.cos(angle), np.sin(angle)])


def linear_map(t: float, x1: float, y1: float, x2: float = 0.0, y2: float = 1.0):
    a = (x2 - y2) / (x1 - y1)
    b = x2 - (x2 - y2) / (x1 - y1) * x1
    return a * t + b


def rainbow_point(x: float, x1: float, y1: float):
    hue = linear_map(x, x1, y1)
    return tuple(map(lambda x: int(x * 255), colorsys.hsv_to_rgb(hue, 1.0, 1.0)))


def pad_3d(arr: np.ndarray):
    return np.hstack((arr, np.ones((len(arr), 1))))


def to_2d(arr: np.ndarray):
    return arr[..., :2]


def translate(tx: float, ty: float):
    T = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [tx, ty, 1],
        ]
    )

    return T


def scale(sx: float, sy: float):
    S = np.array(
        [
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1],
        ]
    )

    return S


def rotate(degrees: float = 0):
    angle = np.deg2rad(degrees)

    R = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )

    return R.T


def normalize(arr: np.ndarray):
    t_min, t_max = np.min(arr), np.max(arr)

    s = 1 / (t_max - t_min)

    return scale(s, s) @ translate(t_min, t_min)


def center(arr: np.ndarray):
    x_min, y_min = np.min(arr, axis=0)
    x_max, y_max = np.max(arr, axis=0)

    dx = np.abs(x_max - x_min)
    dy = np.abs(y_max - y_min)

    return translate(-dx / 2, -dy / 2)


def masked_argmin(arr: np.ndarray, mask: np.ndarray):
    masked = np.ma.array(arr, mask=mask)
    return masked.argmin()


def order_points(
    points: np.ndarray,
):

    tree = rtree.Index()
    for n, p in enumerate(points):
        left, bottom = p
        tree.insert(n, (left, bottom, left, bottom))

    ixs = []
    distances = []

    ix = np.argmin(points[..., 0])
    p = points[ix]

    ixs.append(ix)
    distances.append(0)

    tree.delete(ix, p)

    for _ in range(len(points) - 1):
        neighbors = tree.nearest(p, num_results=1)

        ix = next(neighbors)
        p_new = points[ix]

        ixs.append(ix)
        distances.append(np.linalg.norm(p_new - p))

        p = p_new

        tree.delete(ix, p)

    return points[ixs], np.asarray(distances)


def load_image(image_path: str, size=None):
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    if size is not None:
        im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)

    return im


def find_contours(im: np.ndarray, threshold1: int = 150, threshold2: int = 255):
    im = cv2.medianBlur(im, 1)
    im = cv2.Canny(im, threshold1, threshold2)
    ret, im = cv2.threshold(im, threshold1, threshold2, 0)

    # Good params for findContours
    contours, _ = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [np.squeeze(i, axis=1) for i in contours]

    return contours


def make_traceable_path(points: np.ndarray):
    """Trace out a path within the points array, getting the distance and angle
    between each consecutive pair of points, e.g:

        points = p1, p2, p3 ... p_n
        path = (p1, p2), (p2, p3) ... (p_n - 1, p_n)
    """
    x_min, x_max = np.min(points[..., 0]), np.max(points[..., 0])

    image_path = []

    prev_p = make_direction_vector(0)

    for p in points:
        delta = p - prev_p

        distance = np.linalg.norm(delta)
        angle = np.rad2deg(np.arctan2(delta[1], delta[0]))
        # 0.05 error due to bug with rotation locking in vexcode.
        # + 90 as the robot rotates relative to pi/2
        angle = ((360.05 - angle) + 90) % 360

        color = rainbow_point(p[0], x_min, x_max)

        path_i = {
            "distance": np.round(distance, 2),
            "angle": np.round(angle, 2),
            "p": p,
            "color": color,
        }
        image_path.append(path_i)

        prev_p = p

    return image_path


def generate_code(
    points: np.ndarray,
    distances: np.ndarray,
    mode: Literal["all", "image_path"] = "image_path",
) -> str:
    """Generates the actual code for the robot.
    Bulk of the work is done in make_traceable_path,
    though we optimize the number or drives, turns, and color changes."""
    if mode == "all":
        return f"IMAGE_POINTS = {points.tolist()}"
    elif mode == "image_path":
        image_path = make_traceable_path(points)

        avg_dist = np.average(distances)

        lines = []

        prev_angle = None
        prev_color = None

        thin = None

        for n, p in enumerate(image_path):
            angle, distance, color = p["angle"], p["distance"], p["color"]

            if prev_angle != angle:  # optimizing consecutive same-angle turns
                lines.append(drivetrain.turn_to_heading(angle))
            if prev_color != color:  # similar as above
                lines.append(pen.set_pen_color_rgb(*color))

            if distance > avg_dist:
                # Extra extra thin not currently supported
                lines.append(pen.set_pen_width())
                thin = True
            elif thin:
                lines.append(pen.set_pen_width())
                thin = False

            if n == 1:
                lines.append(pen.move())

            lines.append(drivetrain.drive_for(distance=distance))

            prev_angle = angle
            prev_color = color

        lines.append(pen.move("UP"))

        generate_image_body = "\n".join(f"\t{i}" for i in lines)

        return f"""
{PREAMBLE}
    
def generate_image():
{generate_image_body}

{POSTAMBLE}"""


def main(image_path: str, n: int = 50, plot: bool = False):
    im = load_image(image_path)

    height, width = im.shape
    aspect_ratio = width / height

    # scale the image down; performance savings.
    # INTER_AREA is good for down-scaling.
    im = cv2.resize(
        im, scale_to_aspect_ratio(n, aspect_ratio), interpolation=cv2.INTER_AREA
    )

    contours = find_contours(im)
    all_points = np.vstack((*contours,))
    all_points = np.unique(all_points, axis=0)
    all_points = np.insert(all_points, 0, [0, 0], axis=0)

    # Potential code for maximizing playground draw space.
    # not feasible due to the poor N^2 implementation of order_points.

    # all_points = [tuple(reversed(i)) for i in all_points]
    # t_im = np.zeros(im.shape, dtype=np.uint8)
    # for p in all_points:
    #     t_im[p] = 1

    N, M = scale_to_aspect_ratio(PLAYGROUND_SIZE, aspect_ratio)

    # t_im = cv2.resize(t_im, (N, M), interpolation=cv2.INTER_AREA)

    # all_points = np.transpose(np.nonzero(t_im))

    all_points = all_points.astype(float)

    TSR = center(all_points) @ rotate(180) @ normalize(all_points) @ scale(-1 * N, M)

    all_points = pad_3d(all_points) @ TSR
    all_points = to_2d(all_points)

    all_points, distances = order_points(all_points)

    code = generate_code(all_points, distances)

    with open("./dist/generated.py", "w") as file:
        file.write(code)

    if plot:
        fig, ax = plt.subplots(figsize=(5 * aspect_ratio, 5))

        # plt.scatter(all_points[..., 0], all_points[..., 1], marker=".")
        plt.plot(all_points[..., 0], all_points[..., 1], marker=".")
        plt.show()


if __name__ == "__main__":
    # image_path = "assets/SpongeBob_SquarePants_character.svg.png"
    image_path = "assets/deiywwr-0d9719ae-59b0-4fbc-adc3-a26de869d84e.png"
    n = 1000

    main(image_path=image_path, n=n, plot=True)
