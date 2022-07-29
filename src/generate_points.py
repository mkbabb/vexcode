import colorsys
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree, distance
from sklearn.metrics import pairwise_distances

import src.vexcode as vexcode


def masked_argmin(arr: np.ndarray, mask: np.ma.array):
    masked = np.ma.array(arr, mask=mask)
    return masked.argmin()


def order_points(
    points: np.ndarray,
):
    # dist_mat = distance.squareform(distance.pdist(points, "euclidean"), checks=False)
    dist_mat = pairwise_distances(points, metric="euclidean", n_jobs=-1)
    mask = np.full(len(points), False)
    ix = np.argmin(points[..., 0])

    ixs = []
    distances = []

    for _ in range(len(points)):
        row = dist_mat[ix]

        new_ix = masked_argmin(row, mask)
        mask[new_ix] = True

        d = row[new_ix]

        distances.append(d)
        ixs.append(new_ix)

        ix = new_ix

    return points[ixs], np.asarray(distances)


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


def rotate(degrees=0):
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


def load_image(path, size=None):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    if size is not None:
        im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)

    return im


def find_contours(im, threshold1=150, threshold2=255):
    im = cv2.medianBlur(im, 1)
    im = cv2.Canny(im, threshold1, threshold2)
    ret, im = cv2.threshold(im, threshold1, threshold2, 0)

    contours, _ = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [np.squeeze(i, axis=1) for i in contours]

    return contours


def make_direction_vector(angle: float):
    angle = np.deg2rad(angle)
    return np.array([np.cos(angle), np.sin(angle)])


def linear_map(t: float, x1: float, y1: float, x2=0.0, y2=1.0):
    a = (x2 - y2) / (x1 - y1)
    b = x2 - (x2 - y2) / (x1 - y1) * x1
    return a * t + b


def rainbow_point(x: float, x1: float, y1: float):
    hue = linear_map(x, x1, y1)
    return tuple(map(lambda x: int(x * 255), colorsys.hsv_to_rgb(hue, 1.0, 1.0)))


def make_path(points: np.array):
    x_min, x_max = np.min(points[..., 0]), np.max(points[..., 0])

    path = []

    prev_p = make_direction_vector(0)

    for i in range(1, len(points)):
        p = points[i]

        delta = p - prev_p
        distance = np.linalg.norm(delta)

        angle = np.rad2deg(np.arctan2(delta[1], delta[0]))
        angle = ((360.05 - angle) + 90) % 360

        color = rainbow_point(p[0], x_min, x_max)

        path_i = {
            "ix": i,
            "distance": np.round(distance, 2),
            "angle": np.round(angle, 2),
            "p": p,
            "color": color,
        }

        path.append(path_i)

        prev_p = p

    return path


def generate_code(points: np.ndarray, distances: np.ndarray, mode="path") -> str:
    if mode == "all":
        return f"IMAGE_POINTS = {points.tolist()}"
    elif mode == "path":
        path = make_path(points)

        avg_dist = np.average(distances)

        lines = []

        prev_angle = None
        prev_color = None

        thin = None

        for n, p in enumerate(path):
            angle, distance, color = p["angle"], p["distance"], p["color"]

            if prev_angle != angle:
                lines.append(vexcode.turn_for(angle=angle))
            if prev_color != color:
                lines.append(
                    f"pen.set_pen_color_rgb({color[0]}, {color[1]}, {color[2]}, 100)"
                )

            if distance > avg_dist:
                # Extra extra thin not currently supported
                lines.append("pen.set_pen_width(EXTRA_THIN)")
                thin = True
            elif thin:
                lines.append("pen.set_pen_width(EXTRA_THIN)")
                thin = False

            if n == 1:
                lines.append("pen.move(DOWN)")

            lines.append(vexcode.drive_for(distance=distance, units="MM"))

            prev_angle = angle
            prev_color = color

        lines.append("pen.move(UP)")

        generate_image_body = "\n".join(f"\t{i}" for i in lines)

        return f"""
{vexcode.PREAMBLE}
    
def generate_image():
{generate_image_body}

{vexcode.POSTAMBLE}"""


def scale_to_aspect_ratio(n, aspect_ratio):
    if aspect_ratio <= 1:
        return n, int(n * aspect_ratio)
    else:
        return n, int(n / aspect_ratio)


# path = "assets/1000_F_106135417_llsURwzfzT2aFy7mhWQ2QmLQmmYqGwLs-removebg-preview.png"
# path = "assets/rubber-duck-removebg-preview.png"
# path = "/Users/mkbabb/Programming/fourier-animate/assets/IMG_7304.jpg"
# path = "assets/smiley.png"
# path = "/Users/mkbabb/Programming/fourier-animate/assets/Square_-_black_simple.svg.png"
# path = "assets/SpongeBob_SquarePants_character.svg.png"
# path = "assets/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.png"
# path = (
#     "assets/Mona_Lisa__by_Leonardo_da_Vinci__from_C2RMF_retouched-removebg-preview.png"
# )
# path = "assets/SMASTitle.png"
# path = "assets/super-mario-world---button-fin-1567640652381-removebg-preview.png"
# path = "assets/375.jpg"
# path = "assets/MV5BMGVlOGI0OTctZGZmOC00ZGE5LWE4YzUtOGVjZjVkYTcyY2Q3XkEyXkFqcGdeQXVyMTA0MTM5NjI2._V1_FMjpg_UX1000_.jpg"
# path = "assets/71b6S5TlExL._SL1500_.jpg"
# path = "assets/329527.jpg"
# path = "assets/Eraserhead-and-Present-Mic.jpg"
path = "assets/deiywwr-0d9719ae-59b0-4fbc-adc3-a26de869d84e.png"
# path = "assets/qrcode.png"
# path = "assets/FO0S0IRWUAIF85e.png"

im = load_image(path)

height, width = im.shape
aspect_ratio = width / height

n = 500

im = cv2.resize(
    im, scale_to_aspect_ratio(n, aspect_ratio), interpolation=cv2.INTER_AREA
)


fig, ax = plt.subplots(figsize=(5 * aspect_ratio, 5))


contours = find_contours(im)
all_points = np.vstack((*contours,))
all_points = np.unique(all_points, axis=0)
all_points = np.insert(all_points, 0, [0, 0], axis=0)
# all_points = [tuple(reversed(i)) for i in all_points]

# t_im = np.zeros(im.shape, dtype=np.uint8)
# for p in all_points:
#     t_im[p] = 1

N, M = scale_to_aspect_ratio(vexcode.PLAYGROUND_SIZE, aspect_ratio)
# t_im = cv2.resize(t_im, (N, M), interpolation=cv2.INTER_AREA)

# all_points = np.transpose(np.nonzero(t_im))
all_points = all_points.astype(float)

TSR = center(all_points) @ rotate(180) @ normalize(all_points) @ scale(-1 * N, N)

all_points = pad_3d(all_points) @ TSR
all_points = to_2d(all_points)


all_points, distances = order_points(all_points)

# plt.scatter(all_points[..., 0], all_points[..., 1], marker=".")
plt.plot(all_points[..., 0], all_points[..., 1], marker=".")
plt.show()

code = generate_code(all_points, distances)

with open("./generated.py", "w") as file:
    file.write(code)
