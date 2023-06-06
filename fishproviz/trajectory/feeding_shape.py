from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import fishproviz.config as config
from fishproviz.utils.feeding_maze_config import MAZE, read_maze_data_from_json


class FeedingShape(ABC):
    """Class for feeding shapes"""

    @abstractmethod
    def contains(self, data_points, fish_key, day=None):
        """Checks which points are inside the shape"""
        pass


class FeedingEllipse(FeedingShape):
    """Class for feeding ellipses"""

    def __init__(self):
        """Initializes the ellipse data"""
        self.dict_ellipses = read_maze_data_from_json()

    # overrideing the contains method
    def contains(self, data_points, fish_key, day=None):
        """Checks which points are inside the ellipse, returns these points and a linspace of the ellipse"""
        try:
            ellipse = self.dict_ellipses[fish_key][day][MAZE]
        except KeyError as e:
            print(e, "No ellipse data for fish %s on day %s" % (fish_key, day))
            return pd.DataFrame(columns=data_points.columns), np.array([(0, 0), (0, 0)])
        ori_x = (ellipse["end_x"] + ellipse["origin_x"]) / 2
        ori_y = (ellipse["end_y"] + ellipse["origin_y"]) / 2
        a = (ellipse["end_x"] - ellipse["origin_x"]) / 2
        b = (ellipse["end_y"] - ellipse["origin_y"]) / 2
        in_ellipse = (
            (data_points["xpx"] - ori_x) ** 2 / a**2
            + (data_points["ypx"] - ori_y) ** 2 / b**2
        ) <= 1
        return data_points[in_ellipse], np.array(
            [
                (ori_x + a * np.cos(alpha), ori_y + b * np.sin(alpha))
                for alpha in np.linspace(0, 2 * np.pi, 100)
            ]
        )


class FeedingRectangle(FeedingShape):
    """Class for feeding squares"""

    def __init__(self):
        """Initializes the square data"""
        self.patches = get_feeding_patches()

    # overrideing the contains method
    def contains(self, data_points, fish_key, day=None):
        """Checks which points are inside the square"""
        return get_feeding_box(data_points, *self.patches[fish_key])


def get_feeding_patches():
    patches = pd.read_csv("data/feeding_patch_coords.csv", delimiter=";")
    return dict(
        zip(
            map(
                lambda stuff: "_".join(map(str, stuff)),
                patches[["camera_id", "front_or_back"]].to_numpy(),
            ),
            patches[["TL_x", "TL_y", "TR_x", "TR_y"]].to_numpy(),
        )
    )


def find_cords(camera_id, position, csv):
    f1 = csv["camera_id"] == int(camera_id)
    f2 = csv["front_or_back"] == position
    cords = csv[f1 & f2][["TL_x", "TL_y", "TR_x", "TR_y"]]
    if cords.empty:
        raise Exception(
            "camera: %s with position %s is not known" % (camera_id, position)
        )
    return cords.to_numpy()[0]


def get_feeding_cords(data, camera_id, is_back):
    pos = config.BACK if is_back else config.FRONT
    patches = pd.read_csv("data/feeding_patch_coords.csv", delimiter=";")
    return get_feeding_box(data, *find_cords(camera_id, pos, patches))


def get_feeding_box(data, TL_x, TL_y, TR_x, TR_y):
    scale = 2
    # if x has the same value.
    if abs(TL_x - TR_x) < abs(TL_y - TR_y):
        # config.FRONT
        p_len = abs(TL_y - TR_y) * scale
        f1 = data["xpx"] > TL_x - p_len
        f2 = data["ypx"] < TL_y + p_len
        f3 = data["ypx"] > TR_y - p_len
        TL_y += p_len
        TR_y -= p_len
        box = np.array(
            [
                (TL_x, TL_y),
                (TL_x - p_len, TL_y),
                (TR_x - p_len, TR_y),
                (TR_x, TR_y),
                (TL_x, TL_y),
            ]
        )
    else:
        # config.BACK
        p_len = abs(TL_x - TR_x) * scale
        f1 = data["ypx"] > TR_y - p_len
        f2 = data["xpx"] > TL_x - p_len
        f3 = data["xpx"] < TR_x + p_len
        TL_x -= p_len
        TR_x += p_len
        box = np.array(
            [
                (TL_x, TL_y),
                (TL_x, TL_y - p_len),
                (TR_x, TR_y - p_len),
                (TR_x, TR_y),
                (TL_x, TL_y),
            ]
        )
    feeding = data[f1 & f2 & f3]
    return feeding, box
