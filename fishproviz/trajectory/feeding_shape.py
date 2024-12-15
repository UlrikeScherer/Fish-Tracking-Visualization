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


class FeedingPatch(FeedingShape):
    """Class for feeding squares"""

    def __init__(self):
        """Initializes the square data"""
        self.patches = get_feeding_patches()

    # overrideing the contains method
    def contains(self, data_points, fish_key, day=None):
        """Checks which points are inside the square"""
        return get_feeding_box(data_points, *self.patches[fish_key])


def get_feeding_patches():
    patches = pd.read_csv(config.FEEDING_PATCH_COORDS_FILE, delimiter=config.FEEDING_PATCH_COORDS_SEP)
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
    patches = pd.read_csv(config.FEEDING_PATCH_COORDS_FILE, delimiter=config.FEEDING_PATCH_COORDS_SEP)
    return get_feeding_box(data, *find_cords(camera_id, pos, patches))


def get_feeding_box(data, TL_x, TL_y, TR_x, TR_y):
    cm_len = config.MAGNET_LENGTH_CM
    p_len = np.sqrt((TL_y - TR_y) ** 2 + (TL_x - TR_x) ** 2)
    midpoint_y = abs(TL_y + TR_y) / 2
    midpoint_x = abs(TL_x + TR_x) / 2

    target_width = config.FEEDING_SHAPE_WIDTH * p_len / cm_len
    target_height = config.FEEDING_SHAPE_HEIGHT * p_len / cm_len
    # if x has the same value.
    if abs(TL_x - TR_x) < abs(TL_y - TR_y):
        # config.FRONT
        f1 = data["xpx"] > midpoint_x - target_height
        f2 = data["ypx"] < midpoint_y + target_width/2
        f3 = data["ypx"] > midpoint_y - target_width/2
        f4 = data["xpx"] < midpoint_x
        box = np.array(
            [
                (midpoint_x, midpoint_y + target_width/2),
                (midpoint_x - target_height, midpoint_y + target_width/2),
                (midpoint_x - target_height, midpoint_y - target_width/2),
                (midpoint_x, midpoint_y - target_width/2),
                (midpoint_x, midpoint_y + target_width/2),
            ]
        )
    else:
        # config.BACK
        f1 = data["ypx"] > midpoint_y - target_height
        f2 = data["xpx"] < midpoint_x + target_width/2
        f3 = data["xpx"] > midpoint_x - target_width/2
        f4 = data["ypx"] < midpoint_y
        box = np.array(
            [
                (midpoint_x - target_width/2, midpoint_y),
                (midpoint_x - target_width/2, midpoint_y - target_height),
                (midpoint_x + target_width/2, midpoint_y - target_height),
                (midpoint_x + target_width/2, midpoint_y),
                (midpoint_x - target_width/2, midpoint_y),
            ]
        )
    feeding = data[f1 & f2 & f3 & f4]
    return feeding, box
