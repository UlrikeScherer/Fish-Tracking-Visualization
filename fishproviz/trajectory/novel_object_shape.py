from fishproviz.trajectory.shapes import Shape
import pandas as pd
import numpy as np
from fishproviz.utils.object_config import read_object_data_from_json
from fishproviz.utils.social_zone_config import read_social_zone_data_from_json


class ObjectEllipse(Shape):
    """Class for feeding ellipses"""

    def __init__(self, sociability: bool = False):
        """Initializes the ellipse data"""
        if sociability:
            self.dict_ellipses = read_social_zone_data_from_json()
        else:
            self.dict_ellipses = read_object_data_from_json()

    # overrideing the contains method
    def contains(self, data_points, fish_key, day=None, sociability_zone: str=None):
        """Checks which points are inside the ellipse, returns these points and a linspace of the ellipse"""
        try:
            ellipse = self.dict_ellipses[fish_key][day]
        except KeyError as e:
            print(e, "No ellipse data for fish %s on day %s" % (fish_key, day))
            return pd.DataFrame(columns=data_points.columns), np.array([(0, 0), (0, 0)])
        if sociability_zone is not None:
            try:
                ellipse = ellipse[sociability_zone]
            except KeyError as e:
                print(e, f"Sociability zone {sociability_zone} for fish {fish_key} on day {day} does not match existing zones {list(ellipse.keys())}.")

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
