import pandas as pd
import numpy as np
import plotly.graph_objects as go
from .activity_plotting import get_filepath_metric_plot
import fishproviz.config as config
from fishproviz.metrics.compute_primitives import entropy_for_chunk
from fishproviz.utils.tank_area_config import get_area_functions
from fishproviz.utils import (
    csv_of_the_day,
    get_all_days_of_context,
    get_camera_pos_keys,
    all_error_filters,
)

FIG_NAME = "entropy_density"
ENTROPY_DIR = "entropy_density"
DAYS_PER_WEEK = 7


def split_weeks(days: list[str], days_per_week: int = DAYS_PER_WEEK) -> list[list[str]]:
    """Consecutive groups of days_per_week days; the last group holds the leftover days."""
    return [days[start : start + days_per_week] for start in range(0, len(days), days_per_week)]


def valid_points_of_day(cam: str, day: str, fish_key: str, area_tuple: tuple[str, np.ndarray], is_back: bool) -> np.ndarray:
    """Error-filtered (xpx, ypx) points of one camera-day."""
    batches = csv_of_the_day(cam, day, is_back=is_back)[1]
    if len(batches) == 0:
        return np.empty((0, 2))
    data_px = pd.concat(batches)[["xpx", "ypx"]].to_numpy()
    return data_px[~all_error_filters(data_px, area_tuple, fish_key=fish_key, day=day)]


def week_label(week_i: int, n_days: int) -> str:
    """Window label; a week with fewer than 7 days carries its day count."""
    return "week_%s" % (week_i + 1) if n_days == DAYS_PER_WEEK else "week_%s_%sdays" % (week_i + 1, n_days)


def entropy_density_main() -> None:
    fish_keys = get_camera_pos_keys()
    days = get_all_days_of_context()
    area_f = get_area_functions()
    for fk in fish_keys:
        cam, pos = fk.split("_")
        is_back = pos == config.BACK
        area_tuple = (fk, area_f(fk))
        points_by_day = {day: valid_points_of_day(cam, day, fk, area_tuple, is_back) for day in days}
        if sum(len(points) for points in points_by_day.values()) == 0:
            raise ValueError("No valid points for fish %s over %d days" % (fk, len(days)))
        for week_i, week_days in enumerate(split_weeks(days)):
            label = week_label(week_i, len(week_days))
            entropy_density_plot(
                np.concatenate([points_by_day[day] for day in week_days]),
                area_tuple,
                fig_name="%s_%s_%s" % (fk, label, FIG_NAME),
                fish_key=fk,
                zmax=4000,
                timewindow=label,
            )
        entropy_density_plot(
            np.concatenate(list(points_by_day.values())),
            area_tuple,
            fig_name="%s_overall_%s" % (fk, FIG_NAME),
            fish_key=fk,
            zmax=9000,
            timewindow="overall",
        )


def entropy_density_plot(data: np.ndarray, area_tuple: tuple[str, np.ndarray], **kwargs):
    ent_val, n_valid, n_out_of_range = entropy_for_chunk(data, area_tuple[1])
    return draw_density_entropy(data, entropy_value=ent_val, n_valid=n_valid, n_out_of_range=n_out_of_range, **kwargs)


def draw_density_entropy(
    data,
    fig_name=FIG_NAME,
    entropy_value=0,
    n_valid=0,
    n_out_of_range=0,
    fish_key="undefined",
    zmax=None,
    timewindow="",
):
    x = data[:, 0]
    y = data[:, 1]
    res = go.Histogram2dContour(
        x=x,
        y=y,
        showscale=True,
        autocontour=True,
        contours=dict(coloring="fill", showlines=False),
        ncontours=40,
        # zmid=100,
        zmin=0,
        zmax=zmax,
    )
    fig = go.Figure([res])
    fig.update_layout(
        title="%s entropy: %.4f \t fish_key: %s \t n: %d, out of range: %d" % (timewindow, entropy_value, fish_key, n_valid, n_out_of_range),
        font={"size": 16},
        height=600,
        width=650,
        showlegend=True,
        margin_t=70,
        margin_b=1,
        margin_l=1,
        margin_r=1,
    )
    fig.update_xaxes(visible=True)
    fig.update_yaxes(visible=True)
    fig.write_image(
        get_filepath_metric_plot(fig_name, subdir=ENTROPY_DIR, ext="png"),
        format="png",
        engine="kaleido",
    )
    return fig


if __name__ == "__main__":
    entropy_density_main()
