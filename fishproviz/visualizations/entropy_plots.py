import pandas as pd
import numpy as np
import plotly.graph_objects as go
from .activity_plotting import get_filepath_metric_plot
import fishproviz.config as config
from fishproviz.metrics.metrics import entropy_for_chunk
from fishproviz.utils.tank_area_config import get_area_functions
from fishproviz.utils import (
    csv_of_the_day,
    get_all_days_of_context,
    get_camera_pos_keys,
    all_error_filters,
)
from fishproviz.utils.utile import flatten_list

FIG_NAME = "entropy_density"
ENTROPY_DIR = "entropy_density"


def entropy_density_main():
    fish_keys = get_camera_pos_keys()
    days = get_all_days_of_context()
    area_f = get_area_functions()
    for fk in fish_keys:
        cam, pos = fk.split("_")
        is_back = pos == config.BACK
        start = 0
        days_per_week = 7
        data_by_week = []
        for week_i, end in enumerate(
            range(days_per_week, len(days) + 1, days_per_week)
        ):
            area_tuple = (fk, area_f(fk))
            data_batches = flatten_list(
                [
                    csv_of_the_day(cam, day, is_back=is_back)[1]
                    for day in days[start:end]
                ]
            )
            if len(data_batches) == 0:
                data_by_week.append(np.empty((0, 2)))
                continue
            data = pd.concat(data_batches)
            data_px = data[["xpx", "ypx"]].to_numpy()
            filter_d = all_error_filters(data_px, area_tuple)
            data_by_week.append(data_px[~filter_d])
            entropy_density_plot(
                data_by_week[week_i],
                area_tuple,
                fig_name="%s_week_%s_%s" % (fk, week_i, FIG_NAME),
                fish_key=fk,
                zmax=4000,
                timewindow="week_%s" % (week_i + 1),
            )
            start = end
        data_overall = np.concatenate(data_by_week)
        _ = entropy_density_plot(
            data_overall,
            area_tuple,
            fig_name="%s_overall_%s" % (fk, FIG_NAME),
            fish_key=fk,
            zmax=9000,
            timewindow="overall",
        )


def entropy_density_plot(data, area_tuple, **kwargs):
    ent_val = entropy_for_chunk(data, area_tuple)
    fig = draw_density_entropy(data, entropy_value=ent_val, **kwargs)
    return fig


def draw_density_entropy(
    data,
    fig_name=FIG_NAME,
    entropy_value=0,
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
        title="%s entropy: %.4f \t fish_key: %s"
        % (timewindow, entropy_value, fish_key),
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
