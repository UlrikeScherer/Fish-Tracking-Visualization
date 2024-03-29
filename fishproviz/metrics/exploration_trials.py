import os
import fishproviz.config as config
from fishproviz.utils import (
    get_days_in_order,
    get_camera_pos_keys,
    csv_of_the_day,
    get_seconds_from_time,
    start_time_of_day_to_seconds,
)
from fishproviz.utils.transformation import pixel_to_cm
from fishproviz.metrics import activity
from fishproviz.utils.error_filter import error_default_points
import pandas as pd

BLOCK = "block1"


def exploration_trials(path_trials=config.TRIAL_TIMES_CSV):
    """
    This function takes the path to the csv file containing the trial times and
    writes the same csv file into the results folder with activity for the exploration times
    """
    trial_name = path_trials.split("/")[-2]
    res_mean = f"{ config.RESULTS_PATH}/{trial_name}_mean.csv"
    res_ndf = f"{ config.RESULTS_PATH}/{trial_name}_ndf.csv"

    if not os.path.exists(res_mean):
        tdf = pd.read_csv(path_trials, sep=config.sep)
        tdf_ndf = tdf.copy()
    else:
        tdf = pd.read_csv(res_mean, sep=config.sep)
        tdf_ndf = pd.read_csv(res_ndf, sep=config.sep)
    subcols = ["mean", "std", "n_df"]
    # pd.DataFrame(None, tdf.index, pd.MultiIndex.from_product([get_camera_pos_keys(), ["mean", "std", "n_df"]]))
    tdf_b = tdf[int(BLOCK[-1]) == tdf["block"]]
    for fk in get_camera_pos_keys():
        cam, pos = fk.split("_")
        for d in get_days_in_order(camera=cam, is_back=config.BACK == pos):
            keys, df_days = csv_of_the_day(
                camera=cam, day=d, is_back=config.BACK == pos
            )
            daystr, daytime = d.split("_")
            daytime_DF = (
                start_time_of_day_to_seconds(daytime) * config.FRAMES_PER_SECOND
            )
            for k, df in zip(keys, df_days):
                df.index = df.FRAME + (int(k) * config.BATCH_SIZE) + daytime_DF
            if len(keys) > 0:
                df_d = pd.concat(df_days)
                date = ".".join([daystr[6:8], daystr[4:6], daystr[2:4]])
                start, end = tdf_b[tdf_b["date"] == date][
                    ["trial_start", "trial_end"]
                ].to_numpy()[0]
                s, e = (
                    get_seconds_from_time(start) * config.FRAMES_PER_SECOND,
                    get_seconds_from_time(end) * config.FRAMES_PER_SECOND,
                )
                trial_df = df_d[(df_d.index >= s) & (df_d.index <= e)]
                data = trial_df[["xpx", "ypx"]].to_numpy()
                err_filter = error_default_points(data)
                act = activity(
                    pixel_to_cm(data, fish_key=fk), data.shape[0], err_filter
                )
                tdf.loc[tdf["date"] == date, fk] = act[0][0]
                tdf_ndf.loc[tdf["date"] == date, fk] = act[0][2]

    tdf.to_csv(res_mean, sep=config.sep, index=False)
    tdf_ndf.to_csv(res_ndf, sep=config.sep, index=False)
