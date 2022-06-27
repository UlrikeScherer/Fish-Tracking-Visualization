from datetime import datetime
from time import gmtime, strftime
import pandas as pd
import numpy as np
import re
import os
import glob
from itertools import product
from src.config import (
    DATA_DIR,
    BLOCK,
    N_BATCHES,
    N_BATCHES_FEEDING,
    FRONT,
    BACK,
    FEEDINGTIME,
    STIME,
    dir_feeding_back,
    dir_feeding_front,
    dir_back,
    dir_front,
)
from path_validation import filter_files


def is_valid_dir(directory):
    if not os.path.isdir(directory):
        print(
            "TERMINATED: Please connect to external hard drive with path %s or edit path in scripts/env.sh"
            % directory
        )
        return False
    else:
        return True


def get_start_time_directory(is_feeding):
    return FEEDINGTIME if is_feeding else STIME


def get_directory(is_feeding=None, is_back=None):
    if is_feeding is None or is_back is None:
        raise Exception("define kwargs")
    if is_feeding:
        if is_back:
            return dir_feeding_back
        else:
            return dir_feeding_front
    else:
        if is_back:
            return dir_back
        else:
            return dir_front


def get_number_of_batches(is_feeding=False):
    return N_BATCHES_FEEDING if is_feeding else N_BATCHES


def get_camera_names(is_feeding=False, is_back=False):
    dir_ = get_directory(is_feeding, is_back)
    return sorted(
        [name for name in os.listdir(dir_) if len(name) == 8 and name.isnumeric()]
    )


def get_fish2camera_map(is_feeding=False):
    l_front = list(
        product(get_camera_names(is_feeding, is_back=BACK == FRONT), [FRONT])
    )
    l_back = list(product(get_camera_names(is_feeding, is_back=BACK == BACK), [BACK]))
    return np.array(l_back + l_front)


def get_camera_pos_keys(is_feeding=False):
    m = get_fish2camera_map(is_feeding=is_feeding)
    return ["%s_%s" % (c, p) for (c, p) in m]


def get_fish_ids():
    """
    Return the fish ids defined in ...livehistory.csv corresponding to the indices in fish2camera
    """
    # %ROOT
    info_df = pd.read_csv(
        "{}/DevEx_fingerprint_activity_lifehistory.csv".format(DATA_DIR), delimiter=";"
    )
    # info_df = pd.read_csv("data/DevEx_fingerprint_activity_lifehistory.csv", delimiter=";")
    info_df1 = info_df[info_df["block"] == int(BLOCK[-1])]
    info_df1[["fish_id", "camera", "block", "tank"]],
    fishIDs_order = list()
    FB_char = np.array(list(map(lambda x: str(x[-1]), info_df1["tank"])))
    fish2camera = get_fish2camera_map()
    for i, (c, p) in enumerate(fish2camera):
        f1 = info_df1["camera"] == int(c[6:8])
        f2 = FB_char == p[0].upper()
        ids = info_df1[f1 & f2]["fish_id"].array
        fishIDs_order.append(ids[0])

    return np.array(fishIDs_order)


def print_tex_table(fish_ids, filename):
    tex_dir = "tex/tables"
    os.makedirs(tex_dir, exist_ok=True)
    f = open("%s/%s.tex" % (tex_dir, filename), "w+")
    fids = get_fish_ids()
    fish2camera = get_fish2camera_map()
    for fid in fish_ids:
        camera, position = fish2camera[fid]
        f.write(
            "%d & %s & %s & %s\\\ \n"
            % (fid, camera, position, fids[fid].replace("_", "\_"))
        )
    f.close()


def verify_day_directory(name, camera):
    if name[:8].isnumeric() and name[16:24] == camera:
        return True
    elif name[:8].isnumeric():
        print(
            "WARNING: for CAMERA %s day directory name %s does not follow name-convention of date_starttime.camera_* and will be ignored"
            % (camera, name)
        )
        return False
    else:  # all other directories are ignored
        return False


def get_days_in_order(interval=None, is_feeding=False, is_back=None, camera=None):
    """
    @params
    interval tuple (int i,int j) i < j, to return only days from i to j
    is_feeding: select feeding directory if True, default False.
    camera: concider days of the cameras folder, default: first camera, that expects all cameras to have the same number of cameras.
    """
    if camera is None or is_back is None:
        raise ValueError("provid kwargs is_back and camera")
        # camera = get_camera_names(is_feeding=is_feeding, is_back=is_back)[0]
    dir_ = get_directory(is_feeding, is_back)
    days = [
        name[:15]
        for name in os.listdir(dir_ + "/" + camera)
        if verify_day_directory(name, camera)
    ]
    days_unique = sorted(list(set(days)))
    if len(days_unique) < len(days):
        print(
            "WARNING DUPLICATE DAY: CAMERA %s_%s some days are duplicated, please check the directory"
            % (camera, BACK if is_back else FRONT)
        )
    if interval:
        return days_unique[interval[0] : interval[1]]
    return days_unique


def get_all_days_of_context(is_feeding=False):
    days = list()
    for p in [BACK, FRONT]:
        is_back = p == BACK
        cameras = get_camera_names(is_feeding=is_feeding, is_back=is_back)
        for c in cameras:
            days.extend(
                [
                    d
                    for d in get_days_in_order(
                        is_feeding=is_feeding, is_back=is_back, camera=c
                    )
                    if d not in days
                ]
            )
    return sorted(days)


def get_time_for_day(day, nrF):
    # dateiso = "{}-{}-{}T{}:{}:{}+02:00".format(day[:4],day[4:6],day[6:8],day[9:11],day[11:13],day[13:15])
    hours, minutes, seconds = int(day[9:11]), int(day[11:13]), nrF / 5
    seconds = seconds + minutes * 60 + hours * 3600
    return strftime("%H:%M:%S", gmtime(seconds))


def get_seconds_from_day(day):
    """Retuns the time of the day in seconds from 0=:00 am on"""
    hours, minutes = int(day[9:11]), int(day[11:13])
    return minutes * 60 + hours * 3600


def get_date(day):
    return day[:8]


def get_date_string(day):
    return "%s/%s/%s" % (day[:4], day[4:6], day[6:8])


def get_full_date(day):
    dateiso = "{}-{}-{}T{}:{}:{}+00:00".format(
        day[:4], day[4:6], day[6:8], day[9:11], day[11:13], day[13:15]
    )
    return datetime.fromisoformat(dateiso).strftime("%A, %B %d, %Y %H:%M")


def get_position_string(is_back):
    if is_back:
        return BACK
    else:
        return FRONT


def read_batch_csv(filename, drop_errors):
    df = pd.read_csv(
        filename,
        skiprows=3,
        delimiter=";",
        error_bad_lines=False,
        usecols=["x", "y", "FRAME", "time", "xpx", "ypx"],
        dtype={"xpx": np.float64, "ypx": np.float64, "time": np.float64},
    )
    df.dropna(axis="rows", how="any", inplace=True)
    if drop_errors:
        err_filter = get_error_indices(df[:-1])
        df = df.drop(index=df[:-1][err_filter].index)
    df.reset_index(drop=True, inplace=True)
    return df


def get_error_indices(dataframe):
    """
    @params: dataframe
    returns a boolean pandas array with all indices to filter set to True
    """
    x = dataframe.xpx
    y = dataframe.ypx
    indexNames = ((x == -1) & (y == -1)) | (
        (x == 0) & (y == 0)
    )  # except the last index for time recording
    return indexNames


def merge_files(filenames, drop_errors):
    batches = []
    for f in filenames:
        df = read_batch_csv(f, drop_errors)
        batches.append(df)
    return batches


def csv_of_the_day(
    camera,
    day,
    is_back=False,
    drop_out_of_scope=False,
    is_feeding=False,
    print_log=False,
):
    """
    @params: camera, day, is_back, drop_out_of_scope
    returns csv of the day for camera: front or back
    """

    dir_ = get_directory(is_feeding=is_feeding, is_back=is_back)

    filenames_f = [
        f
        for f in glob.glob(
            "{}/{}/{}*/{}_{}*.csv".format(dir_, camera, day, camera, day),
            recursive=True,
        )
        if re.search(r"[0-9].*\.csv$", f[-6:])
    ]

    LOG, _, filtered_files = filter_files(
        camera, day, filenames_f, get_number_of_batches(is_feeding)
    )  # filters for duplicates in the batches for a day. It takes the LAST one!!!
    file_keys = list(filtered_files.keys())
    correct_files = list(filtered_files.values())
    if print_log and len(LOG) > 0:
        print("\n {}/{}/{}*: \n".format(dir_, camera, day), "\n".join(LOG))
    return file_keys, merge_files(correct_files, drop_out_of_scope)
