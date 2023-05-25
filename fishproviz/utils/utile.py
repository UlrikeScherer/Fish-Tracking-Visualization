from datetime import datetime
from time import gmtime, strftime
import pandas as pd
import numpy as np
import re
import os
from os import path, makedirs
import glob
from itertools import product
from fishproviz.config import (
    CONFIG_DATA,
    HOURS_PER_DAY,
    MAX_BATCH_IDX,
    MIN_BATCH_IDX,
    N_BATCHES,
    FRONT,
    BACK,
    N_SECONDS_PER_HOUR,
    P_FEEDING,
    P_TRAJECTORY,
    dir_back,
    dir_front,
)

def flatten_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


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
    return P_FEEDING if is_feeding else P_TRAJECTORY


def get_interval_name_from_seconds(seconds):
    if seconds == N_SECONDS_PER_HOUR:
        return "hour"
    if seconds == int(N_SECONDS_PER_HOUR * HOURS_PER_DAY):
        return "day"
    else:
        return "%s_sec" % seconds


def get_directory(is_back=None):
    if is_back is None:
        raise Exception("define kwargs is_back")
    if is_back:
        return dir_back
    else:
        return dir_front

def get_camera_names(is_back=False):
    dir_ = get_directory(is_back)
    return sorted(
        [name for name in os.listdir(dir_) if len(name) == 8 and name.isnumeric()]
    )


def get_fish2camera_map():
    l_front = list(
        product(get_camera_names(is_back=BACK == FRONT), [FRONT])
    )
    l_back = list(product(get_camera_names(is_back=BACK == BACK), [BACK]))
    return np.array(l_back + l_front)


def get_camera_pos_keys():
    m = get_fish2camera_map()
    return ["%s_%s" % (c, p) for (c, p) in m]


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


def get_days_in_order(interval=None, is_back=None, camera=None):
    """
    @params
    interval tuple (int i,int j) i < j, to return only days from i to j
    camera: concider days of the cameras folder, default: first camera, that expects all cameras to have the same number of cameras.
    """
    if camera is None or is_back is None:
        raise ValueError("provid kwargs is_back and camera")
    dir_ = get_directory(is_back)
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


def get_all_days_of_context():
    days = list()
    for p in [BACK, FRONT]:
        is_back = p == BACK
        cameras = get_camera_names(is_back=is_back)
        for c in cameras:
            days.extend(
                [
                    d
                    for d in get_days_in_order(
                        is_back=is_back, camera=c
                    )
                    if d not in days
                ]
            )
    return sorted(days)

def get_seconds_from_time(time): 
    """
    @time string hh:mm
    return seconds (int)
    """
    return sum([int(t)*f for (t, f) in zip(time.split(":"),[3600, 60])])

def start_time_of_day_to_seconds(START_TIME):
    """
    @start_time hhmmss
    return seconds (int)
    """
    if len(START_TIME) == 6:
        return int(START_TIME[:2]) * 3600 + int(START_TIME[2:4]) * 60 + int(START_TIME[4:])
    else:
        raise ValueError("START_TIME must be of length 6")

def get_time_for_day(day, nrF):
    # dateiso = "{}-{}-{}T{}:{}:{}+02:00".format(day[:4],day[4:6],day[6:8],day[9:11],day[11:13],day[13:15])
    hours, minutes, seconds = int(day[9:11]), int(day[11:13]), nrF / 5
    seconds = seconds + minutes * 60 + hours * 3600
    return strftime("%H:%M:%S", gmtime(seconds))


def get_seconds_from_day(day):
    """Retuns the time of the day in seconds from 0=:00 am on"""
    hours, minutes = int(day[9:11]), int(day[11:13])
    return minutes * 60 + hours * 3600

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
        usecols=["x", "y", "FRAME", "time", "xpx", "ypx"],
        dtype={"xpx": np.float64, "ypx": np.float64, "time": np.float64},
    )
    if drop_errors:
        df.dropna(axis="rows", how="any", inplace=True)
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
    batch_keys_remove=[],
    print_logs=False,
):
    """
    @params: camera, day, is_back, drop_out_of_scope
    returns csv of the day for camera: front or back
    """

    dir_ = get_directory(is_back=is_back)

    filenames_f = [
        f
        for f in glob.glob(
            "{}/{}/{}*/{}_{}*.csv".format(dir_, camera, day, camera, day),
            recursive=True,
        )
        if re.search(r"[0-9].*\.csv$", f[-6:])
    ]
    filenames_f = sorted(filenames_f)
    LOG, _, filtered_files = filter_files(
        camera, day, filenames_f, n_files=MAX_BATCH_IDX + 1, min_idx=MIN_BATCH_IDX
    )  # filters for duplicates in the batches for a day. It takes the LAST one!!!
    for key in batch_keys_remove:
        filtered_files.pop(key, None)
    file_keys = list(filtered_files.keys())
    correct_files = list(filtered_files.values())
    if print_logs and len(LOG) > 0:
        print("\n {}/{}/{}*: \n".format(dir_, camera, day), "\n".join(LOG))
    return file_keys, merge_files(correct_files, drop_out_of_scope)


def filter_files(c, d, files, n_files=15, min_idx=0, Logger=None):
    """
    @params:
    c: camera_id
    d: folder name of a day
    files: list of files that are to be filtered
    n_files: number of files to expect.
    logger: Logger defined in path_validation
    @Returns: LOG, duplicate_f, correct_f
    msg_counter: number of debug-messages
    duplicate_f: a list of all duplicates occurring
    correct_f: dict of the correct files for keys i in 0,...,n_files-1
    """
    msg_counter = 0
    missing_numbers = []
    duplicate_f = []
    correct_f = dict()
    for i in range(min_idx,n_files):
        key_i = "{:06d}".format(i)
        pattern = re.compile(
            ".*{}_{}.{}_{}_\d*-\d*-\d*T\d*_\d*_\d*_\d*.csv".format(c, d[:15], c, key_i)
        )
        i_f = [f for f in files if pattern.match(f) is not None]

        if len(i_f) > 1:
            i_f.sort()
            duplicate_f.extend(i_f[:-1])
            correct_f[key_i] = i_f[-1]
        elif len(i_f) == 0:
            missing_numbers.append(key_i)
        else:
            correct_f[key_i] = i_f[-1]

    pattern_general = re.compile(
        ".*{}_{}.{}_\d*_\d*-\d*-\d*T\d*_\d*_\d*_\d*.csv".format(c, d[:15], c)
    )
    corrupted_f = [f for f in files if pattern_general.match(f) is None]

    if len(missing_numbers) > 0:
        msg_counter += 1
        Logger.debug(
            "The following files are missing: \n \t\t\t\t{}".format(
                " ".join(missing_numbers)
            )
        )
    if len(duplicate_f) > 0:
        msg_counter += 1
        Logger.debug(
            "The following files are duplicates: \n\t\t\t\t{}".format(
                "\n\t".join(duplicate_f)
            )
        )
    if len(corrupted_f) > 0:
        msg_counter += 1
        Logger.debug(
            "The following file names are corrupted, maybe wrong folder: \n\t\t\t\t{}".format(
                "\n\t".join(corrupted_f)
            )
        )
    return msg_counter, duplicate_f, correct_f


def get_timestamp(
    format = "%d-%m-%Y_%H:%M:%S"
):
    current_time = datetime.now()
    timestamp = current_time.timestamp()
    date_time = datetime.fromtimestamp(timestamp)
    str_date_time = date_time.strftime(format)
    return str_date_time


def create_directory(
    directory_name: str
): 
    dir_path = path.join(os.getcwd(), directory_name)
    makedirs(dir_path, exist_ok=True)
    return dir_path