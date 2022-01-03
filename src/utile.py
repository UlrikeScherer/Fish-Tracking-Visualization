from datetime import datetime
import pandas as pd
import dask
import dask.dataframe as dd
import numpy as np
import os
import re
import glob

MEAN_GLOBAL = 0.22746102241709162
SD_GLOBAL = 1.0044248513034164
S_LIMIT = MEAN_GLOBAL + 3 * SD_GLOBAL
dir_front = "../FE_block1_autotracks_front"
dir_back = "../FE_block1_autotracks_back"

def get_camera_names():
    return [name for name in os.listdir(dir_front) if len(name)==8 and name.isnumeric()]

def get_days_in_order():
    cameras = get_camera_names()
    days = [name[:13] for name in os.listdir(dir_front+"/"+cameras[0]) if name[:8].isnumeric()]
    days.sort()
    return days

def get_time_for_day(day, nrF):
    dateiso = "{}-{}-{}T{}:{}:{}+02:00".format(day[:4],day[4:6],day[6:8],day[9:11],day[11:13],day[13:15])
    ts = datetime.fromisoformat(dateiso).timestamp() + nrF/5
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")

def get_date(day):
    return day[:8]

def get_full_date(day):
    dateiso = "{}-{}-{}T{}:{}:{}+00:00".format(day[:4],day[4:6],day[6:8],day[9:11],day[11:13], day[13:15])
    return datetime.fromisoformat(dateiso).strftime("%A, %B %d, %Y %H:%M")
    
def get_position_string(is_back):
    if is_back:
        return "BACK"
    else:
        return "FRONT"
    
def read_batch_csv(filename, drop_errors):
    df = pd.read_csv(filename,skiprows=3, delimiter=';', error_bad_lines=False, usecols=["x", "y", "FRAME", "time", "xpx", "ypx"])
    df.dropna(axis="rows", how="any", inplace=True)
    if drop_errors:
        indexNames = df[:-1][ df[:-1].x.array <= -1].index # exept the last index for time recording
        df = df.drop(index=indexNames)
    df.reset_index(drop=True, inplace=True)
    return df

def merge_files(filenames, drop_errors):
    batches = []
    filenames.sort()
    for f in filenames:
        df = read_batch_csv(f, drop_errors)
        batches.append(df)
        #print(get_time_for_day(df.time[0]), get_time_for_day(df.time[len(df.time)-1]))
    return batches


def csv_of_the_day(camera, day, is_back=False, drop_out_of_scope=False):
    """
    @params: camera, day, is_back, drop_out_of_scope
    returns csv of the day for camera: front or back
    """
    dir_ = dir_front 
    if is_back: 
        dir_ = dir_back
    filenames_f = [f for f in glob.glob("{}/{}/{}*/{}_{}*.csv".format(dir_, camera, day, camera, day), recursive=True) if re.search(r'[0-9].*\.csv$', f[-6:])]
    #filenames_b = glob.glob("{}/{}/{}*/*.csv".format(dir_back, camera, day), recursive=True)
    return merge_files(filenames_f, drop_out_of_scope)#, merge_files(filenames_b, drop_out_of_scope))
            