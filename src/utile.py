from datetime import datetime
from time import gmtime, strftime
import pandas as pd
import numpy as np
import os
import re
import glob
from itertools import product
from envbash import load_envbash
load_envbash('tex/env.sh')

# Calculated MEAN and SD for the data set filtered for erroneous frames 
MEAN_GLOBAL = 0.22746102241709162
SD_GLOBAL = 1.0044248513034164
S_LIMIT = MEAN_GLOBAL + 3 * SD_GLOBAL
BATCH_SIZE = 9999
ROOT=os.environ["rootserver"]
DIR_CSV=os.environ["path_csv"] # BLOCK
BLOCK = os.environ["BLOCK"] # 
DIR_CSV_LOCAL = os.environ["path_csv_local"] # 
dir_front = "%s/FE_%s_autotracks_front"%(DIR_CSV_LOCAL, BLOCK)
dir_back  = "%s/FE_%s_autotracks_back"%(DIR_CSV_LOCAL, BLOCK)
FRONT, BACK = "front", "back"

N_FISHES = 24
N_SECONDS_OF_DAY = 24*3600

def get_camera_names():
    return [name for name in os.listdir(dir_front) if len(name)==8 and name.isnumeric()]

fish2camera=np.array(list(product(get_camera_names(), [FRONT, BACK])))

def get_days_in_order(interval=None):
    cameras = get_camera_names()
    days = [name[:13] for name in os.listdir(dir_front+"/"+cameras[0]) if name[:8].isnumeric()]
    days.sort()
    if interval:
        return days[interval[0]: interval[1]]
    return days

def get_time_for_day(day, nrF):
    # dateiso = "{}-{}-{}T{}:{}:{}+02:00".format(day[:4],day[4:6],day[6:8],day[9:11],day[11:13],day[13:15])
    hours, minutes, seconds = int(day[9:11]),int(day[11:13]), nrF/5
    seconds = seconds + minutes*60 + hours * 3600
    return strftime("%H:%M:%S", gmtime(seconds))

def get_seconds_from_day(day):
    """Retuns the time of the day in seconds from 0=:00 am on"""
    hours, minutes = int(day[9:11]),int(day[11:13])
    return minutes*60 + hours * 3600

def get_date(day):
    return day[:8]

def get_full_date(day):
    dateiso = "{}-{}-{}T{}:{}:{}+00:00".format(day[:4],day[4:6],day[6:8],day[9:11],day[11:13], day[13:15])
    return datetime.fromisoformat(dateiso).strftime("%A, %B %d, %Y %H:%M")
    
def get_position_string(is_back):
    if is_back:
        return BACK
    else:
        return FRONT
    
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
    return merge_files(filenames_f, drop_out_of_scope)
            