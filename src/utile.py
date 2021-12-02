from datetime import datetime
import pandas as pd
import os
import glob

dir_front = "../FE_block1_autotracks_front"
dir_back = "../FE_block1_autotracks_back"

def get_camera_names():
    return [name for name in os.listdir(dir_front) if len(name)==8 and name.isnumeric()]

def get_days_in_order():
    cameras = get_camera_names()
    days = [name[:8] for name in os.listdir(dir_front+"/"+cameras[0]) if name[:8].isnumeric()]
    m_days = map(lambda d: read_batch_csv(glob.glob("{}/{}/{}*/*.csv".format(dir_front, cameras[0], d), recursive=True)[0]).time[0],days)
    days_map = list(zip(days, m_days))
    days_map.sort(key=lambda x: x[1])
    return days_map

def get_time_for_day(t_stamp):
    return datetime.fromtimestamp(t_stamp/1000.0).strftime("%H:%M:%S")

def get_date_from_time(t_stamp):
    return datetime.fromtimestamp(t_stamp/1000.0).strftime("%Y%m%d")

def get_full_date(t_stamp):
    return datetime.fromtimestamp(t_stamp/1000.0).strftime("%A, %B %d, %Y %H:%M")

def get_position_string(is_back):
    if is_back:
        return "back"
    else:
        return "front"
    
def read_batch_csv(filename):
    df = pd.read_csv(filename,skiprows=3, delimiter=';', error_bad_lines=False, usecols=["x", "y", "time"])
    df.dropna(axis=0, how="any", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def merge_files(filenames):
    batches = []
    filenames.sort()
    for f in filenames:
        df = read_batch_csv(f)
        batches.append(df)
        #print(get_time_for_day(df.time[0]), get_time_for_day(df.time[len(df.time)-1]))
    return batches

"""
@params: camera, day
returns (front, back) csv of the day 
"""
def csv_of_the_day(camera, day):
    data_f = []
    data_b = []
    filenames_f = glob.glob("{}/{}/{}*/*.csv".format(dir_front, camera, day), recursive=True)
    filenames_b = glob.glob("{}/{}/{}*/*.csv".format(dir_back, camera, day), recursive=True)
    return (merge_files(filenames_f), merge_files(filenames_b))
            