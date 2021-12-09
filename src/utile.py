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
    days = [name[:13] for name in os.listdir(dir_front+"/"+cameras[0]) if name[:8].isnumeric()]
    days.sort()
    return days

def get_time_for_day(day, nrF):
    dateiso = "{}-{}-{}T{}:{}:{}+00:00".format(day[:4],day[4:6],day[6:8],day[9:11],day[11:13],day[13:15])
    ts = datetime.fromisoformat(dateiso).timestamp() + nrF/5
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")

def get_date(day):
    return day[:8]

def get_full_date(day):
    dateiso = "{}-{}-{}T{}:{}:{}+00:00".format(day[:4],day[4:6],day[6:8],day[9:11],day[11:13], day[13:15])
    return datetime.fromisoformat(dateiso).strftime("%A, %B %d, %Y %H:%M")
    
def get_position_string(is_back):
    if is_back:
        return "back"
    else:
        return "front"
    
def read_batch_csv(filename):
    df = pd.read_csv(filename,skiprows=3, delimiter=';', error_bad_lines=False, usecols=["x", "y", "FRAME", "time"])
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
            