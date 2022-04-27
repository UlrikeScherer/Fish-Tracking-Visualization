from datetime import datetime
from time import gmtime, strftime
import pandas as pd
import numpy as np
import os
import re
import glob
import json
from itertools import product
from envbash import load_envbash
from path_validation import filter_files
load_envbash('scripts/env.sh')

# Calculated MEAN and SD for the data set filtered for erroneous frames 
MEAN_GLOBAL = 0.22746102241709162
SD_GLOBAL = 1.0044248513034164
S_LIMIT = 15 #MEAN_GLOBAL + 3 * SD_GLOBAL
BATCH_SIZE = 9999
FRAMES_PER_SECOND = 5
ROOT=os.environ["rootserver"]
ROOT_LOCAL=os.environ["root_local"]
DIR_CSV=os.environ["path_csv"] # 
DIR_CSV_LOCAL=os.environ["path_csv_local"] # 
BLOCK = os.environ["BLOCK"] # block1 or block2

# TRAJECTORY 
dir_front = os.environ["dir_front"]
dir_back = os.environ["dir_back"]
STIME = os.environ["STIME"]

# FEEDING 
dir_feeding_front = os.environ["dir_feeding_front"]
dir_feeding_back = os.environ["dir_feeding_back"]
FEEDINGTIME = os.environ["FEEDINGTIME"]

FRONT, BACK = "front", "back"
ROOT_img = "plots"
DATA_DIR = "data"

N_BATCHES=15
N_BATCHES_FEEDING=8

N_FISHES = 24
N_DAYS = 28
HOURS_PER_DAY = 8
N_SECONDS_PER_HOUR = 3600
N_SECONDS_OF_DAY = 24*N_SECONDS_PER_HOUR

def get_directory(is_feeding=False, is_back=False):
    if is_feeding:
        if is_back: return dir_feeding_back
        else: return dir_feeding_front
    else:
        if is_back:return dir_back
        else: return dir_front
    
def get_number_of_batches(is_feeding=False):
    return N_BATCHES_FEEDING if is_feeding else N_BATCHES

def get_camera_names(is_feeding=False, is_back=False):
    dir_ = get_directory(is_feeding, is_back)
    return sorted([name for name in os.listdir(dir_) if len(name)==8 and name.isnumeric()])

def get_fish2camera_map(is_feeding=False):
    l_front = list(product(get_camera_names(is_feeding, is_back=BACK==FRONT), [FRONT]))
    l_back = list(product(get_camera_names(is_feeding, is_back=BACK==BACK), [BACK]))
    return np.array([j for i in zip(l_front,l_back) for j in i])

def get_camera_pos_keys(is_feeding=False):
    m = get_fish2camera_map(is_feeding=is_feeding)
    return ["%s_%s"%(c,p) for (c,p) in m]

def get_fish_ids():
    """
    Return the fish ids defined in ...livehistory.csv corresponding to the indices in fish2camera
    """
    # %ROOT
    info_df = pd.read_csv("{}/DevEx_fingerprint_activity_lifehistory.csv".format(DATA_DIR), delimiter=";")
    #info_df = pd.read_csv("data/DevEx_fingerprint_activity_lifehistory.csv", delimiter=";")
    info_df1=info_df[info_df["block"]==int(BLOCK[-1])]
    info_df1[["fish_id", "camera", "block", "tank"]],
    fishIDs_order = list()
    FB_char = np.array(list(map(lambda x: str(x[-1]),info_df1["tank"])))
    fish2camera = get_fish2camera_map()
    for i, (c,p) in enumerate(fish2camera):
        f1 = info_df1["camera"] == int(c[-2:])
        f2 = FB_char == p[0].upper()
        ids = info_df1[f1 & f2]["fish_id"].array
        fishIDs_order.append(ids[0])
        
    return np.array(fishIDs_order)

def write_area_data_to_json(area_data):
    area_d = dict(zip(area_data.keys(), map(lambda v: v.tolist(),area_data.values())))
    with open("{}/{}_area_data.json".format(DATA_DIR,BLOCK), "w") as outfile:
        json.dump(area_d, outfile, indent=2)
        
def read_area_data_from_json():
    with open("{}/{}_area_data.json".format(DATA_DIR,BLOCK), "r") as infile:
        area_data = json.load(infile)
        for k in area_data.keys():
            area_data[k]=np.array(area_data[k])
        return area_data

def print_tex_table(fish_ids, filename):
    tex_dir = "tex/tables"
    os.makedirs(tex_dir, exist_ok=True)
    f = open("%s/%s.tex"%(tex_dir,filename), "w+")
    fids = get_fish_ids()
    fish2camera = get_fish2camera_map()
    for fid in fish_ids:
        camera, position = fish2camera[fid]
        f.write("%d & %s & %s & %s\\\ \n"%(fid, camera, position, fids[fid].replace("_","\_")))
    f.close()

def get_days_in_order(interval=None, is_feeding=False, camera=None):
    """
    @params
    interval tuple (int i,int j) i<j, to return only days from i to j
    is_feeding: select feeding directory if True, default False. 
    camera: concider days of the cameras folder, default: first camera, that expects all cameras to have the same number of cameras. 
    """
    if camera is None: camera = get_camera_names(is_feeding)[0]
    dir_ = dir_feeding_front if is_feeding else dir_front
    days = [name[:15] for name in os.listdir(dir_+"/"+camera) if name[:8].isnumeric()]
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

def get_date_string(day):
    return "%s/%s/%s"%(day[:4], day[4:6], day[6:8])

def get_full_date(day):
    dateiso = "{}-{}-{}T{}:{}:{}+00:00".format(day[:4],day[4:6],day[6:8],day[9:11],day[11:13], day[13:15])
    return datetime.fromisoformat(dateiso).strftime("%A, %B %d, %Y %H:%M")
    
def get_position_string(is_back):
    if is_back:
        return BACK
    else:
        return FRONT
    
def read_batch_csv(filename, drop_errors):
    df = pd.read_csv(filename,skiprows=3, delimiter=';', error_bad_lines=False, usecols=["x", "y", "FRAME", "time", "xpx", "ypx"], 
                     dtype={"xpx": np.float64, "ypx": np.float64, "time":np.float64})
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
    indexNames = ((x == -1) & (y == -1)) | ((x == 0) & (y == 0)) # except the last index for time recording
    return indexNames

def merge_files(filenames, drop_errors):
    batches = []
    for f in filenames:
        df = read_batch_csv(f, drop_errors)
        batches.append(df)
    return batches

def csv_of_the_day(camera, day, is_back=False, drop_out_of_scope=False, is_feeding=False, print_log=False):
    """
    @params: camera, day, is_back, drop_out_of_scope
    returns csv of the day for camera: front or back
    """
    dir_ = dir_back if is_back else dir_front
    if is_feeding:
        dir_ = dir_feeding_back if is_back else dir_feeding_front

    filenames_f = [f for f in glob.glob("{}/{}/{}*/{}_{}*.csv".format(dir_, camera, day, camera, day), recursive=True) if re.search(r'[0-9].*\.csv$', f[-6:])]
    
    LOG, _, filtered_files = filter_files(camera,day,filenames_f, get_number_of_batches(is_feeding)) # filters for duplicates in the batches for a day. It takes the LAST one!!!
    file_keys = list(filtered_files.keys())
    correct_files = list(filtered_files.values())
    if print_log and len(LOG)>0:
        print("\n {}/{}/{}*: \n".format(dir_, camera, day),"\n".join(LOG))
    return file_keys, merge_files(correct_files, drop_out_of_scope)     
    
def activity_for_day_hist(fish_id, day_idx=1):
    fish2camera=get_fish2camera_map()
    camera_id, is_back = fish2camera[fish_id,0], fish2camera[fish_id,1]=="back"
    mu_sd = list()
    all_days = get_days_in_order()
    day = all_days[day_idx]
    keys, csv_by_day = csv_of_the_day(camera_id, day, is_back=is_back, drop_out_of_scope=True)
    df = pd.concat(csv_by_day)
    c = calc_steps(df[["x", "y"]].to_numpy())
    p = plt.hist(c, bins=50,range=[0, 5], density=True, alpha=0.75)
    plt.ylim(0, 3)
    plt.xlabel('cm')
    plt.ylabel('Probability')
    plt.title('Histogram of step lengh per Frame')
    mu, om = np.mean(c), np.std(c)
    plt.text(mu, om, r'$\mu=\ $'+ "%.2f, "%mu + r'$\sigma=$'+"%.2f"%om)
    plt.show()