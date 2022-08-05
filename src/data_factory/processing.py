import os
import pandas as pd
from src.methods import turning_directions, distance_to_wall_chunk, calc_steps
from src.metrics.tank_area_config import get_area_functions
from src.utils.error_filter import all_error_filters
from src.utils.transformation import px2cm, normalize_origin_of_compartment
from src.metrics.metrics import update_filter_three_points
from src.config import BACK, BLOCK
import numpy as np
import motionmapperpy as mmpy
from wavelet import mm_findWavelets

from src.utils import csv_of_the_day
from src.utils.utile import get_camera_pos_keys, get_days_in_order

projectRoot = 'content'
projectPath = '%s/Fish_moves'%projectRoot
wavelet = 'wavelet'

def transform_to_traces_high_dim(data, filter_index, area_tuple):
    L = data.shape[0]
    fk, area = area_tuple
    data, new_area = normalize_origin_of_compartment(data, area, BACK in fk)
    steps = px2cm(calc_steps(data))
    xy_steps = px2cm(data[:-1]-data[1:])
    t_a = turning_directions(data)
    #wall = px2cm(distance_to_wall_chunk(data, new_area))
    f3 = update_filter_three_points(steps, filter_index)
    X = np.array((np.arange(1,L-1),xy_steps[:-1,0],xy_steps[:-1,1], t_a)).T 
    return X[~f3]

def compute_projections(fish_key, day, area_tuple, write_file=False):
    cam, pos = fish_key.split("_")
    is_back = pos==BACK
    file_path = f"{projectPath}/Projections/{BLOCK}_{cam}_{pos}_{day}.npy"
    if os.path.exists(file_path):
        return np.load(file_path)
    data = pd.concat(csv_of_the_day(cam,day, is_back=is_back)[1])
    data_px = data[["xpx", "ypx"]].to_numpy()
    filter_index = all_error_filters(data_px, area_tuple)
    X = transform_to_traces_high_dim(data_px, filter_index, area_tuple)
    if write_file:
        np.save(file_path,X)
    return X

def subsample_train_dataset(parameters, fish_keys=None):
    area_f = get_area_functions()
    train_list = []
    if fish_keys is None:
        fish_keys = get_camera_pos_keys()
    for fk in fish_keys:
        days = get_days_in_order(camera=fk.split("_")[0], is_back=fk.split("_")[1]==BACK)
        for day in days:
            area_tuple = (fk, area_f(fk,day))
            wave_file = f"{projectPath}/Projections/{wavelet}/{BLOCK}_{fk}_{day}_{wavelet}.npy"
            if os.path.exists(wave_file):
                wlets 
            X = compute_projections(fk, day, area_tuple, write_file=True)
            print(f"{fk} {fish_keys.index(fk)} {day} {days.index(day)} {X.shape}", flush=True)
            wlets, _ = mm_findWavelets(X[:,1:], parameters.pcaModes, parameters)
            select = np.random.choice(wlets.shape[0], parameters.training_points_of_day, replace=False)
            train_list.append((fk, day, wlets[select]))
    return train_list




