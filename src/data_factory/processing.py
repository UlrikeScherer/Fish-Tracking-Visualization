import os, glob
import time
import pandas as pd
import multiprocessing as mp

from src.utils.excluded_days import get_excluded_days, block1_remove
from .utils import get_days, set_parameters
from src.methods import turning_directions, distance_to_wall_chunk, calc_steps
from src.utils.tank_area_config import get_area_functions
from src.utils.error_filter import all_error_filters
from src.utils.transformation import px2cm, normalize_origin_of_compartment
from src.metrics.metrics import update_filter_three_points
from src.config import BACK, BATCH_SIZE, BLOCK, FRAMES_PER_SECOND, HOURS_PER_DAY
import numpy as np
import hdf5storage
from src.utils import csv_of_the_day
from src.utils.utile import get_camera_pos_keys, get_days_in_order, start_time_of_day_to_seconds

WAVELET = 'wavelet'
clusterStr = 'clusters'

def transform_to_traces_high_dim(data,frame_idx, filter_index, area_tuple):
    L = data.shape[0]
    fk, area = area_tuple
    data, new_area = normalize_origin_of_compartment(data, area, BACK in fk)
    steps = px2cm(calc_steps(data))
    #xy_steps = px2cm(data[:-1]-data[1:])
    t_a = turning_directions(data)
    wall = px2cm(distance_to_wall_chunk(data, new_area))
    f3 = update_filter_three_points(steps, filter_index)
    X = np.array((frame_idx[1:-1],steps[:-1], t_a, wall[1:-1], data[1:-1,0], data[1:-1,1])).T 
    return X[~f3], new_area

def compute_projections(fish_key, day, area_tuple, excluded_days=dict()):
    cam, pos = fish_key.split("_")
    is_back = pos==BACK
    keys,data_in_batches = csv_of_the_day(cam,day,is_back=is_back, 
            batch_keys_remove=excluded_days.get(f"{BLOCK}_{fish_key}_{day}",[])
            )
    if len(data_in_batches)==0:
        print("%s for day %s is empty! "%(fish_key,day))
        return None,None
    daytime_DF = start_time_of_day_to_seconds(day.split("_")[1])*FRAMES_PER_SECOND
    for k, df in zip(keys,data_in_batches):
        df.index = df.FRAME+(int(k)*BATCH_SIZE)+daytime_DF
    data = pd.concat(data_in_batches)
    data_px = data[["xpx", "ypx"]].to_numpy()
    filter_index = all_error_filters(data_px, area_tuple, fish_key=fish_key, day=day)
    X, new_area = transform_to_traces_high_dim(data_px,data.index, filter_index, area_tuple)
    return X, new_area

def compute_and_write_projection(fk, day, area_tuple, filename, recompute=False, excluded_days=dict()):
    if not recompute and os.path.exists(filename):
        return None
    X, new_area = compute_projections(fk, day, area_tuple, excluded_days=excluded_days)
    if X is None: return None
    print(f"{fk} {day} {X.shape}")
    if X.shape[0]<1000:
        print("Skipe number of datapoints to small")
        return None
    hdf5storage.write(data={'projections': X[:,1:4], 
                            'positions': X[:,4:], 
                            'area':new_area, 
                            'df_time_index':X[:,0],
                            'day':day,
                            'fish_key':fk},
                        path='/', truncate_existing=True,
                filename=filename,
                store_python_metadata=False, matlab_compatible=True)
    return None

def compute_all_projections(projectPath, fish_keys=None, recompute=False, excluded_days=dict()):
    area_f = get_area_functions()
    if fish_keys is None:
        fish_keys = get_camera_pos_keys()
    numProcessors = mp.cpu_count()
    for i, fk in enumerate(fish_keys):
        t1 = time.time()
        pool = mp.Pool(numProcessors)
        days = get_days_in_order(camera=fk.split("_")[0], is_back=fk.split("_")[1]==BACK)
        outs = pool.starmap(compute_and_write_projection, 
                [(fk, day, (fk, area_f(fk,day)), projectPath + f'/Projections/{BLOCK}_{fk}_{day}_pcaModes.mat', recompute, excluded_days) for day in days])
        pool.close()
        pool.join()
        print('\t Processed fish #%4i %s out of %4i in %0.02fseconds.\n'%(i+1, fk, len(fish_keys), time.time()-t1))

def load_trajectory_data(parameters,fk="", day=""):
    data_by_day = []
    pfile = glob.glob(parameters.projectPath+f'/Projections/{fk}*_{day}*_pcaModes.mat')
    pfile.sort()
    for f in pfile: 
        data = hdf5storage.loadmat(f)
        data_by_day.append(data)
    return data_by_day

def load_trajectory_data_concat(parameters,fk="", day=""):
    data_by_day = load_trajectory_data(parameters,fk, day)
    if len(data_by_day)==0:
        return None
    daily_df = 5*(60**2)*8 #get_num_tracked_frames_per_day()
    positions = np.concatenate([trj["positions"] for trj in data_by_day])
    projections = np.concatenate([trj["projections"] for trj in data_by_day])
    days = sorted(list(set(map(lambda d: d.split("_")[0], get_days(parameters, prefix=fk.split("_")[0])))))
    df_time_index = np.concatenate([trj["df_time_index"].flatten()+(daily_df*days.index(trj["day"].flatten()[0].split("_")[0])) for trj in data_by_day]) # 6h plus in the beginning ? TODO
    area = data_by_day[0]["area"]
    return dict(positions=positions, projections=projections, df_time_index=df_time_index, area=area)

def load_zVals(parameters,fk="", day=""):
    data_by_day = []
    zValstr = get_zValues_str(parameters)
    pfile = glob.glob(parameters.projectPath+f'/Projections/{fk}*_{day}*_pcaModes_{zValstr}.mat')
    pfile.sort()
    for f in pfile: 
        data = hdf5storage.loadmat(f)
        data_by_day.append(data)
    return data_by_day

def load_zVals_concat(parameters,fk="", day=""):
    zVals_by_day = load_zVals(parameters,fk, day)
    embeddings = None
    if len(zVals_by_day)==0:
        return dict(embeddings=embeddings)
    embeddings = np.concatenate([x["zValues"] for x in zVals_by_day])
    return dict(embeddings=embeddings)

def load_clusters(parameters,fk="", day="", k=None):
    if k is None:
        k = parameters.kmeans
    data_by_day = []
    pfile = glob.glob(parameters.projectPath+f'/Projections/{fk}*_{day}*_pcaModes_{clusterStr}_{k}.mat')
    pfile.sort()
    for f in pfile: 
        data = hdf5storage.loadmat(f)
        data_by_day.append(data)
    return data_by_day

def load_clusters_concat(parameters,fk="", day="", k=None):
    clusters_by_day = load_clusters(parameters,fk, day, k)
    if len(clusters_by_day)==0:
        return None
    clusters = np.concatenate([x["clusters"] for x in clusters_by_day], axis=1).flatten()
    return clusters

def get_zValues_str(parameters):
    if parameters.method == 'TSNE':
        zValstr = 'zVals'
    else:
        zValstr = 'uVals'
    return zValstr

def get_regions_for_fish_key(wshedfile, fish_key="", day=""):
    index_fk = np.where([(f"{fish_key}_{day}" in file.flatten()[0]) for file in wshedfile['zValNames'][0]])[0]
    if len(index_fk) == 0:
        print(fish_key, day, " no corresponding regions found!")
        return None
    if index_fk[0]==0:
        idx_p = [0,wshedfile['zValLens'].flatten().cumsum()[index_fk[-1]]]
    else:
        idx_p = wshedfile['zValLens'].flatten().cumsum()[[index_fk[0]-1, index_fk[-1]]]
    return wshedfile['watershedRegions'].flatten()[idx_p[0]:idx_p[1]]

def get_fish_info_from_wshed_idx(wshedfile, idx_s, idx_e):
    lens = wshedfile['zValLens'].flatten()
    cumsum_lens = lens.cumsum()
    hit_idx = np.where(cumsum_lens>idx_s)[0][0]
    if not idx_e < cumsum_lens[hit_idx]:
        raise ValueError("%s,%s < %s are not from the same day"%(idx_s, idx_e, cumsum_lens[hit_idx]))
    file_name = wshedfile['zValNames'].flatten()[hit_idx].flatten()[0].split("_")
    return "_".join(file_name[1:3]), file_name[3], idx_s-cumsum_lens[hit_idx]+lens[hit_idx], idx_e-cumsum_lens[hit_idx]+lens[hit_idx]
  
def load_summerized_data(wshedfile, parameters, fish_key="", day=""):
    data_dict = load_trajectory_data_concat(parameters, fish_key, day=day)
    clusters = get_regions_for_fish_key(wshedfile, fish_key, day=day)
    zVals_dict = load_zVals_concat(parameters, fish_key, day=day)
    data_dict.update(dict(clusters=clusters))
    data_dict.update(zVals_dict)
    return data_dict

def return_normalization_func(parameters):
    data = np.concatenate([d["projections"] for d in load_trajectory_data(parameters)])
    #maxi = np.abs(data).max(axis=0)
    std = data.std(axis=0)
    return lambda pro: pro/std

def get_num_tracked_frames_per_day():
    return HOURS_PER_DAY * (60**2) * FRAMES_PER_SECOND

def rename_clusters(clusters, rating_feature):
    n_clusters = clusters.max()
    avg_feature = np.zeros(n_clusters)
    for i in range(n_clusters):
        avg_feature[i] = np.mean(rating_feature[clusters == i])
    index_map = np.argsort(avg_feature)
    renamed_clusters = np.empty(clusters.shape, dtype=int)
    for i, j in enumerate(index_map):
        renamed_clusters[clusters == j] = i
    return renamed_clusters


if __name__ == "__main__":
    print("Start computation for: ", BLOCK)
    parameters = set_parameters()
    fish_keys = get_camera_pos_keys() # get all fish keys
    for key in block1_remove:
        if BLOCK in key:
            fish_keys.remove("_".join(key.split("_")[1:]))
    excluded=get_excluded_days(list(map(lambda f: f"{BLOCK}_{f}", fish_keys)))
    compute_all_projections(parameters.projectPath,fish_keys,excluded_days=excluded,recompute=True)