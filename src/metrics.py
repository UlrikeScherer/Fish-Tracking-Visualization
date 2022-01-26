import numpy as np
from src.utile import S_LIMIT, BACK, fish2camera, get_days_in_order, csv_of_the_day, BLOCK, N_SECONDS_OF_DAY, get_seconds_from_day, N_FISHES
from methods import activity, calc_steps, turning_angle
import pandas as pd
import os

DATA_results = "results"

def mean_sd(steps):
    mean = np.mean(steps)
    sd = np.std(steps)
    return mean, sd

def num_of_spikes(steps):
    return np.sum(steps > S_LIMIT)

def calc_length_of_steps(df):
    ysq = (df.y.array[1:] - df.y.array[:-1])**2
    xsq = (df.x.array[1:] - df.x.array[:-1])**2
    c=np.sqrt(ysq + xsq)
    return c

def calc_step_per_frame(df):
    """ This function calculates the eucleadian step length in centimerters per FRAME, this is useful as a speed measument after the removal of erroneous data points."""
    ysq = (df.y.array[1:] - df.y.array[:-1])**2
    xsq = (df.x.array[1:] - df.x.array[:-1])**2
    frame_dist = df.FRAME.array[1:] - df.FRAME.array[:-1]
    c=np.sqrt(ysq + xsq)/frame_dist
    return c

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def determinant(v,w):
    """ Determinant of two vectors. """
    return v[0]*w[1]-v[1]*w[0]

def direction_angle(v,w):
    """ Return the angle between v,w anti clockwise from direction v to m. """
    cos = np.dot(v,w)
    r = np.arccos(np.clip(cos, -1, 1))
    det = determinant(v,w)
    if det < 0: 
        return r
    else:
        return -r
    
def angle(v,w):
    cos = np.dot(v,w)
    return np.arccos(np.clip(cos, -1, 1))

def sum_of_angles(df):
    y = (df.ypx.array[1:] - df.ypx.array[:-1])
    x = (df.xpx.array[1:] - df.xpx.array[:-1])
    if len(x) == 0:
        return 0
    u = unit_vector([x[0],y[0]])
    sum_alpha = 0
    for i in range(1,y.size):
        v = unit_vector([x[i],y[i]])
        if np.any(np.isnan(v)):
            continue
        if np.any(np.isnan(u)):
            u = v
            continue
        alpha = direction_angle(u,v)
        sum_alpha += alpha
        u = v
    return sum_alpha

def metric_per_interval(fish_ids=[i for i in range(N_FISHES)], time_interval=100, day_interval = [0, 29], metric=activity, write_to_csv=False):
    """Aplies a given function to all fishes in fish_ids with the time_interval, for all days in the day_interval interval"""
    if isinstance(fish_ids, int):
        fish_ids = [fish_ids]
    days = get_days_in_order(interval=day_interval)
    results = list()
    for i,fish in enumerate(fish_ids):
        camera_id, is_back = fish2camera[fish,0], fish2camera[fish,1]==BACK
        day_list = list()
        for j,day in enumerate(days):
            df_day = csv_of_the_day(camera_id, day, is_back=is_back, drop_out_of_scope=True) ## True or False testing needed
            if len(df_day)>0:
                df = pd.concat(df_day)
                result = metric(df[["x", "y"]].to_numpy(),time_interval*5)
                day_list.append(result)
            else: day_list.append(np.empty([0, 2]))
        results.append(day_list)
    if write_to_csv:
        metric_data_to_csv(results, time_interval=time_interval, fish_ids=fish_ids, day_interval=day_interval, metric=metric)
    return results

def metric_data_to_csv(results, time_interval=100, fish_ids=[0], day_interval=[0,1], metric=activity):
    days = get_days_in_order(interval=day_interval)
    for i,fish in enumerate(fish_ids):
        time = list()
        for j,day in enumerate(days):
            time.extend([(day,t*time_interval+j*N_SECONDS_OF_DAY+get_seconds_from_day(day)) for t in range(1,results[i][j].shape[0]+1)])
        time_np = np.array(time)
        concat_r = np.concatenate(results[i])
        print(time_np.shape, concat_r.shape)
        data = np.concatenate((time_np, concat_r), axis=1)
        
        df = pd.DataFrame(data, columns=["day", "time", "mean", "std"])
        directory="%s/%s/%s/"%(DATA_results, BLOCK,metric.__name__,)
        os.makedirs(directory, exist_ok=True)
        df.to_csv("%s/%s_%s.csv"%(directory,time_interval,"_".join(fish2camera[fish])))

def activity_per_interval(*args, **kwargs):
    return metric_per_interval(*args, **kwargs, metric=activity)

def turning_angle_per_interval(*args, **kwargs):
    return metric_per_interval(*args, **kwargs, metric=turning_angle)

