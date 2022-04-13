import numpy as np
from scipy.stats import entropy
from src.utile import S_LIMIT, BACK, get_days_in_order, csv_of_the_day, BLOCK, N_SECONDS_OF_DAY, get_seconds_from_day, N_FISHES, get_fish2camera_map
from src.transformation import pixel_to_cm
from methods import activity, calc_steps, turning_angle, tortuosity_of_chunk #cython
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

def calc_step_per_frame(batchxy, frames):
    """ This function calculates the eucleadian step length in centimeters per FRAME, this is useful as a speed measument after the removal of erroneous data points."""
    xsq = (batchxy[1:,0]-batchxy[:-1,0])**2
    ysq = (batchxy[1:,1]-batchxy[:-1,1])**2
    frame_dist = frames[1:] - frames[:-1]
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

def entropy_for_chunk(chunk):
    hist = np.histogram2d(chunk[:,0],chunk[:,1], bins=(40, 20), density=True)[0]
    prob = list()
    l_x,l_y = hist.shape
    indi_1 = np.tril_indices(l_y,k=1)
    indi_1 = indi_1[0],  (l_y-1) - indi_1[1]
    indi_2 = np.triu_indices(l_y, k=-2)
    indi_2 = indi_2[0]+l_y, indi_2[1] 
    prob.extend(hist[indi_1])
    prob.extend(hist[indi_2])
    return entropy(prob),np.std(prob)*100

def entropy_for_data(data, frame_interval):
    SIZE = data.shape[0]
    len_out = int(np.ceil(SIZE/frame_interval))
    mu_sd = np.zeros([len_out,2], dtype=float)
    for i,s in enumerate(range(frame_interval, data.shape[0], frame_interval)):
        chunk = data[s-frame_interval:s]
        chunk = chunk[chunk[:,0] > -1] # only consider valid points. 
        result = entropy_for_chunk(chunk)
        mu_sd[i, 0] = result[0]
        mu_sd[i, 1] = result[1]
    return mu_sd

def average_by_metric(data, frame_interval, metric_f):
    SIZE = data.shape[0]
    len_out = int(np.ceil(SIZE/frame_interval))
    mu_sd = np.zeros([len_out,2], dtype=float)
    for i,s in enumerate(range(frame_interval, data.shape[0], frame_interval)):
        chunk = data[s-frame_interval:s]
        chunk = chunk[chunk[:,0] > -1] # only consider valid points. 
        avg_metric = metric_f(chunk)
        result_size = avg_metric.size
        mu_sd[i, 0] = sum(avg_metric)/result_size
        mu_sd[i, 1] = np.sqrt(sum((avg_metric-mu_sd[i, 0])**2)/result_size)
    return mu_sd

def tortuosity(data, frame_interval):
    return average_by_metric(data, frame_interval, tortuosity_of_chunk)

def metric_per_interval(fish_ids=[i for i in range(N_FISHES)], time_interval=100, day_interval = (0, 29), metric=activity, write_to_csv=False):
    """
    Applies a given function to all fishes in fish_ids with the time_interval, for all days in the day_interval interval
    Args:
        fish_ids(list, int):    List of fish ids
        time_interval(int):     Time Interval to apply the metric to
        day_interval(Tuple):    Tuple of the first day to the last day, out of 0 to 29. 
        metric(function):       A function to apply to the data, {activity, tortuosity, turning_angle,...}
        write_to_csv(bool):     Indicate weather the results should be written to a csv
    Returns: 
        results(list):          List of computed results
    """
    if isinstance(fish_ids, int):
        fish_ids = [fish_ids]
    days = get_days_in_order(interval=day_interval)
    fish2camera = get_fish2camera_map()
    results = list()
    for i,fish in enumerate(fish_ids):
        camera_id, is_back = fish2camera[fish,0], fish2camera[fish,1]==BACK
        day_list = list()
        for j,day in enumerate(days):
            df_day = csv_of_the_day(camera_id, day, is_back=is_back, drop_out_of_scope=True) ## True or False testing needed
            if len(df_day)>0:
                df = pd.concat(df_day)
                result = metric(pixel_to_cm(df[["xpx", "ypx"]].to_numpy()),time_interval*5)
                day_list.append(result)
            else: day_list.append(np.empty([0, 2]))
        results.append(day_list)
    if write_to_csv:
        metric_data_to_csv(results, time_interval=time_interval, fish_ids=fish_ids, day_interval=day_interval, metric=metric)
    return results

def metric_data_to_csv(results, time_interval=100, fish_ids=[0], day_interval=[0,1], metric=activity):
    days = get_days_in_order(interval=day_interval)
    fish2camera = get_fish2camera_map()
    for i,fish in enumerate(fish_ids):
        time = list()
        for j,day in enumerate(days):
            time.extend([(day,t*time_interval+j*N_SECONDS_OF_DAY+get_seconds_from_day(day)) for t in range(1,results[i][j].shape[0]+1)])
        time_np = np.array(time)
        concat_r = np.concatenate(results[i])
        data = np.concatenate((time_np, concat_r), axis=1)
        
        df = pd.DataFrame(data, columns=["day", "time", "mean", "std"])
        directory="%s/%s/%s/"%(DATA_results, BLOCK,metric.__name__)
        os.makedirs(directory, exist_ok=True)
        df.to_csv("%s/%s_%s.csv"%(directory,time_interval,"_".join(fish2camera[fish])))

def activity_per_interval(*args, **kwargs):
    return metric_per_interval(*args, **kwargs, metric=activity)

def turning_angle_per_interval(*args, **kwargs):
    return metric_per_interval(*args, **kwargs, metric=turning_angle)

def tortuosity_per_interval(*args, **kwargs):
    return metric_per_interval(*args, **kwargs, metric=tortuosity)

def entropy_per_interval(*args, **kwargs): 
    return metric_per_interval(*args, **kwargs, metric=entropy_for_data)
