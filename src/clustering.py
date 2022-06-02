import os
from src.config import BATCH_SIZE, BLOCK, STIME, BACK, sep
from src.error_filter import all_error_filters, error_default_points
from src.transformation import rotation, pixel_to_cm
from src.metrics import entropy_for_data, distance_to_wall
from src.utile import  get_fish2camera_map, csv_of_the_day
from src.tank_area_config import get_area_functions
from methods import activity, turning_angle, absolute_angles
from itertools import product
import pandas as pd
from scipy.spatial import ConvexHull
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

MU_STR, SD_STR = "mu", "sd"
DIR_TRACES = "%s/%s/%s"%("results",BLOCK,"traces")

def get_traces_type():
    _, names = get_metrics_for_traces()
    traces_type = np.dtype({
        'names': ["CAMERA_POSITION","DAY", "BATCH", "DATAFRAME"]+names,
        'formats': ["str"]*4 + ["f4"]*len(names)
    })
    return traces_type

def get_trace_file_path(trace_size, format="csv"):
    return "%s/traces_%s_%s_%s.%s"%(DIR_TRACES, BLOCK, STIME, trace_size, format)
    
def load_traces(trace_size):
    trace_path = get_trace_file_path(trace_size)
    trace_path_npy = get_trace_file_path(trace_size, "npy")
    if not os.path.exists(trace_path) or not os.path.exists(trace_path_npy):
        raise Exception("Trace for path %s does not exist"%trace_path)
    traces = pd.read_csv(trace_path, delimiter=sep, index_col=0)
    with open(trace_path_npy,"rb") as f:
        samples = np.load(f)
    return traces, samples

def get_trace_filter(traces):
    return traces.isna().any(axis=1)

def calculate_traces(fish_indices, days, trace_size, write_to_file=False):
    fish2cams = get_fish2camera_map()
    Xs, nSs = list(), list()
    area_func = get_area_functions()
    
    for i in fish_indices:
        cam, is_back = fish2cams[i][0], fish2cams[i][1]==BACK
        fish_key = "%s_%s"%(cam, fish2cams[i,1])
        for d in days:
            batch_keys, batches = csv_of_the_day(cam, d, is_back=is_back)

            if len(batches) == 0: continue
            b = pd.concat(batches)
            fit_len = fit_data_to_trace_size(len(b), trace_size) 
            data = b[["xpx", "ypx"]].to_numpy()[:fit_len]
            area_tuple = (fish_key,area_func(fish_key, day=d))
            # filter for errorouse data points 
            filter_index = all_error_filters(data, area_tuple)[:fit_len]

            X=transform_to_traces_metric_based(data, trace_size, filter_index, area_tuple)
            X_df = table_factory(fish_key, d, batch_keys, X, trace_size)
            Xs.append(X_df)
            nSs.append(trajectory_snippets(data, trace_size))
    traces = pd.concat(Xs)
    tfilter = traces.isna().any(axis=1)
    traces=traces[~tfilter]
    traces.reset_index(drop=True, inplace=True)
    #traces = normalize_data_metrics(traces)
    nSs = np.concatenate(nSs)[~tfilter]
    if write_to_file:
        os.makedirs(DIR_TRACES, exist_ok=True)
        traces.to_csv(get_trace_file_path(trace_size), sep=sep)
        with open(get_trace_file_path(trace_size, format="npy"), 'wb') as f:
            np.save(f, nSs)
    return traces, nSs

def traces_as_numpy(traces):
    return traces.to_numpy()[:,4:].astype(float)

def get_traces_columns():
    _, names = get_metrics_for_traces()

    return ["CAMERA_POSITION","DAY", "BATCH", "DATAFRAME"]+names

def table_factory(key_c_p, day, batch_keys, traces_of_day, trace_size):
    col = get_traces_columns()
    traces_df = pd.DataFrame(columns=col, index=range(traces_of_day.shape[0]))
    traces_df.loc[:,col[4:]] = traces_of_day

    traces_df.loc[:, col[0:2]] = key_c_p, day
    
    dataframe_pointer = np.array(range(traces_of_day.shape[0]))*trace_size
    traces_df.loc[:, col[3]] = dataframe_pointer
    for i, b_key in enumerate(batch_keys):
        mask = (dataframe_pointer >= (BATCH_SIZE * i)) & (dataframe_pointer < (BATCH_SIZE * (i+1)))
        traces_df.loc[mask, col[2]] = b_key
    return traces_df

def fit_data_to_trace_size(size1, trace_size):
    n_snippets = size1 // trace_size
    length = n_snippets * trace_size
    return length

def trajectory_snippets(data, trace_size):
    size1, size2 = data.shape
    n_snippets = size1 // trace_size
    return np.reshape(data,(n_snippets,trace_size, size2))

def rotate_trace(trace):
    alph = np.arctan2(*trace[0])
    return np.dot(trace,rotation(-alph))

def transform_to_traces(batch, trace_size):
    setX = batch[["xpx", "ypx"]].to_numpy()
    setX = setX[1:]-setX[:-1]
    lenX = setX.shape[0]
    sizeSet = int(np.ceil(lenX/trace_size))
    newSet = np.zeros((sizeSet,trace_size,2))
    for i in range(sizeSet-1):
        newSet[i,:,:] = rotate_trace(setX[i*trace_size:(i+1)*trace_size])
    X = np.reshape(newSet, (sizeSet, trace_size*2))
    return X, newSet

def get_metrics_for_traces():
    metrics_f = [activity, turning_angle, absolute_angles, entropy_for_data, distance_to_wall] 
    names = ["%s_%s"%(m,s) for m,s in product(map(lambda m: m.__name__, metrics_f), [MU_STR, SD_STR])]
    names.remove("%s_%s"%(entropy_for_data.__name__, SD_STR)) # remove entropy_sd
    return metrics_f, names

def transform_to_traces_metric_based(data, trace_size, filter_index, area):
    lenX = data.shape[0]
    sizeSet = lenX//trace_size
    metric_functions, _ = get_metrics_for_traces() #
    newSet = np.zeros((sizeSet,len(metric_functions)*2))
    for i, f in enumerate(metric_functions):
        idx = i*2
        if f.__name__ == distance_to_wall.__name__:
            newSet[:,idx:idx+2] = f(data, trace_size, filter_index, area)[:,:2]
        elif f.__name__ == entropy_for_data.__name__:
            entropy_idx = i*2+1
            newSet[:,idx:idx+2] = f(data, trace_size, filter_index, area)[:,:2] # only take entropy not std over hist. 
        else:
            newSet[:,idx:idx+2] = f(pixel_to_cm(data), trace_size, filter_index)[:,:2]
    newSet = np.delete(newSet, entropy_idx, axis=1)
    #np.nan_to_num(newSet, copy=False, nan=0.0)
    return newSet

def normalize_data_metrics(traces):
    for i in range(traces.shape[1]):
        d_std, d_mean = np.std(traces[:,i]), np.mean(traces[:,i])
        traces[:,i]=(traces[:,i]-d_mean)/d_std
    return traces

def TSNE_vis(X_embedded):
    cmap=["Blues", "Reds"]
    p = sns.kdeplot(x=X_embedded[:,0], y=X_embedded[:,1], cmap=cmap[0], shade=True, thresh=0)
    print(p)
    #plt.scatter(X_embedded[i*half:(i+1)*half,0], X_embedded[i*half:(i+1)*half, 1], marker="o", label=i, alpha=0.5)
    plt.savefig("TSNE.pdf")
    return p

def get_convex_hull(points): 
    hull = ConvexHull(points)
    cycle = points[hull.vertices]
    cycle = np.concatenate([cycle, cycle[:1]])
    return cycle

def TSNE_vis_2(X_embedded, centers=None, clusters=None, scatter=True, fig_name="TSNE_vis"):
    x = X_embedded[:,0]
    y = X_embedded[:,1]
    res = go.Histogram2dContour(x = x, y = y, colorscale = 'Blues')
    fig = go.Figure(res)
    if centers is not None: 
        fig.add_trace(go.Scatter(
            x = centers[:,0], y = centers[:,1],
            xaxis = 'x', yaxis = 'y',
            mode = 'markers',
            marker = dict(
                color = 'rgba(255, 255, 102,0.5)',
                size = 20
            ),
            text=["cluster %d, n: %d"%(i,n) for (i,n) in enumerate(centers[:,2])]
        ))
    if scatter: 
        fig.add_trace(go.Scatter(
            x = x, y = y,
            xaxis = 'x', yaxis = 'y',
            mode = 'markers',
            marker = dict(
                cmax=clusters.max(),
                cmin=0,
                color=clusters,
                colorbar=dict(
                    title="Clusters"
                ),
                colorscale="Viridis",
                #color = 'rgba(0,0,0,0.1)',
                size = 3,
                opacity=0.5
            )
        ))
    
    fig.update_layout( height = 600, width = 600 )
    fig.write_image("%s.pdf"%fig_name)
    return fig

def get_neighbourhood_selection(X_embedded, nSs, c_x=None, c_y=None, radius=1):
    if c_x is None and c_y is None:
        hist, x,y = np.histogram2d(X_embedded[:,0], X_embedded[:,1], bins=10)
        max_x, max_y = int(hist.argmax()/10),hist.argmax()%10
        c_x, c_y = (x[max_x]+x[max_x+1])/2, (y[max_y]+y[max_y+1])/2
    find = np.nonzero((np.abs(X_embedded[:,0]-c_x)+np.abs(X_embedded[:,1]-c_y)) <= radius)
    return nSs[find[0]]

def set_of_neighbourhoods(X_embedded, nSs, radius=1, bins=15):
    hist, x,y = np.histogram2d(X_embedded[:,0], X_embedded[:,1], bins=bins)
    len_x = len(x)
    neighbourhoods = dict()
    centers = list()
    for (max_x,max_y) in zip(range(len_x),hist.argsort()[:,-1]):
        c_x, c_y = (x[max_x]+x[max_x+1])/2, (y[max_y]+y[max_y+1])/2
        centers.append([c_x, c_y,hist[max_x, max_y]])
        neighbourhoods["x:%.2f, y:%.2f, n:%d"%(c_x,c_y, hist[max_x, max_y])] = get_neighbourhood_selection(X_embedded, nSs, c_x, c_y, radius=radius)
    return neighbourhoods, centers

def plot_lines(lines_to_plot, ax=None, title="x:, y: "):
    if ax is not None: ax.set_title(title)
    for line in lines_to_plot:
        error_default_points
        if ax is None:
            plt.plot(line[:,0], line[:,1])
            plt.savefig("lines_%s.pdf"%(title))
        else: 
            ax.plot(line[:,0], line[:,1])

def boxplot_characteristics_of_cluster(traces_c, ax):
    _, metric_names = get_metrics_for_traces()
    ax.boxplot([*traces_c.T], labels=metric_names, showfliers=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")