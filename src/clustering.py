from re import L, T
from src.transformation import rotation, px2cm, pixel_to_cm
from methods import turning_directions, calc_steps, tortuosity_of_chunk, activity, turning_angle, absolute_angles
from src.metrics import entropy_for_chunk, entropy_for_data, tortuosity, distance_to_wall
from src.utile import get_error_indices,error_points_out_of_area, get_fish2camera_map, csv_of_the_day, BACK
from src.tank_area_config import get_area_functions
from itertools import product
import pandas as pd
from scipy.spatial import ConvexHull
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

MU_STR, SD_STR = "mu", "sd"

def get_traces(fish_indices, days, trace_size):
    fish2cams = get_fish2camera_map()
    Xs, nSs = list(), list()
    area_func = get_area_functions()
    
    for i in fish_indices:
        cam, is_back = fish2cams[i][0], fish2cams[i][1]==BACK
        fish_key = "%s_%s"%(cam, fish2cams[i,1])
        for d in days:
            keys, batches = csv_of_the_day(cam, d, is_back=is_back)
            if len(batches) == 0: continue
            b = pd.concat(batches)
            fit_len = fit_data_to_trace_size(len(b), trace_size) 
            data = b[["xpx", "ypx"]].to_numpy()[:fit_len]
            area_tuple = (fish_key,area_func(fish_key, day=d))
            ## filter for errorouse datapoints 
            filter_index = get_error_indices(b).to_numpy()[:fit_len] | error_points_out_of_area(data, area_tuple, day=d)[:fit_len]

            X=transfrom_to_traces_metric_based(data, trace_size, filter_index, area_tuple)
            Xs.append(X)
            nSs.append(trajectory_snippets(data, trace_size))
    traces = np.concatenate(Xs)
    traces = normalize_data_metrics(traces)
    nSs = np.concatenate(nSs)
    return traces, nSs

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

def transfrom_to_traces(batch, trace_size):
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
    return metrics_f, names

def transfrom_to_traces_metric_based(data, trace_size, filter_index, area):
    lenX = data.shape[0]
    sizeSet = lenX//trace_size
    metric_functions, names = get_metrics_for_traces() #
    newSet = np.zeros((sizeSet,len(metric_functions)*2))
    for i, f in enumerate(metric_functions):
        idx = i*2
        if f.__name__ == distance_to_wall.__name__ or f.__name__ == entropy_for_data.__name__:
            newSet[:,idx:idx+2] = f(data, trace_size, filter_index, area)[:,:2]
        else:
            newSet[:,idx:idx+2] = f(pixel_to_cm(data), trace_size, filter_index)[:,:2]
    remove_7 = np.array(names) != "%s_%s" % (entropy_for_data.__name__, SD_STR)
    np.nan_to_num(newSet, copy=False, nan=0.0)
    return newSet[:, remove_7]
        
def normalize_data(traces):
    d_std, d_mean = np.std(traces[:,0::2]), np.mean(traces[:,0::2])
    a_std, a_mean = np.std(traces[:,1::2]), np.mean(traces[:,1::2])
    traces[:,0::2]=(traces[:,0::2]-d_mean)/d_std
    traces[:,1::2]=(traces[:,1::2]-a_mean)/a_std
    return traces

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

def plot_lines_angle(lines_to_plot, limit=20):
    for line in lines_to_plot[:limit]:
        angels = np.cumsum(line[:,1])
        x = line[:,0]*np.cos(angels)
        y = line[:,0]*np.sin(angels)
        plt.plot(np.cumsum(x), np.cumsum(y))
    plt.savefig("lines_exp.pdf")
    
def plot_lines_cumsum(lines_to_plot, limit=20, ax=None, title="x:, y: "):
    if ax is not None: ax.set_title(title)
    for line in lines_to_plot[:limit]:
        line = np.cumsum(px2cm(line),axis=0)
        if ax is None:
            plt.plot(line[:,0], line[:,1])
        else: 
            ax.plot(line[:,0], line[:,1])
            
    plt.savefig("lines_exp.pdf")

def plot_lines(lines_to_plot, ax=None, title="x:, y: "):
    if ax is not None: ax.set_title(title)
    for line in lines_to_plot:
        if ax is None:
            plt.plot(line[:,0], line[:,1])
        else: 
            ax.plot(line[:,0], line[:,1])

def boxplot_characteristics_of_cluster(traces_c, ax):
    _, metric_names = get_metrics_for_traces()
    metric_names.remove("%s_%s"%(entropy_for_data.__name__, SD_STR))
    ax.boxplot([*traces_c.T], labels=metric_names, showfliers=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")