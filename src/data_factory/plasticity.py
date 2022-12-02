
import os
import pandas as pd
from scipy.stats import entropy as entropy_m
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
from src.config import HOURS_PER_DAY
from src.utils.utile import get_all_days_of_context
from .processing import get_regions_for_fish_key, load_zVals_concat, load_trajectory_data_concat

DIR_PLASTCITY = "plasticity"

def cluster_entropy_plot(parameters, get_clusters_func, fish_keys, n_clusters, name="cluster_entropy_for_days", by_the_hour=False):
    fig = plt.figure(figsize=(10,5))
    ax = fig.subplots()
    days = get_all_days_of_context()
    colors_map = plt.cm.get_cmap(lut=len(fish_keys))
    all_vals_df = df.DataFrame(columns=fish_keys, index=len(days)*HOURS_PER_DAY if by_the_hour else days)
    for j,fk in enumerate(fish_keys):
        for i,d in enumerate(days):
            clusters = get_clusters_func(fk,d)# get_regions_for_fish_key(wshedfile,fk,d)
            if clusters is not None:
                if by_the_hour:
                    time_df = load_trajectory_data_concat(parameters, fk, d)["df_time_index"]
                    time_df=time_df-time_df[0]
                    idx_s = 0
                    for h in range(1,HOURS_PER_DAY+1):
                        where_hour_ends = np.where(time_df >= (h*5*(60**2)))[0]
                        if HOURS_PER_DAY!=h and len(where_hour_ends)==0:
                            print(h,"is not recorded", time_df.shape,fk, d, (h*5*(60**2)))
                            break
                        if HOURS_PER_DAY ==h:idx_e = len(clusters)
                        else: idx_e = where_hour_ends[0]
                        dist = compute_cluster_distribution(clusters[idx_s:idx_e],n_clusters)
                        all_vals_df.loc[(i+1)*HOURS_PER_DAY,fk] = entropy_m(dist)
                        idx_s=idx_e
                else:
                    dist = compute_cluster_distribution(clusters,n_clusters)
                    all_vals_df.loc[i,fk]= entropy_m(dist)
            all_vals_df.plot(kind="scatter",x="index", y=fk,color=colors_map(j))
    a,c = np.polyfit(*zip(*all_vals), 1)
    time = np.arange(len(days)*(HOURS_PER_DAY if by_the_hour else 1))
    corrcof = pearsonr(*zip(*all_vals))
    ax.plot(time, a*time+c, "-", color="k", markersize=12, label="pearsonr: %0.3f"%(corrcof[0]))
    ax.legend()
    ax.set_ylabel("entropy")
    ax.set_xlabel("hours" if by_the_hour else "days")
    fig.tight_layout()
    dir_p = f"{parameters.projectPath}/{DIR_PLASTCITY}"
    os.makedirs(dir_p, exist_ok=True)
    fig.savefig(f"{dir_p}/{name}.pdf")
    return fig

def compute_cluster_distribution(clusters, n):
    uqs, counts = np.unique(clusters, return_counts=True)
    dist = np.zeros(n)[(uqs-1)]=counts/clusters.shape[0]
    return dist

def plot_feature_distribution(input_data, name="distributions_step_angle_dist", n_dfs=[1], names = ["step", "angle", "dist wall"]):
    n_rows, n_cols = len(n_dfs),len(names)
    fig = plt.figure(figsize=(n_cols*5,n_rows*5))
    axes = fig.subplots(n_rows,n_cols, sharex="col", squeeze=False)
    for axes_i, n_df in zip(axes,n_dfs):
        for i, ax in enumerate(axes_i):
            if i==0: 
                ax.set_ylabel('%0.1f sec'%(n_df/5))
            # Draw the plot
            data_in = input_data[:,i]
            new_len = data_in.size//n_df
            data_in = data_in[:n_df*new_len].reshape((new_len, n_df)).mean(axis=1)
            ax.hist(data_in, bins = int(200),density=False,stacked=False, range=(0,4) if i==0 else None,
                     color = 'blue', edgecolor = 'black')
            # Title and labels
            ax.set_xlabel(names[i])
    plt.tight_layout()
    fig.savefig(f"{name}.pdf")
    return fig

def plasticity_cv_fig(parameters, wshedfile, fish_keys, name=None,n_df=50, forall=False):
    fig = plt.figure(figsize=(10,10))
    axes = fig.subplots(3,1)
    fig.suptitle("Coefficient of Variation for %0.1f sec data"%(n_df/5))
    names = ["step length", "turning angle", "distance to wall"]
    colors_map = plt.cm.get_cmap(lut=len(fish_keys))
    sum_data = [{'time': [], 'cv': []}, {'time': [], 'cv': []}, {'time': [], 'cv': []}]
    for j, fk in enumerate(fish_keys):
        sum_data_fk = load_trajectory_data_concat(wshedfile, parameters, fish_key=fk)
        data = sum_data_fk["projections"]
        time_idx = sum_data_fk["df_time_index"].flatten()
        for i, ax in enumerate(axes):
            data_in = data[:,i] if i != 1 else data[:,i]+np.pi # shift for turning angle 
            new_len = data_in.size//n_df
            data_in = data_in[:n_df*new_len].reshape((new_len, n_df)).mean(axis=1)
            n_means = (5*(60**2))//n_df
            sw_len = data_in.size//n_means
            sw_view = data_in[:sw_len*n_means].reshape(sw_len,n_means)
            cv = sw_view.std(axis=1)/sw_view.mean(axis=1)
            time = time_idx[::n_df][:sw_len*n_means:n_means]
            if time.shape!=cv.shape:
                print(time.shape,cv.shape, fk)
                continue
            corrcof = pearsonr(time,cv)
            if forall:
                sum_data[i]["time"].append(time)
                sum_data[i]["cv"].append(cv)
            ax.plot(time, cv, "o",color=colors_map(j), label="%s - pearsonr: %0.3f"%(fk[6:10], corrcof[0]) if not forall else None)
            ax.set_ylabel(names[i])
            if not forall: 
                a,c = np.polyfit(time,cv, 1)
                ax.plot(time, a*time+c, "-", color=colors_map(j))
                ax.legend(ncol=1)
    if forall:
        for i, ax in enumerate(axes):
            time = np.concatenate(sum_data[i]["time"])
            cv = np.concatenate(sum_data[i]["cv"])
            corrcof = pearsonr(time,cv)
            a,b,c = np.polyfit(time,cv, 2)
            t_range = np.arange(0, time.max())
            ax.plot(t_range, a*(t_range**2)+b*t_range+c, "-", color="k", markersize=12, label="pearsonr: %0.3f"%(corrcof[0]))
            ax.legend(ncol=1)
    ax.set_xlabel("tracked data frames")
    fig.tight_layout()
    if name:
        fig.savefig(f"{name}.pdf")
    return fig

