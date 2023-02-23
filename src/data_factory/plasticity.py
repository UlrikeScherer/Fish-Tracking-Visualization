
import os
import pandas as pd
from scipy.stats import entropy as entropy_m
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
from src.config import BLOCK1, BLOCK2, HOURS_PER_DAY
from src.utils.plot_helpers import remove_spines
from .utils import get_days, split_into_batches
from .processing import load_trajectory_data_concat

DIR_PLASTCITY = "plasticity"
def day2date(d):
    return "%s.%s.%s" % (d[4:6], d[6:8],d[:4])

def correlation_fit(x,y,deg=1):
    if x.shape[0] < deg:
        return [], [], np.nan, [np.nan for i in range(deg+1)]
    abc = np.polyfit(x,y, deg)
    corrcof = pearsonr(x,y)
    time = np.linspace(x.min(),x.max(),100)
    fit_y = np.polyval(abc,time)
    return time, fit_y, corrcof[0], abc

def plot_index_columns(df=None, columns=None, title=None, xlabel="index", ylabel="entropy", filename=None, ax=None, forall=True, fit_degree=1):
    if df is None:
        if filename is None:
            raise ValueError("filename must be provided if df is not provided")
        df = pd.read_csv(filename+".csv")
    if ax is None:
        fig = plt.figure(figsize=(10,5))
        axis = fig.subplots()
    else:
        axis = ax

    colors_map = plt.cm.get_cmap(lut=len(columns))
    df = df[columns]
    leg_h = []
    for i,col in enumerate(columns):
        time = df.index.to_numpy()
        y = df[col].to_numpy().astype(np.float64)
        f_nan = ~ np.isnan(y)
        x,y=time[f_nan],y[f_nan]
        axis.scatter(x=x,y=y,color=colors_map(i), s=1)
        if not forall: 
            t,f_y, corr, abc = correlation_fit(x,y,deg=fit_degree)
            line, = axis.plot(t, f_y, "-", color=colors_map(i))
            leg_h.append((line,"%s - pearsonr: %0.3f"%(col[13:17], corr)))

    if forall:
        x = np.column_stack((df.index.to_numpy() for i in range(len(columns)))).reshape(-1).astype(np.int)
        y = df.to_numpy().reshape(-1).astype(np.float64)
        is_not_nan = ~ np.isnan(y)
        x,y = x[is_not_nan],y[is_not_nan]
        t,f_y, corr, abc = correlation_fit(x,y,deg=fit_degree)
        line, = axis.plot(t, f_y, "-", color="k", markersize=12)
        leg_h.append((line,"pearsonr: %0.3f"%(corr)))
    axis.legend(*zip(*leg_h), loc="best")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)

    if ax is not None:
        return None
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename+".png")
    return fig

def cluster_entropy_plot(parameters, get_clusters_func, fish_keys, n_clusters, name="cluster_entropy_for_days", by_the_hour=False, fit_degree=1, forall=True):
    days = get_days(parameters,prefix=fish_keys[0].split("_")[0])
    all_vals_df = pd.DataFrame(columns=fish_keys, index=range(1,1+(len(days)*HOURS_PER_DAY if by_the_hour else len(days))))
    for j,fk in enumerate(fish_keys):
        for i,d in enumerate(get_days(parameters=parameters, prefix=fk)):
            clusters = get_clusters_func(fk,d)
            if clusters is not None:
                if by_the_hour:
                    time_df = load_trajectory_data_concat(parameters, fk, d)["df_time_index"]
                    _,cluster_hourly = split_into_batches(time_df, clusters)
                    all_vals_df.loc[HOURS_PER_DAY*(i):HOURS_PER_DAY*(i+1),fk] = [entropy_m(compute_cluster_distribution(c)) for c in cluster_hourly]
                else:
                    dist = compute_cluster_distribution(clusters,n_clusters)
                    all_vals_df.loc[i+1,fk]= entropy_m(dist)
    
    dir_p = f"{parameters.projectPath}/{DIR_PLASTCITY}/{name}"
    os.makedirs(dir_p, exist_ok=True)
    if not by_the_hour:
        all_vals_df = all_vals_df.join(pd.DataFrame({f"date_{BLOCK1}":map(day2date,get_days(parameters, prefix=BLOCK1)), f"date_{BLOCK2}":map(day2date, get_days(parameters, prefix=BLOCK2))}, index=all_vals_df.index), how="left")
    time_str = "hourly" if by_the_hour else "daily"
    filename = f"{dir_p}/{name}_{time_str}_{n_clusters}"
    all_vals_df.to_csv(f"{filename}.csv")
    fig = plot_index_columns(df=all_vals_df, columns=fish_keys, title=f"{name} {time_str} {n_clusters}", filename=filename, ylabel="entropy", xlabel=time_str, fit_degree=fit_degree, forall=forall)
    return fig, all_vals_df

def compute_cluster_distribution(clusters, n):
    uqs, counts = np.unique(clusters, return_counts=True)
    dist = np.zeros(n)[(uqs-1)]=counts/clusters.shape[0]
    return dist

def plot_feature_distribution(input_data, name="distributions_step_angle_dist", n_dfs=[1], names = ["step length", "turning angle", "distance to the wall"]):
    n_rows, n_cols = len(n_dfs),len(names)
    fig = plt.figure(figsize=(n_cols*4,n_rows*4))
    axes = fig.subplots(n_rows,n_cols, sharex="col", squeeze=False)
    for axes_i, n_df in zip(axes,n_dfs):
        for i, ax in enumerate(axes_i):
            if i==0: 
                ax.set_ylabel("counts")#'%0.1f sec'%(n_df/5))
            # Draw the plot
            data_in = input_data[:,i]
            new_len = data_in.size//n_df
            data_in = data_in[:n_df*new_len].reshape((new_len, n_df)).mean(axis=1)
            ax.hist(data_in, bins = 50,density=False,stacked=False, range=(0,8) if i==0 else None,
                     color = '#6161b0', edgecolor='#6161b0', log=True)
            # Title and labels
            ax.set_title("$\mu=%.02f$, $\sigma=%.02f$"%(data_in.mean(), data_in.std()), size = 12)
            ax.set_xlabel(names[i], size = 12)
            remove_spines(ax)
    #plt.tight_layout()
    fig.savefig(f"{name}.pdf")
    return fig

def plasticity_cv_fig(parameters, fish_keys, n_df=50, forall=False, by_the_hour=True, fit_degree=1):
    """
    Compute the coefficient of variation for the step length, turning angle and distance to wall
    for each fish and each day. The coefficient of variation is computed for each hour of the day
    parameters: parameters object
    fish_keys: list of fish keys
    n_df: number of data frames to average over
    forall: interpolate over all data points
    by_the_hour: compute the coefficient of variation for each hourly or daily
    fit_degree: degree of the polynomial fit
    """
    fig = plt.figure(figsize=(10,10))
    axes = fig.subplots(3,1)
    fig.suptitle("Coefficient of Variation for %0.1f sec data"%(n_df/5))
    names = ["step length", "turning angle", "distance to wall"]
    days = get_days(parameters, prefix=BLOCK1)
    size = len(days)*(HOURS_PER_DAY) if by_the_hour else len(days)
    sum_data = [np.full((size,len(fish_keys)),np.nan) for i in range(3)]
    for j, fk in enumerate(fish_keys):
        for di, day in enumerate(get_days(parameters=parameters, prefix=fk.split("_")[0])):
            sum_data_fk = load_trajectory_data_concat(parameters, fk=fk, day=day)
            if sum_data_fk is None: continue
            data = sum_data_fk["projections"]
            time_df = sum_data_fk["df_time_index"].flatten()
            for i in range(3):
                if by_the_hour:
                    _, datas = split_into_batches(time_df, data[:,i] if i != 1 else data[:,i]+np.pi) # shift for turning angle
                    for h, data_in in enumerate(datas):
                        new_len = data_in.size//n_df
                        data_means = data_in[:n_df*new_len].reshape((new_len, n_df)).mean(axis=1)
                        cv = data_means.std()/data_means.mean()
                        sum_data[i][(di*HOURS_PER_DAY)+(h),j]=cv
                else:
                    data_in = data[:,i] if i != 1 else data[:,i]+np.pi
                    new_len = data_in.size//n_df
                    data_means = data_in[:n_df*new_len].reshape((new_len, n_df)).mean(axis=1)
                    cv = data_means.std()/data_means.mean()
                    sum_data[i][di,j]=cv
    filename = f"{parameters.projectPath}/{DIR_PLASTCITY}/cv"
    os.makedirs(filename, exist_ok=True)
    time_str = "hourly" if by_the_hour else "daily"
    for i in range(3):
        df = pd.DataFrame(sum_data[i], columns=fish_keys, index=range(1, size+1))
        if not by_the_hour:
            df = df.join(pd.DataFrame({f"date_{BLOCK1}":map(day2date,get_days(parameters, prefix=BLOCK1)), f"date_{BLOCK2}":map(day2date, get_days(parameters, prefix=BLOCK2))}, index=df.index), how="left")
        _ = plot_index_columns(df, ax=axes[i], columns=fish_keys, ylabel="cv", xlabel="hour", title=names[i], forall=forall, fit_degree=fit_degree)
        df.to_csv(f"{filename}/cv_{names[i]}_{time_str}_ndf_{n_df}.csv")

    fig.tight_layout()
    fig.savefig(f"{filename}/cv_{time_str}_ndf_{n_df}.pdf")
    return fig

