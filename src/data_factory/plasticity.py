
import os
import pandas as pd
from scipy.stats import entropy as entropy_m
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
from src.config import BLOCK1, BLOCK2, HOURS_PER_DAY
from .utils import get_days
from src.utils.utile import get_all_days_of_context
from .processing import get_regions_for_fish_key, load_zVals_concat, load_trajectory_data_concat

DIR_PLASTCITY = "plasticity"

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
        #print(y, col)
        f_nan = ~ np.isnan(y)
        x,y=time[f_nan],y[f_nan]
        axis.scatter(x=x,y=y,color=colors_map(i))
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
                    time_df=time_df-time_df[0]
                    idx_s = 0
                    for h in range(1,HOURS_PER_DAY+1):
                        where_hour_ends = np.where(time_df >= (h*5*(60**2)))[0]
                        if HOURS_PER_DAY!=h and len(where_hour_ends)==0:
                            print(h,"is not recorded", time_df.shape,fk, d, (h*5*(60**2)))
                            break
                        if HOURS_PER_DAY==h:idx_e = len(clusters)
                        else: idx_e = where_hour_ends[0]
                        dist = compute_cluster_distribution(clusters[idx_s:idx_e],n_clusters)
                        all_vals_df.loc[(HOURS_PER_DAY*(i)+h),fk] = entropy_m(dist)
                        idx_s=idx_e
                else:
                    dist = compute_cluster_distribution(clusters,n_clusters)
                    all_vals_df.loc[i+1,fk]= entropy_m(dist)
    
    dir_p = f"{parameters.projectPath}/{DIR_PLASTCITY}/{name}"
    os.makedirs(dir_p, exist_ok=True)
    day2date = lambda d: "%s.%s.%s" % (d[4:6], d[6:8],d[:4])
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
    #plt.tight_layout()
    fig.savefig(f"{name}.pdf")
    return fig

def plasticity_cv_fig(parameters, fish_keys, name=None,n_df=50, forall=False, fit_degree=1):
    fig = plt.figure(figsize=(10,10))
    axes = fig.subplots(3,1)
    fig.suptitle("Coefficient of Variation for %0.1f sec data"%(n_df/5))
    names = ["step length", "turning angle", "distance to wall"]
    days = get_days(parameters, prefix=BLOCK1)
    sum_data = [np.full((len(days)*(HOURS_PER_DAY+1),len(fish_keys)),np.nan) for i in range(3)]
    for j, fk in enumerate(fish_keys):
        sum_data_fk = load_trajectory_data_concat(parameters, fk=fk)
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
            sum_data[i][(time//(5*(60**2))).astype(int),j]=cv

    filename = f"{parameters.projectPath}/{DIR_PLASTCITY}/cv"
    os.makedirs(filename, exist_ok=True)
    for i in range(3):
        df = pd.DataFrame(sum_data[i], columns=fish_keys)
        _ = plot_index_columns(df, ax=axes[i], columns=fish_keys, ylabel="cv", xlabel="hour", title=names[i], forall=forall, fit_degree=fit_degree)
        df.to_csv(f"{filename}/cv_{names[i]}.csv")

    fig.tight_layout()
    if name:
        fig.savefig(f"{name}.pdf")
    return fig

