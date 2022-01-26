import matplotlib.pyplot as plt
import numpy as np
import os
from src.utile import BLOCK
#########################

def plot_activity(data, time_interval):
    """ Plots the average activity my mean and vaiance over time
    input: data, time_interval
    return: figure
    """
    fig, ax = plt.subplots(figsize=(15*(data.shape[0]/300),5))
    plt.tight_layout()
    offset = int(time_interval/2)
    ax.errorbar(range(offset, offset + len(data)*time_interval, time_interval), data[:,0], 
                [data[:,0],data[:,1]], 
                marker='.', linestyle='None', elinewidth=0.7)
    ax.set_xlabel("seconds")
    return fig

def sliding_window(dataset, time_interval, sw, fish_ids=[0], xlabel="seconds", ylabel="average cm/Frame", name="activity"):
    """Summerizes the data for a sliding window and plots a continuous line over time """
    offset = int(time_interval*sw/2)
    if isinstance(dataset, np.ndarray):
        dataset = [[dataset]]
    if isinstance(dataset[0], np.ndarray):
        dataset = [[d] for d in dataset]
    n_fishes = len(fish_ids)
    n_days = len(dataset[0])
    print(n_fishes, n_days)
    ncols=6
    nrows=int(np.ceil(n_days/ncols))
    fig, axes = plt.subplots(ncols = ncols, nrows=nrows, figsize=(ncols*8*(dataset[0][0].shape[0]/100),3*nrows), sharey=True)
    if n_days==1: axes = [axes]
    if nrows > 1: axes = np.ravel(axes)
    plt.tight_layout()
    
    color_map = plt.get_cmap('tab20b').colors + plt.get_cmap('tab20b').colors[:4]
    
    
    for f_idx in range(n_fishes):
        n_days = len(dataset[f_idx])
        for d_idx in range(n_days):
            data = dataset[f_idx][d_idx]
            slide_data = [np.mean(data[i:i+sw,0]) for i in range(0, data.shape[0]-sw)]
            axes[d_idx].plot(range(offset, offset + (len(data)-sw)*time_interval, time_interval), slide_data,'-', label="fish %s"%fish_ids[f_idx], color=color_map[fish_ids[f_idx]])
            if f_idx == 0:
                axes[d_idx].set_title("day %s"%d_idx)
                if d_idx >= (nrows-1)*ncols:
                    axes[d_idx].set_xlabel(xlabel)
                if d_idx % ncols==0:
                    axes[d_idx].set_ylabel(ylabel)
            
    axes[0].legend(loc='upper center', bbox_to_anchor=(3, 1.25),
          ncol=n_fishes, fancybox=True, shadow=True)
    data_dir = "{}/{}/".format("vis", BLOCK)
    os.makedirs(data_dir, exist_ok=True)
    fig.savefig("{}/{}.pdf".format(data_dir,name),bbox_inches='tight', dpi=100)
    return fig
    
def plot_turning_direction(data, time_interval):
    fig, ax = plt.subplots(figsize=(15*(data.shape[0]/300),5))
    plt.tight_layout()
    offset = int(time_interval/2)
    ax.errorbar(range(offset,offset + len(data)*time_interval, time_interval), data[:,0], 
                data[:,1], 
                marker='.', linestyle='None', elinewidth=0.7)
    ax.set_xlabel("seconds")
    return fig

