import matplotlib.pyplot as plt
import numpy as np
import os
from src.utile import BLOCK, get_days_in_order, get_date_string
#########################
colors=np.array([
(166,206,227),
(31,120,180),
(178,223,138),
(51,160,44),
(251,154,153),
(227,26,28),
(253,191,111),
(255,127,0),
(202,178,214),
(106,61,154),
(255,255,153),
(177,89,40)])/255

# color map for each of the 24 fishes 
color_map = [colors[int(k/2)] for k in range(colors.shape[0]*2)]

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

def sliding_window(dataset, time_interval, sw, fish_ids=[0], xlabel="seconds", ylabel="average cm/Frame", name="activity", write_fig=False, logscale=False, baseline=None):
    """Summerizes the data for a sliding window and plots a continuous line over time """
    offset = int(time_interval*sw/2)
    x_max = offset
    if isinstance(dataset, np.ndarray):
        dataset = [[dataset]]
    if isinstance(dataset[0], np.ndarray):
        dataset = [[d] for d in dataset]
    n_fishes = len(fish_ids)
    n_days = len(dataset[0])
    print("Number of fishes:",n_fishes," Number of days: ", n_days)
    ncols=6
    nrows=int(np.ceil(n_days/ncols))
    fig, axes = plt.subplots(ncols = ncols, nrows=nrows, figsize=(ncols*6,3*nrows), sharey=True)
    if n_days==1: axes = [axes]
    if nrows > 1: axes = np.ravel(axes)
    plt.tight_layout()
    days_date = [get_date_string(d) for d in get_days_in_order()]
    
    #color_map = plt.get_cmap('tab20b').colors + plt.get_cmap('tab20b').colors[:4]
    
    for f_idx in range(n_fishes):
        n_days = len(dataset[f_idx])
        for d_idx in range(n_days):
            data = dataset[f_idx][d_idx]
            slide_data = [np.mean(data[i:i+sw,0]) for i in range(0, data.shape[0]-sw)]
            x_end = offset + (len(data)-sw)*time_interval
            x_max = max(x_max, x_end) # x_max update to draw the dashed baseline
            axes[d_idx].plot(range(offset, x_end, time_interval), slide_data,'-', label="fish %s"%fish_ids[f_idx], color=color_map[fish_ids[f_idx]], linewidth=2)
            if f_idx == 0:
                axes[d_idx].set_title("Date %s"%days_date[d_idx], y=1.0, pad=-14)
                if logscale:
                    axes[d_idx].set_yscale('log')
                if d_idx >= (nrows-1)*ncols:
                    axes[d_idx].set_xlabel(xlabel)
                if d_idx % ncols==0:
                    axes[d_idx].set_ylabel(ylabel)
    if baseline != None:
        for i in range(n_days):
            axes[i].plot((offset, x_max), (baseline, baseline), ":", color="black")
                
    for i in range(n_days, len(axes)):
        axes[i].axis('off')
            
    leg = axes[0].legend(loc='upper center', bbox_to_anchor=(ncols/2 + 0.15, 1.55),
          ncol=n_fishes, fancybox=True, fontsize=18, markerscale=2)
    for line in leg.get_lines():
        line.set_linewidth(7.0)
    
    if write_fig:
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

