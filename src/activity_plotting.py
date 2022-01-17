import matplotlib.pyplot as plt
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

def sliding_window(dataset, time_interval, sw, fish_ids=[0], xlabel="seconds", ylabel="average cm/Frame"):
    """Summerizes the data for a sliding window and plots a continuous line over time """
    offset = int(time_interval*sw/2)
    if isinstance(dataset, np.ndarray):
        dataset = [dataset]
    fig, ax = plt.subplots(figsize=(15*(dataset[0].shape[0]/300),5))
    plt.tight_layout()
    for (i, data) in enumerate(dataset):
        slide_data = [np.mean(data[i:i+sw,0]) for i in range(0, data.shape[0]-sw)]
        ax.plot(range(offset, offset + (len(data)-sw)*time_interval, time_interval), slide_data,'-', label="fish %s"%fish_ids[i])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
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

