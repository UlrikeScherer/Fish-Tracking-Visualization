import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .processing import load_trajectory_data
from .utils import get_individuals_keys, get_days, set_parameters, split_into_batches
from src.config import BLOCK1, BLOCK2, HOURS_PER_DAY


from src.utils.plot_helpers import remove_spines

def plot_covariance(cov_matrix, label="day",title="Correlation",symmetric_bounds=True,clip_percentile=99.5,**imshow_kwargs):
    fig, ax = plt.subplots(figsize=(8,7))
    mask= np.tri(cov_matrix.shape[0], k=-1).T
    vmin, vmax = None, None
    if symmetric_bounds:
        vmax = np.percentile(np.abs(cov_matrix), clip_percentile)
        vmin = -vmax
    cov_matrix = np.ma.array(cov_matrix, mask=mask) 
    cax = ax.imshow(
        cov_matrix,
        vmin=vmin, vmax=vmax,
        aspect="equal", interpolation='none', **imshow_kwargs)
    remove_spines(ax)
    ax.set_title(title)
    ax.set_xlabel(label)
    ax.set_ylabel(label)
    fig.colorbar(cax)
    return fig

def hour_index(days, batch_minutes):
    return np.concatenate([[(d,h) for h in range(0,HOURS_PER_DAY*60, batch_minutes)] for d in days])

def step_length_avg(parameters, fish_keys, batch_minutes=None):
    if batch_minutes:
        hidx1 = hour_index(get_days(parameters, prefix=BLOCK1), batch_minutes=batch_minutes)
        hidx2 = hour_index(get_days(parameters,prefix=BLOCK2), batch_minutes=batch_minutes)
        main_df = pd.DataFrame({
            BLOCK1:hidx1[:,0],
            BLOCK2:hidx2[:,0],
            "minutes":hidx1[:,1]
            })
    else:
        main_df = pd.DataFrame({BLOCK1:get_days(parameters,prefix=BLOCK1), BLOCK2:get_days(parameters, prefix=BLOCK2)})
    for fk in fish_keys:
        block = BLOCK1 if BLOCK1 in fk else BLOCK2
        data = load_trajectory_data(parameters, fk)
        if batch_minutes: 
            step_means = np.concatenate([list(
                map(
                    lambda batch: (
                        np.mean(batch[1]),
                        di["day"][0][0],
                        map_to_batch_minutes(batch[0], batch_minutes)
                                  ),
                    zip(*split_into_batches(di["df_time_index"].flatten(),di["projections"][:,0], batch_size_minutes=batch_minutes))
                )
            ) for di in data])

            step_df = pd.DataFrame({
                fk:step_means[:,0], 
                block:step_means[:,1],
                "minutes":step_means[:,2]
                               }).dropna()
        else:
            step_df = pd.DataFrame({
                fk:[data[i]["projections"][:,0].mean() for i in range(len(data))], 
                block:[data[i]["day"][0][0] for i in range(len(data))]
                               }).dropna()
        main_df = main_df.merge(step_df, on=[block,"minutes"], how="outer")
    return main_df

# maps to the next multiple of batch_minutes
def map_to_batch_minutes(batch, batch_minutes):
    if len(batch)==0: return np.nan
    else: return int((batch[0]//(5*60*batch_minutes))*batch_minutes - (6*60))

if __name__ == "__main__":
    parameters = set_parameters()
    FISH_KEYS = get_individuals_keys(parameters, block=BLOCK1)
    
    # Example usage
    batch_minutes=10
    #matrix = step_length_avg(parameters, FISH_KEYS, batch_minutes=batch_minutes)
    #matrix.to_csv(parameters.projectPath+"/avg_step_by_%dmin.csv"%batch_minutes)
    matrix = pd.read_csv(parameters.projectPath+"/avg_step_by_day.csv")
    cov_matrix = matrix[FISH_KEYS].T.corr(numeric_only=False)
    fig= plot_covariance(cov_matrix, label="day",title="Correlation",symmetric_bounds=True,clip_percentile=99.5,cmap="RdBu_r")
    fig.savefig(parameters.projectPath+"/correlation_avg_step_by_day_block1.pdf")
´