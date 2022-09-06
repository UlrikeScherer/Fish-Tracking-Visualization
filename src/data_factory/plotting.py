
import matplotlib.pyplot as plt
import numpy as np
from random import sample
from time import gmtime, strftime
import motiomapper as mmpy
from src.clustering.clustering import get_results_filepath, boxplot_characteristics_of_cluster

def plot_lines_for_cluster2(
    positions,
    projections,
    area,
    clusters,
    n_clusters,
    limit=20,
    fig_name="cluster_characteristics.pdf",
):
    nrows = 2
    fig, axs = plt.subplots(
        nrows=nrows, ncols=n_clusters, figsize=(n_clusters * 4, nrows * 4), sharey="row"
    )
    for cluster_id in range(1,n_clusters+1):
        ax_b = axs[1, cluster_id-1]
        ax = axs[0, cluster_id-1]
        boxplot_characteristics_of_cluster(projections[clusters == cluster_id], ax_b, metric_names=["speed x", "speed y", "angle"])
        samples_c_idx = np.where(clusters == cluster_id)[0]
        cluster_share = samples_c_idx.shape[0] / projections.shape[0]
        select = sample(range(len(samples_c_idx)), k=limit)
        plot_lines_select(
            positions,
            samples_c_idx[select],
            area, 
            ax=ax,
            title="cluster: %d,      share: %.2f" % (cluster_id, cluster_share),
        )
        ax_b.yaxis.set_tick_params(which="both", labelbottom=True)

    fig.savefig(get_results_filepath(1, fig_name), bbox_inches="tight")
    plt.close(fig)
    
def plot_lines_select(positions, samples_idx, area, ax, title):
    ax.set_title(title)
    plot_area(area, ax)
    for idx in samples_idx:
        s,t = max(idx-50,0),min(idx+50, len(positions)-1)
        ax.plot(positions[s:t, 0], positions[s:t, 1])

def plot_area(area_box, ax):
    area_box = np.concatenate((area_box, [area_box[0]]))
    ax.plot(*area_box.T)
    
    
def ethnogram_of_clusters(clusters, start_time=0, end_time=8*(60**2), fish_key="", day="", write_fig=False):
    wregs = clusters[start_time*5:end_time*5]
    f2min = (60**2)*5
    rows = 4
    len_half = wregs.shape[0]//rows
    step = len_half//10
    ethogram = np.zeros((wregs.max(), len(wregs)))

    for wreg in range(1, wregs.max()+1):
        ethogram[wreg-1, np.where(wregs==wreg)[0]] = 1.0

    ethogram = np.split(ethogram.T, np.array([len_half*i for i in range(1,rows)]))

    fig, axes = plt.subplots(rows, 1, figsize=(30,3*rows))
    axes[0].set_title(f"{fish_key} {day}")
    for k, (e, ax) in enumerate(zip(ethogram, axes.flatten())):
        ax.imshow(e.T, aspect='auto', cmap=mmpy.gencmap())
        ax.set_yticks([i for i in range(0, wregs.max(), 1)])
        ax.set_yticklabels(['Region %i'%(j+1) for j in range(0, wregs.max(), 1)])
        xticklocs = [i for i in range(0,len_half, step)]
        ax.set_xticks(xticklocs)
        ax.set_xticklabels([strftime("%H:%M", gmtime((j+(k*len_half))//5)) for j in xticklocs])

    ax.set_xlabel('Time (H:M)')
    if write_fig:
        path_e = f"{VIS_DIR}/{BLOCK}_ethograms"
        if not os.path.exists(path_e):
            os.mkdir(path_e)
        fig.savefig(f"{path_e}/ethogram_{fish_key}_{day}.pdf")
    return fig