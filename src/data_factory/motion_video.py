import numpy as np
# this packages helps load and save .mat files older than v7
import hdf5storage, h5py, os
from time import gmtime, strftime
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
# moviepy helps open the video files in Python
from moviepy.editor import VideoClip, VideoFileClip
from moviepy.video.io.bindings import mplfig_to_npimage
import motionmapperpy as mmpy
from .processing import load_summerized_data, get_fish_info_from_wshed_idx, get_regions_for_fish_key
from .utils import get_cluster_seqences
from src.config import STIME
from src.utils import get_date_string, get_seconds_from_day

VIDEOS_DIR = "videos"
def motion_video(wshedfile, parameters,fish_key, day, start=0, end=None, save=False, filename=""):
    try:
        tqdm._instances.clear()
    except:
        pass

    #data 
    sum_data = load_summerized_data(wshedfile,parameters,fish_key,day)
    zValues = sum_data['embeddings']
    positions = sum_data['positions']
    clusters = sum_data["clusters"]
    area_box = sum_data['area']

    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    tfolder = parameters.projectPath+'/%s/'%parameters.method
    with h5py.File(tfolder + 'training_embedding.mat', 'r') as hfile:
        trainingEmbedding = hfile['trainingEmbedding'][:].T
        
    m = np.abs(trainingEmbedding).max()
    sigma=1.0
    _, xx, density = mmpy.findPointDensity(trainingEmbedding, sigma, 511, [-m-10, m+10])
    axes[0].imshow(density, cmap=mmpy.gencmap(), extent=(xx[0], xx[-1], xx[0], xx[-1]), origin='lower')
    axes[0].axis('off')
    axes[0].set_title(' ')
    sc = axes[0].scatter([],[],marker='o', c="b", s=300)
    
    area_box = np.concatenate((area_box, [area_box[0]]))
    axes[1].plot(*area_box.T)
    (line,) = axes[1].plot([],[], "-o")
    tstart = start

    def animate(t):
        t = int(t*50)+tstart
        line.set_data(*positions[t-100:t+100].T)
        axes[1].axis('off')
        axes[0].set_title('%s %s H:M:S: %s'%(fish_key, get_date_string(day), strftime("%H:%M:%S",gmtime(t//5))))
        sc.set_offsets(zValues[t])
        sc.set_color(get_color(clusters[t]))
        return mplfig_to_npimage(fig) #im, ax

    anim = VideoClip(animate, duration=20) # will throw memory error for more than 100.
    plt.close()
    if save:
        dir_v = f'{parameters.projectPath}/{VIDEOS_DIR}'
        if not os.path.exists(dir_v):
            os.mkdir(dir_v)
        anim.write_videofile(f'{dir_v}/{filename}{fish_key}_{day}.mp4', fps=10, audio=False, threads=1)
    return anim

def get_color(ci):
    color = ['lightcoral', 'darkorange', 'olive', 'teal', 'violet', 
         'skyblue']
    return color[ci%len(color)]


def cluster_motion_axes(wshedfile, parameters,fish_key, day, start=0, ax=None, score=0):
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=(5,5))
    sum_data = load_summerized_data(wshedfile,parameters,fish_key,day)
    embeddings = sum_data['embeddings']
    positions = sum_data['positions']
    clusters = sum_data["clusters"]
    area = sum_data['area']
    
    area_box = np.concatenate((area, [area[0]]))
    ax.plot(*area_box.T, color="black")
    (line,) = ax.plot([],[], "-o", color="blue")
    
    day_start = get_seconds_from_day(day+"_"+STIME)
    def update_line(t):
        t = int(t*50)+start
        line.set_data(*positions[t-100:t+100].T)
        ax.axis('off')
        ax.set_title('%s %s %s ratio:%.2f'%(fish_key, get_date_string(day), strftime("%H:%M:%S",gmtime((t//5)+day_start)), score))
        
    return update_line


def cluster_motion_video(wshedfile, parameters, cluster_id):
    clusters = get_regions_for_fish_key(wshedfile)
    results = get_cluster_seqences(clusters, [cluster_id], sw=2*60*5, th=0.8)
    fig, axes = plt.subplots(2, 2, figsize=(10,10))
    update_functions = list()
    for (s,e,score),ax in zip(results[cluster_id], axes.flatten()):
        try:
            fk, day, start, end = get_fish_info_from_wshed_idx(wshedfile,s,e)
            up_f = cluster_motion_axes(wshedfile, parameters, fk, day,start, ax=ax, score=score)
            update_functions.append(up_f)
        except ValueError as e:
            print(e)
    
    def animate(t):
        for f in update_functions:
            f(t)
        return mplfig_to_npimage(fig)
    
    anim = VideoClip(animate, duration=30) # will throw memory error for more than 100.
    plt.close()
    dir_v = f'{parameters.projectPath}/{VIDEOS_DIR}'
    if not os.path.exists(dir_v):
        os.mkdir(dir_v)
    anim.write_videofile(f'{dir_v}/cluster_{str(cluster_id)}.mp4', fps=10, audio=False, threads=4)
    return anim
    
    