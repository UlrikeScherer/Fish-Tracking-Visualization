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
from .processing import load_summerized_data

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
    axes[0].set_title('Method : %s'%parameters.method)
    sc = axes[0].scatter([],[],marker='o', c="b", s=300)
    area_box = np.concatenate((area_box, [area_box[0]]))
    axes[1].plot(*area_box.T)
    (line,) = axes[1].plot([],[], "-o")
    tstart = start

    def animate(t):
        t = int(t*80)+tstart
        line.set_data(*positions[t-100:t+100].T)
        axes[1].axis('off')
        axes[0].set_title('%s %s H:M:S: %s'%(fish_key, day, strftime("%H:%M:%S",gmtime(t//5))))
        sc.set_offsets(zValues[t])
        sc.set_color(get_color(clusters[t]))
        return mplfig_to_npimage(fig) #im, ax

    def get_color(ci):
        color = ['lightcoral', 'darkorange', 'olive', 'teal', 'violet', 
             'skyblue']
        return color[ci%len(color)]

    anim = VideoClip(animate, duration=40) # will throw memory error for more than 100.
    plt.close()
    if save:
        if not os.path.exists(VIDEOS_DIR):
            os.mkdir(VIDEOS_DIR)
        anim.write_gif(f'{VIDEOS_DIR}/{filename}{fish_key}_{day}.gif', fps=5)
    return anim
