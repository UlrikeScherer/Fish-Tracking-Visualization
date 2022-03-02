import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import sys
from src.utile import csv_of_the_day, get_position_string, get_time_for_day, BLOCK, ROOT_img, fish2camera, BACK
from src.metrics import num_of_spikes, calc_step_per_frame, mean_sd
from methods import avg_and_sum_angles # import cython functions for faster for-loops. 

mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['lines.linestyle'] = '-'
mpl.rcParams['lines.markersize'] = 1.0
mpl.rcParams["figure.figsize"] = (4,2)

def plots_for_tex(day_list, marker_char=""):
    i = 0
    N = len(fish2camera) * len(day_list)
    for i,(camera_id,pos) in enumerate(fish2camera):
        is_back= pos==BACK
        fo = set_figure(is_back=is_back,  marker_char=marker_char)
        for day in day_list:
            sys.stdout.write('\r')
            # write the progress to stdout 
            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*i/N), 100*i/N))
            sys.stdout.flush()
            i+=1
            day_df = csv_of_the_day(camera_id, day, is_back=is_back, drop_out_of_scope=True)
            plot_day_camera_fast(day_df, camera_id, day, fo, is_back=is_back)

def set_figure(is_back=False, marker_char=""):
    """
    is_back: 
    marker_char: default "" no marker, optional maker_char = [o,*,+....]
    return: a figure triple (fig, ax, line)
    """
    xlim=[-5, 90]
    ylim=[-42, 2]
    if is_back: ylim=[-2, 42] 
    fig, ax = plt.subplots(figsize=(5,2.5))
    plt.tight_layout()
    #ax.set_ylabel('y coordinate')
    #ax.set_ylim(ylim)
    #ax.set_xlabel('x coordinate')
    #ax.set_xlim(xlim)
    line, = ax.plot(xlim, ylim,'b-%s'%marker_char, alpha=0.7, solid_capstyle="projecting", markersize=0.2)
    plt.show(block=False)
    return (fig, ax, line)
    
def plot_day_camera_fast(data, camera_id, date, figure_obj, is_back=False):
    nrows=4
    ncols=4
    position = get_position_string(is_back)
    data_dir = "{}/{}/{}/{}/{}".format(ROOT_img, BLOCK, position, camera_id, date)
    
    if len(data)==0:
        return None
    
    nr_of_frames = 0
    
    for i in range(nrows):       
        for j in range(ncols):
            if (i == j and j == 0):
                continue
            idx = i * ncols + j - 1
            if idx >= len(data):
                break
            batch = data[idx]
            up = len(batch.x)-1
            time_span="batch: {},    {} - {}".format(idx,get_time_for_day(date,nr_of_frames),get_time_for_day(date, nr_of_frames+batch.FRAME[up]))
            
            subplot_trajectory(figure_obj, batch, date, data_dir, "%s%s"%(i,j), time_span=time_span, is_back=is_back) # plot ij.pdf
            nr_of_frames+=batch.FRAME[up]
    return None

def subplot_trajectory(fig_obj, batch, date, directory, name, time_span="batch: 1,   00:00:00 - 00:30:00", is_back=False, write_fig=False):
    (fig, ax, line) = fig_obj
    #low, up = 0, len(batch.x)-1
    ax.set_title(time_span,fontsize=10)

    if batch.x.array[-1] <= -1:
        batch.drop(batch.tail(1).index)

    #for k in range(low, up, batch_size-1):
    line.set_data(batch.x.array, batch.y.array)
    #ax.draw_artist(ax.patch)
    #ax.draw_artist(line)
    remove_text = meta_text_for_trajectory(ax, batch, is_back=is_back)
    
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    if write_fig:
        fig.savefig("{}/{}.pdf".format(directory,name),bbox_inches='tight')
    remove_text()
    return fig

def meta_text_for_trajectory(ax, batch, is_back=False):
    steps = calc_step_per_frame(batch)
    mean, sd = mean_sd(steps)
    spiks = num_of_spikes(steps)
    avg_alpha, sum_alpha = avg_and_sum_angles(batch)
    N = len(steps)
    text_l = [r"$\alpha_{avg}: %.3f$"%avg_alpha,r"$\sum \alpha_i : %.f$"%sum_alpha, r"$\mu + \sigma: %.2f + %.2f$"%(mean, sd)]
    text_r = [" ",r"$N: %.f$"%N, r"$ \# spikes: %.f$"%(spiks)]
    return meta_text_for_plot(ax, text_l=text_l, text_r=text_r, is_back=is_back)

    
def meta_text_for_plot(ax, text_l=[], text_r=[], is_back=False):
    """
    Optimal for 3 entries lext_l and 2 entries text_r
    Returns a callback function to remove the text from ax 
    """
    pos_x1, pos_x2, pos_y = get_text_positions(ax, is_back)
    if is_back:
        text_l.reverse()
        text_r.reverse()
    text1 = ax.text(pos_x1,pos_y, "\n".join(text_l))
    text2 = ax.text(pos_x2,pos_y, "\n".join(text_r))
    return lambda: (text1.remove(), text2.remove())

def get_text_positions(ax, is_back):
    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    pos_y = y_lim[0] + (y_lim[1] - y_lim[0]) * 0.05 
    pos_x1 = x_lim[0] + (x_lim[1] - x_lim[0]) * 0.01
    pos_x2 = x_lim[0] + (x_lim[1] - x_lim[0]) * 0.7
    if is_back: 
        pos_y = y_lim[1] - ((y_lim[1] - y_lim[0]) * 0.35)
    return pos_x1, pos_x2, pos_y
    