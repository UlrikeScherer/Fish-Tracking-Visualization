import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import sys
from src.utile import csv_of_the_day, get_position_string, get_time_for_day, BLOCK
from src.metrics import num_of_spikes, calc_step_per_frame, mean_sd
from methods import avg_and_sum_angles # import cython functions for faster for-loops. 

mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['lines.linestyle'] = '-'
mpl.rcParams['lines.markersize'] = 1.0
mpl.rcParams["figure.figsize"] = (4,2)

ROOT_img = "plots"

def plots_for_tex(camera_list, day_list, dpi=50):
    i = 0
    N = 2 * len(camera_list) * len(day_list)
    for is_back in [True, False]:
        fo = set_figure(is_back)
        for camera_id in camera_list:
            for day in day_list:
                sys.stdout.write('\r')
                # write the progress to stdout 
                sys.stdout.write("[%-20s] %d%%" % ('='*int(20*i/N), 100*i/N))
                sys.stdout.flush()
                i+=1
                day_df = csv_of_the_day(camera_id, day, is_back=is_back, drop_out_of_scope=True)
                plot_day_camera_fast(day_df, camera_id, day, fo, is_back=is_back, dpi=dpi)

def set_figure(is_back=False):
    xlim=[-5, 90]
    ylim=[-42, 2]
    if is_back: ylim=[-2, 42] 
    fig, ax = plt.subplots(figsize=(5,2.5))
    plt.tight_layout()
    #ax.set_ylabel('y coordinate')
    ax.set_ylim(ylim)
    #ax.set_xlabel('x coordinate')
    ax.set_xlim(xlim)
    line, = ax.plot(xlim, ylim,'b-', alpha=0.7, solid_capstyle="projecting")
    plt.show(block=False)
    return (fig, ax, line)
    
def plot_day_camera_fast(data, camera_id, date, figure_obj, is_back=False, dpi=50):
    (fig, ax, line) = figure_obj
    nrows=4
    ncols=4
    position = get_position_string(is_back)
    batch_size= 100
    
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
            low, up = 0, len(data[idx].x)-1
            
            time_span="{} - {}".format(get_time_for_day(date,nr_of_frames),get_time_for_day(date, nr_of_frames+data[idx].FRAME[up]))
            nr_of_frames+=data[idx].FRAME[up]
            ax.set_title(time_span,fontsize=10)
            
            if data[idx].x.array[-1] <= -1:
                data[idx].drop(data[idx].tail(1).index)
            
            #for k in range(low, up, batch_size-1):
            line.set_data(data[idx].x.array, data[idx].y.array)
            #ax.draw_artist(ax.patch)
            #ax.draw_artist(line)
            remove_text = meta_text_for_plot(ax, data[idx], is_back=is_back)
            
            data_dir = "{}/{}/{}/{}/{}".format(ROOT_img, BLOCK, position, camera_id, date)
            if not os.path.isdir(data_dir):
                os.makedirs(data_dir, exist_ok=True)
            fig.savefig("{}/{}{}.pdf".format(data_dir,i,j),bbox_inches='tight', dpi=dpi)
            remove_text()
    return None

def meta_text_for_plot(ax, batch, is_back=False):
    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    pos_y = y_lim[0] + (y_lim[1] - y_lim[0]) * 0.05 
    pos_x1 = x_lim[0] + (x_lim[1] - x_lim[0]) * 0.01
    pos_x2 = x_lim[0] + (x_lim[1] - x_lim[0]) * 0.7
    if is_back: 
        pos_y = y_lim[1] - ((y_lim[1] - y_lim[0]) * 0.35)
    steps = calc_step_per_frame(batch)
    mean, sd = mean_sd(steps)
    spiks = num_of_spikes(steps)
    avg_alpha, sum_alpha = avg_and_sum_angles(batch)
    N = len(steps)
    meta_l = [r"$\alpha_{avg}: %.3f$"%avg_alpha,r"$\sum \alpha_i : %.f$"%sum_alpha, r"$\mu + \sigma: %.2f + %.2f$"%(mean, sd)]
    meta_r = [" ",r"$N: %.f$"%N, r"$ \# spikes: %.f$"%(spiks)]
    if is_back:
        meta_l.reverse()
        meta_r.reverse()
    text1 = ax.text(pos_x1,pos_y, "\n".join(meta_l))
    text2 = ax.text(pos_x2,pos_y, "\n".join(meta_r))
    return lambda: (text1.remove(), text2.remove())