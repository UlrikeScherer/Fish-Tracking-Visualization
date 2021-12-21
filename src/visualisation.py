import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from src.utile import csv_of_the_day, get_position_string, get_time_for_day
from src.metrics import meta_text_for_plot

mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['lines.linestyle'] = '-'
mpl.rcParams['lines.markersize'] = 1.0
mpl.rcParams["figure.figsize"] = (4,2)

ROOT_img = "plots"

def plots_for_tex(camera_list, day_list, dpi=50):
    for is_back in [True, False]:
        fo = set_figure(is_back)
        for camera_id in camera_list:
            for day in day_list:
                day_df = csv_of_the_day(camera_id, day, is_back=is_back, drop_out_of_scope=True)
                plot_day_camera_fast(day_df, camera_id, day, fo, is_back=is_back, dpi=dpi)

def set_figure(is_back=False):
    xlim=[-5, 90]
    ylim=[-40, 2]
    if is_back: ylim=[-2, 40] 
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
            
            data_dir = "{}/{}/{}/{}".format(ROOT_img, position, camera_id, date)
            if not os.path.isdir(data_dir):
                os.makedirs(data_dir, exist_ok=True)
            fig.savefig("{}/{}{}.jpeg".format(data_dir,i,j),bbox_inches='tight', dpi=dpi)
            remove_text()
                
    return None