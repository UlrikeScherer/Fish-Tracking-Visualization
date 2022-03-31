import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import sys
from src.utile import csv_of_the_day, get_position_string, get_time_for_day, BLOCK, ROOT_img, get_fish2camera_map, BACK, STIME, FEEDINGTIME
from src.metrics import num_of_spikes, calc_step_per_frame, mean_sd
from src.transformation import pixel_to_cm
from methods import avg_and_sum_angles # import cython functions for faster for-loops. 


mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['lines.linestyle'] = '-'
mpl.rcParams['lines.markersize'] = 1.0
mpl.rcParams["figure.figsize"] = (4,2)

class Trajectory:

    is_feeding = False

    def __init__(self, marker_char="", y_max=55, write_fig=True):
        self.marker_char = marker_char
        self.y_max = y_max
        self.y_min = 2
        self.x_max = 90
        self.x_min = -5
        self.figsize = (5,(self.y_min+self.y_max)/(-self.x_min+self.x_max)*5)
        self.write_fig = write_fig
        self.fish2camera=get_fish2camera_map()
        self.N_fishes = self.fish2camera.shape[0]
        self.fig_obj_front = self.set_figure(is_back=False)
        self.fig_obj_back = self.set_figure(is_back=True)

    def reset_data(self):
        pass

    def subplot_function(self, batch, date, directory, name, fish_id, time_span="batch: 1,   00:00:00 - 00:30:00", is_back=False):
        (fig, ax, line) = self.fig_obj_back if is_back else self.fig_obj_front
        
        ax.set_title(time_span,fontsize=10)
        if batch.x.array[-1] <= -1:
            batch.drop(batch.tail(1).index)

        batchxy = pixel_to_cm(batch[["xpx", "ypx"]].to_numpy())
        line.set_data(*batchxy.T)
        #ax.draw_artist(ax.patch)
        #ax.draw_artist(line)
        remove_text = self.meta_text_for_trajectory(ax, batchxy, batch.FRAME.array, is_back=is_back)
        
        if self.write_fig:
            self.write_figure(fig, directory, name)
        remove_text()
        return fig

    def write_figure(self, fig, directory, name):
        if not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)
        fig.savefig("{}/{}.pdf".format(directory,name),bbox_inches='tight')

    def meta_text_for_trajectory(self, ax, batchxy, frames, is_back=False):
        steps = calc_step_per_frame(batchxy, frames)
        mean, sd = mean_sd(steps)
        spiks = num_of_spikes(steps)
        avg_alpha, sum_alpha = avg_and_sum_angles(batchxy)
        N = len(steps)
        text_l = [r"$\alpha_{avg}: %.3f$"%avg_alpha,r"$\sum \alpha_i : %.f$"%sum_alpha, r"$\mu + \sigma: %.2f + %.2f$"%(mean, sd)]
        text_r = [" ",r"$N: %.f$"%N, r"$ \# spikes: %.f$"%(spiks)]
        return self.meta_text_for_plot(ax, text_l=text_l, text_r=text_r, is_back=is_back)

        
    def meta_text_for_plot(self, ax, text_l=[], text_r=[], is_back=False):
        """
        Optimal for 3 entries lext_l and 2 entries text_r
        Returns a callback function to remove the text from ax 
        """
        pos_x1, pos_x2, pos_y = self.get_text_positions(ax, is_back)
        if is_back:
            text_l.reverse()
            text_r.reverse()
        text1 = ax.text(pos_x1,pos_y, "\n".join(text_l))
        text2 = ax.text(pos_x2,pos_y, "\n".join(text_r))
        return lambda: (text1.remove(), text2.remove())

    def get_text_positions(self, ax, is_back):
        x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
        pos_y = y_lim[0] + (y_lim[1] - y_lim[0]) * 0.05 
        pos_x1 = x_lim[0] + (x_lim[1] - x_lim[0]) * 0.01
        pos_x2 = x_lim[0] + (x_lim[1] - x_lim[0]) * 0.7
        if is_back: 
            pos_y = y_lim[1] - ((y_lim[1] - y_lim[0]) * 0.35)
        return pos_x1, pos_x2, pos_y

    def plots_for_tex(self, fish_ids, day_list):
        n_D = len(day_list)
        N = len(self.fish2camera[fish_ids])*n_D
        self.reset_data()
        for i, fish_idx in enumerate(fish_ids):
            camera_id,pos = self.fish2camera[fish_idx]
            is_back = pos==BACK
            for j, day in enumerate(day_list):
                sys.stdout.write('\r')
                # write the progress to stdout 
                progress = (i*n_D+j)
                sys.stdout.write("[%-20s] %d%%" % ('='*int(20*progress/N), 100*progress/N))
                sys.stdout.flush()
        
                day_df = csv_of_the_day(camera_id, day, is_back=is_back, drop_out_of_scope=True, is_feeding=self.is_feeding)
                self.plot_day_camera_fast(day_df, camera_id, day, fish_idx, is_back=is_back)

    def set_figure(self, is_back=False):
        """
        is_back:  
        marker_char: default "" no marker, optional maker_char = [o,*,+....]
        return: a figure triple (fig, ax, line)
        """
        xlim=[self.x_min, self.x_max]
        ylim=[-self.y_min, self.y_max] if is_back else [-self.y_max, self.y_min]
        fig, ax = plt.subplots(figsize=self.figsize)
        plt.tight_layout()
        line, = ax.plot(xlim, ylim,'b-%s'%self.marker_char, alpha=0.7, solid_capstyle="projecting", markersize=0.2)
        plt.show(block=False)
        return (fig, ax, line)
        
    def plot_day_camera_fast(self, data, camera_id, date, fish_id, is_back=False):
        position = get_position_string(is_back)
        time_dir = FEEDINGTIME if self.is_feeding else STIME
        data_dir = "{}/{}/{}/{}/{}/{}".format(ROOT_img,time_dir, BLOCK, position, camera_id, date)
        
        if len(data)==0:
            return None
        
        nr_of_frames = 0
        for idx in range(len(data)):       
                batch = data[idx]
                up = len(batch.x)-1
                time_span="batch: {},    {} - {}".format(idx,get_time_for_day(date,nr_of_frames),get_time_for_day(date, nr_of_frames+batch.FRAME[up]))
                
                fig = self.subplot_function(batch, date, data_dir, "%02d"%(idx), fish_id, time_span=time_span, is_back=is_back) # plot ij.pdf
                nr_of_frames+=batch.FRAME[up]
        return None

        #__set_figure = self.set_figure # copy set_figure
