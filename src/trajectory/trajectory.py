import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import sys
from src.config import BLOCK, PLOTS_TRAJECTORY, BACK, STIME, FEEDINGTIME
from src.utils import (
    csv_of_the_day,
    get_position_string,
    get_time_for_day,
    get_fish2camera_map,
    get_days_in_order,
)
from src.metrics import (
    num_of_spikes,
    calc_length_of_steps,
    get_gaps_in_dataframes,
    activity_mean_sd,
)
from src.utils.transformation import pixel_to_cm
from src.methods import (
    avg_and_sum_angles,
)  # import cython functions for faster for-loops.

mpl.rcParams["lines.linewidth"] = 0.5
mpl.rcParams["lines.linestyle"] = "-"
mpl.rcParams["lines.markersize"] = 1.0
mpl.rcParams["figure.figsize"] = (4, 2)


class Figure:
    def __init__(self, is_back=False, marker_char=""):
        self.is_back = is_back
        self.marker_char = marker_char
        self.y_max = 55
        self.y_min = 2
        self.x_max = 90
        self.x_min = -5
        self.figsize = (5, (self.y_min + self.y_max) / (-self.x_min + self.x_max) * 5)
        self.fig, self.ax, self.line = self.set_figure()

    def set_figure(self):
        """
        is_back:
        marker_char: default "" no marker, optional maker_char = [o,*,+....]
        return: a figure triple (fig, ax, line)
        """
        xlim = [self.x_min, self.x_max]
        ylim = [-self.y_min, self.y_max] if self.is_back else [-self.y_max, self.y_min]
        fig, ax = plt.subplots(figsize=self.figsize)
        plt.tight_layout()
        (line,) = ax.plot(
            xlim,
            ylim,
            "b-%s" % self.marker_char,
            alpha=0.7,
            solid_capstyle="projecting",
            markersize=0.2,
        )
        plt.show(block=False)
        return (fig, ax, line)

    def write_figure(self, directory, name):
        if not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)
        self.fig.savefig("{}/{}.pdf".format(directory, name), bbox_inches="tight")

    def meta_text_for_trajectory(
        self, mean, sd, avg_alpha, sum_alpha, spikes, misses, N
    ):
        text_l = [
            r"$\alpha_{avg}: %.3f$" % avg_alpha,
            r"$\sum \alpha_i : %.f$" % sum_alpha,
            r"$\mu + \sigma: %.2f + %.2f$" % (mean, sd),
        ]
        text_r = [
            r"$N: %.f$" % N,
            r"$\# misses: %.f$" % (misses),
            r"$ \# spikes: %.f$" % (spikes),
        ]
        return self.meta_text_for_plot(text_l=text_l, text_r=text_r)

    def meta_text_for_plot(self, text_l=[], text_r=[]):
        """
        Optimal for 3 entries lext_l and 2 entries text_r
        Returns a callback function to remove the text from ax
        """
        pos_x1, pos_x2, pos_y = self.get_text_positions()
        if self.is_back:
            text_l.reverse()
            text_r.reverse()
        text1 = self.ax.text(pos_x1, pos_y, "\n".join(text_l))
        text2 = self.ax.text(pos_x2, pos_y, "\n".join(text_r))
        return lambda: (text1.remove(), text2.remove())

    def get_text_positions(self):
        x_lim, y_lim = self.ax.get_xlim(), self.ax.get_ylim()
        pos_y = y_lim[0] + (y_lim[1] - y_lim[0]) * 0.05
        pos_x1 = x_lim[0] + (x_lim[1] - x_lim[0]) * 0.01
        pos_x2 = x_lim[0] + (x_lim[1] - x_lim[0]) * 0.7
        if self.is_back:
            pos_y = y_lim[1] - ((y_lim[1] - y_lim[0]) * 0.35)
        return pos_x1, pos_x2, pos_y

    def remove_extra_lines(self, index=1):
        lines = self.ax.get_lines()
        for line in lines[index:]:
            line.remove()


class Trajectory:

    is_feeding = False

    def __init__(self, marker_char="", write_fig=True):
        self.write_fig = write_fig
        self.fish2camera = get_fish2camera_map(is_feeding=self.is_feeding)
        self.N_fishes = self.fish2camera.shape[0]
        self.fig_front = Figure(is_back=False, marker_char=marker_char)
        self.fig_back = Figure(is_back=True, marker_char=marker_char)

    def reset_data(self):
        pass

    def subplot_function(
        self,
        batch,
        date,
        directory,
        name,
        fish_id,
        time_span="batch: 1,   00:00:00 - 00:30:00",
        is_back=False,
    ):
        F = self.fig_back if is_back else self.fig_front
        F.ax.set_title(time_span, fontsize=10)
        last_frame = batch.FRAME.array[-1]
        if (batch.xpx.array[-1] == -1 and batch.ypx.array[-1] == -1) or (
            batch.xpx.array[-1] == 0 and batch.ypx.array[-1] == 0
        ):  # remove error point, only need to carry it to this point to record the last frame number
            batch.drop(batch.tail(1).index)

        batchxy = pixel_to_cm(batch[["xpx", "ypx"]].to_numpy())
        F.line.set_data(*batchxy.T)
        # draw spikes where datapoints were lost
        # ax.draw_artist(ax.patch)
        # ax.draw_artist(line)
        F.remove_extra_lines(index=1)
        gaps_idx, gaps_select = get_gaps_in_dataframes(batch.FRAME.array)
        for i in gaps_idx:
            F.ax.plot(
                *batchxy.T[:, i : i + 2], "r-", alpha=0.7, solid_capstyle="projecting"
            )

        # metric calculations
        steps = calc_length_of_steps(batchxy)
        spikes, spike_places = num_of_spikes(steps)
        ignore_flags = (
            spike_places | gaps_select
        )  # spike or gap ignore them for the next calculation
        mean, sd = activity_mean_sd(steps, ignore_flags)
        avg_alpha, sum_alpha = avg_and_sum_angles(batchxy)
        N = len(steps)
        misses = last_frame - N
        remove_text = F.meta_text_for_trajectory(
            mean, sd, avg_alpha, sum_alpha, spikes, misses, N
        )

        if self.write_fig:
            F.write_figure(directory, name)
        remove_text()
        return F.fig

    def plots_for_tex(self, fish_ids):
        N = len(self.fish2camera[fish_ids])
        self.reset_data()
        for i, fish_idx in enumerate(fish_ids):
            camera_id, pos = self.fish2camera[fish_idx]
            is_back = pos == BACK
            day_list = get_days_in_order(
                is_feeding=self.is_feeding, camera=camera_id, is_back=is_back
            )
            N_days = len(day_list)
            for j, day in enumerate(day_list):
                sys.stdout.write("\r")
                # write the progress to stdout
                progress = i / N + j / (N * N_days)
                sys.stdout.write(
                    "[%-20s] %d%%" % ("=" * int(20 * progress), 100 * progress)
                )
                sys.stdout.flush()

                keys, day_df = csv_of_the_day(
                    camera_id,
                    day,
                    is_back=is_back,
                    drop_out_of_scope=True,
                    is_feeding=self.is_feeding,
                )
                self.plot_day_camera_fast(
                    day_df, keys, camera_id, day, fish_idx, is_back=is_back
                )

    def plot_day_camera_fast(self, data, keys, camera_id, date, fish_id, is_back):
        position = get_position_string(is_back)
        time_dir = FEEDINGTIME if self.is_feeding else STIME
        data_dir = "{}/{}/{}/{}/{}/{}".format(
            PLOTS_TRAJECTORY, time_dir, BLOCK, position, camera_id, date
        )

        if len(data) == 0:
            return None

        nr_of_frames = 0
        for idx in range(len(data)):
            batch = data[idx]
            up = len(batch.x) - 1
            time_span = "batch: {},    {} - {}".format(
                keys[idx],
                get_time_for_day(date, nr_of_frames),
                get_time_for_day(date, nr_of_frames + batch.FRAME[up]),
            )

            fig = self.subplot_function(
                batch,
                date,
                data_dir,
                keys[idx],
                fish_id,
                time_span=time_span,
                is_back=is_back,
            )  # plot ij.pdf
            nr_of_frames += batch.FRAME[up]
        return fig

        # __set_figure = self.set_figure # copy set_figure
