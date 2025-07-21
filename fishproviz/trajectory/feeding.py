import pandas as pd
import os
import numpy as np
from time import gmtime, strftime
import fishproviz.config as config
from fishproviz.metrics.metrics import compute_step_lengths, num_of_spikes
from fishproviz.trajectory.feeding_shape import FeedingEllipse, FeedingPatch
from fishproviz.utils import (
    get_days_in_order,
    get_all_days_of_context,
    get_camera_pos_keys,
)
from .trajectory import ExperimentalTrajectory
from fishproviz.utils.transformation import pixel_to_cm

map_shape = {"patch": FeedingPatch, "ellipse": FeedingEllipse}


class FeedingTrajectory(ExperimentalTrajectory):
    is_feeding = True
    dir_data_feeding = "%s/%s" % (
        config.RESULTS_PATH,
        config.P_FEEDING,
    )
    dir_tex_feeding = config.TEX_DIR

    def __init__(self, shape=config.FEEDING_SHAPE, **kwargs):
        super().__init__(**kwargs)
        self.set_feeding_box(is_back=False)
        self.set_feeding_box(is_back=True)
        self.feeding_times = []
        self.visits = []
        self.num_df_feeding = []
        self.reset_data()
        self.FeedingShape = map_shape[shape]()

    def reset_data(self):
        self.feeding_times = [dict() for i in range(self.N_fishes)]
        self.visits = [dict() for i in range(self.N_fishes)]
        self.num_df_feeding = [dict() for i in range(self.N_fishes)]

    def set_feeding_box(self, is_back=False):
        F = self.fig_back if is_back else self.fig_front
        _ = F.ax.plot([0, 0], [0, 0], "y--")

    def subplot_function(
        self,
        batch,
        date,
        directory,
        batch_number,
        fish_id,
        time_span="batch: 1,   00:00:00 - 00:30:00",
        is_back=False,
    ):
        F = self.fig_back if is_back else self.fig_front

        start_idx, end_idx = self.get_start_end_index(date, batch_number)
        F.ax.set_title(time_span, fontsize=10)
        last_frame = batch.FRAME.array[-1]
        if batch.x.array[-1] <= -1:
            batch.drop(batch.tail(1).index)

        feeding_filter = batch.FRAME.between(start_idx, end_idx)
        fish_key = "%s_%s" % tuple(self.fish2camera[fish_id])

        batchxy = pixel_to_cm(batch[feeding_filter][["xpx", "ypx"]].to_numpy(), fish_key=fish_key)
        F.line.set_data(*batchxy.T)

        feeding_b, box = self.FeedingShape.contains(
            batch[feeding_filter], fish_key, date
        )  # feeding_b: array of data frames that are inside the feeding box.
        feeding_size = feeding_b.shape[
            0
        ]  # size of the feeding box gives us the time spent in the box in number of frames.
        # The next line identifies the indices of feeding_b array where the fish swims from in to out of the box in the next frame
        index_visits = []
        n_entries = 0
        # a case distinction has to be made: when the are no visits to the feeding box index_visits is empty.
        if feeding_size > 0:
            index_swim_in = np.where(
                feeding_b.FRAME[1:].array - feeding_b.FRAME[:-1].array != 1
            )[0]
            index_visits = [
                0,
                *(index_swim_in + 1),
                feeding_size - 1,
            ]  # The first visit to the box clearly happens at index 0 of feeding_b and the last visit ends at the last index of feeding_b
            n_entries = len(index_visits) - 1  # -1 for the last out index

        fb = pixel_to_cm(feeding_b[["xpx", "ypx"]].to_numpy(), fish_key=fish_key).T
        lines = F.ax.get_lines()
        # UPDATE BOX
        box_cm = pixel_to_cm(box, fish_key)
        lines[1].set_data(*box_cm.T)

        lines = lines[2:]

        for i, l in enumerate(lines):
            if i < n_entries:
                s, e = index_visits[i], index_visits[i + 1]
                l.set_data(*fb[:, s:e])
            else:
                l.remove()
        for i in range(len(lines), n_entries):
            s, e = index_visits[i], index_visits[i + 1]
            _ = F.ax.plot(
                *fb[:, s:e],
                "r-",
                alpha=0.7,
                solid_capstyle="projecting",
                markersize=0.2,
            )

        text_l = [
            " ",
            "#Visits: %s" % (n_entries),
            r"$\Delta$ Feeding: %s" % (strftime("%M:%S", gmtime(feeding_size / 5))),
        ]
        steps = compute_step_lengths(batchxy)
        spikes, spike_places = num_of_spikes(steps)
        N = batchxy.shape[0]
        text_r = F.meta_text_rhs(N, last_frame - N, spikes)
        remove_text = F.meta_text_for_plot(text_l=text_l, text_r=text_r)

        # ax.draw_artist(ax.patch)
        # ax.draw_artist(line)
        self.update_feeding_and_visits(
            fish_id, date, feeding_size, n_entries, sum(feeding_filter)
        )

        if self.write_fig:
            F.write_figure(directory, batch_number)
        remove_text()
        return F.fig

    def update_feeding_and_visits(
        self, fish_id, date, feeding_size, visits, num_df_feeding
    ):
        if date not in self.feeding_times[fish_id]:
            self.feeding_times[fish_id][date] = 0
            self.visits[fish_id][date] = 0
            self.num_df_feeding[fish_id][date] = 0
        self.feeding_times[fish_id][date] += feeding_size
        self.visits[fish_id][date] += visits
        self.num_df_feeding[fish_id][date] += num_df_feeding

    def feeding_data_to_csv(self):
        fish_names = get_camera_pos_keys()
        days = get_all_days_of_context()
        df_feeding = pd.DataFrame(columns=[fish_names], index=days)
        df_visits = pd.DataFrame(columns=[fish_names], index=days)
        df_num_df_feeding = pd.DataFrame(columns=[fish_names], index=days)
        for i, fn in enumerate(fish_names):
            for d in days:
                if d in self.feeding_times[i]:
                    df_feeding.loc[d, fn] = self.feeding_times[i][d]
                    df_visits.loc[d, fn] = self.visits[i][d]
                    df_num_df_feeding.loc[d, fn] = self.num_df_feeding[i][d]

        os.makedirs(self.dir_data_feeding, exist_ok=True)
        df_feeding.to_csv("%s/%s.csv" % (self.dir_data_feeding, "feeding_times"))
        df_visits.to_csv("%s/%s.csv" % (self.dir_data_feeding, "visits"))
        df_num_df_feeding.to_csv(
            "%s/%s.csv" % (self.dir_data_feeding, "num_df_feeding")
        )

    def feeding_data_to_tex(self):
        text = """\newcommand\ftlist{}\newcommand\setft[2]{\csdef{ft#1}{#2}}\newcommand\getft[1]{\csuse{ft#1}}""".replace(
            "\n", "\\n"
        ).replace(
            "\f", "\\f"
        )

        for i, (c, p) in enumerate(self.fish2camera):
            days = get_days_in_order(camera=c, is_back=p == config.BACK)
            for d in days:
                if d in self.feeding_times[i]:
                    text += "\setft{%s%s%s}{%s}" % (
                        c,
                        p,
                        d,
                        strftime("%H:%M:%S", gmtime(self.feeding_times[i][d] / 5)),
                    )
                    text += "\setft{%s%s%sv}{%s}" % (c, p, d, self.visits[i][d])
                    text += "\setft{%s%s%snum}{%s}" % (
                        c,
                        p,
                        d,
                        strftime("%H:%M:%S", gmtime(self.num_df_feeding[i][d] / 5)),
                    )
        text_file = open("%s/feedingtime.tex" % (self.dir_tex_feeding), "w")
        text_file.write(text)
        text_file.close()
