import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import warnings
from time import gmtime, strftime
import fishproviz.config as config
from fishproviz.metrics.metrics import compute_step_lengths, num_of_spikes
from fishproviz.trajectory.novel_object_shape import ObjectEllipse
from fishproviz.utils import (
    get_days_in_order,
    get_all_days_of_context,
    get_camera_pos_keys,
)
from fishproviz.utils.utile import start_time_of_day_to_seconds, get_seconds_from_time
from .trajectory import Trajectory
from fishproviz.utils.transformation import pixel_to_cm

map_shape = {"ellipse": ObjectEllipse}
FT_DATE, FT_START, FT_END = (
    "day",
    "time_in_start",
    "time_out_stop",
)  # time_in_stop, time_out_start


class SociabilityTrajectory(Trajectory):
    dir_data_object = "%s/%s" % (
        config.RESULTS_PATH,
        config.P_SOCIABILITY,
    )
    dir_tex_object = config.TEX_DIR

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_object_box(is_back=False)
        self.set_object_box(is_back=True)
        self.feeding_times = []
        self.visits = []
        self.num_df_feeding = []
        self.start_end_times = feeding_times_start_end_dict()
        self.reset_data()
        self.ObjectShape = map_shape["ellipse"](sociability=True)
        self.sociability_zones = ['inflow', 'outflow']

    def reset_data(self):
        self.sociability_zones = ['inflow', 'outflow']
        self.data = {}
        for zone in self.sociability_zones:
            self.data[zone] = {}
            self.data[zone]['feeding_times'] = [dict() for i in range(self.N_fishes)]
            self.data[zone]['visits'] = [dict() for i in range(self.N_fishes)]
            self.data[zone]['num_df_feeding'] = [dict() for i in range(self.N_fishes)]

    def set_object_box(self, is_back=False):
        F = self.fig_back if is_back else self.fig_front
        _ = F.ax.plot([0, 0], [0, 0], "y--")

    def get_start_end_index(self, day_key, batch_number):
        if self.start_end_times is None:
            return 0, config.BATCH_SIZE
        (day, track_start) = day_key.split("_")[:2]
        ts_sec = start_time_of_day_to_seconds(track_start)
        (f_start, f_end) = self.start_end_times[day]
        if f_start is None or f_end is None:
            warnings.warn("No start or end time for day %s" % day)
            return 0, config.BATCH_SIZE
        start = int((f_start - ts_sec) * config.FRAMES_PER_SECOND)
        end = int((f_end - ts_sec) * config.FRAMES_PER_SECOND)
        # get start index
        if start < int(batch_number) * config.BATCH_SIZE:
            start_idx = 0
        elif start > (int(batch_number) + 1) * config.BATCH_SIZE:
            start_idx = config.BATCH_SIZE
        else:
            start_idx = start - int(batch_number) * config.BATCH_SIZE
        # get end index
        if end < int(batch_number) * config.BATCH_SIZE:
            end_idx = 0
        elif end > (int(batch_number) + 1) * config.BATCH_SIZE:
            end_idx = config.BATCH_SIZE
        else:
            end_idx = end - int(batch_number) * config.BATCH_SIZE
        return start_idx, end_idx

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

        batchxy = pixel_to_cm(batch[["xpx", "ypx"]].to_numpy(), fish_key=fish_key)
        F.line.set_data(*batchxy.T)
        n_entries_per_zone = []
        feeding_sizes = []
        index_visits = []
        n_entries = 0
        all_lines = []
        for j, zone in enumerate(self.sociability_zones):
            feeding_b, box = self.ObjectShape.contains(
                batch[feeding_filter], fish_key, date, zone
            )  # feeding_b: array of data frames that are inside the feeding box.
            feeding_size = feeding_b.shape[
                0
            ]  # size of the feeding box gives us the time spent in the box in number of frames.
            # The next line identifies the indices of feeding_b array where the fish swims from in to out of the box in the next frame

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

            if j == 0:
                lines = F.ax.get_lines().copy()
            # UPDATE BOX
            box_cm = pixel_to_cm(box, fish_key)
            #lines[1].set_data(*box_cm.T)
            (box_line, ) = F.ax.plot(*box_cm.T, linestyle='--', color='lightgreen')
            all_lines.append(box_line)
            lines = lines[2:]

            for i, l in enumerate(lines):
                if i < n_entries:
                    s, e = index_visits[i], index_visits[i + 1]
                    l.set_data(*fb[:, s:e])
                    all_lines.append(l)
                #else:
                #    l.remove()
            for i in range(len(lines), n_entries):
                s, e = index_visits[i], index_visits[i + 1]

                _ = F.ax.plot(
                    *fb[:, s:e],
                    "r-",
                    alpha=0.7,
                    solid_capstyle="projecting",
                    markersize=0.2,
                )

            # ax.draw_artist(ax.patch)
            # ax.draw_artist(line)
            self.update_feeding_and_visits(
                fish_id, date, feeding_size, n_entries, sum(feeding_filter), zone
            )
            n_entries_per_zone.append(n_entries)
            feeding_sizes.append(feeding_size)

        text_l = [
            f"#Visits {self.sociability_zones[0]}: %s" % (n_entries_per_zone[0]),
            f"#Visits {self.sociability_zones[1]}: %s" % (n_entries_per_zone[1]),
            fr"$\Delta$ {self.sociability_zones[0]}: %s" % (strftime("%M:%S", gmtime(feeding_sizes[0] / 5))),
            fr"$\Delta$ {self.sociability_zones[1]}: %s" % (strftime("%M:%S", gmtime(feeding_sizes[1] / 5))),
        ]

        steps = compute_step_lengths(batchxy)
        spikes, spike_places = num_of_spikes(steps)
        N = batchxy.shape[0]
        text_r = F.meta_text_rhs(N, last_frame - N, spikes)
        remove_text = F.meta_text_for_plot(text_l=text_l, text_r=text_r)

        if self.write_fig:
            F.write_figure(directory, batch_number)
        remove_text()
        for line in F.ax.get_lines()[2:]:
            line.remove()
        return F.fig

    def update_feeding_and_visits(
        self, fish_id, date, feeding_size, visits, num_df_feeding, zone
    ):
        if date not in self.data[zone]['feeding_times'][fish_id]:
            self.data[zone]['feeding_times'][fish_id][date] = 0
            self.data[zone]['visits'][fish_id][date] = 0
            self.data[zone]['num_df_feeding'][fish_id][date] = 0
        self.data[zone]['feeding_times'][fish_id][date] += feeding_size
        self.data[zone]['visits'][fish_id][date] += visits
        self.data[zone]['num_df_feeding'][fish_id][date] += num_df_feeding

    def object_data_to_csv(self):
        fish_names = get_camera_pos_keys()
        days = get_all_days_of_context()
        for zone in self.sociability_zones:
            df_feeding = pd.DataFrame(columns=[fish_names], index=days)
            df_visits = pd.DataFrame(columns=[fish_names], index=days)
            df_num_df_feeding = pd.DataFrame(columns=[fish_names], index=days)
            for i, fn in enumerate(fish_names):
                for d in days:
                    if d in self.data[zone]['feeding_times'][i]:
                        df_feeding.loc[d, fn] = self.data[zone]['feeding_times'][i][d]
                        df_visits.loc[d, fn] = self.data[zone]['visits'][i][d]
                        df_num_df_feeding.loc[d, fn] = self.data[zone]['num_df_feeding'][i][d]

            os.makedirs(self.dir_data_object, exist_ok=True)
            df_feeding.to_csv("%s/%s.csv" % (self.dir_data_object, f"feeding_times_{zone}"))
            df_visits.to_csv("%s/%s.csv" % (self.dir_data_object, f"visits_{zone}"))
            df_num_df_feeding.to_csv(
                "%s/%s.csv" % (self.dir_data_object, f"num_df_feeding_{zone}")
            )

    def object_data_to_tex(self):
        for zone in self.sociability_zones:
            text = """\newcommand\ftlist{}\newcommand\setft[2]{\csdef{ft#1}{#2}}\newcommand\getft[1]{\csuse{ft#1}}""".replace(
                "\n", "\\n"
            ).replace(
                "\f", "\\f"
            )

            for i, (c, p) in enumerate(self.fish2camera):
                days = get_days_in_order(camera=c, is_back=p == config.BACK)
                for d in days:
                    if d in self.data[zone]['feeding_times'][i]:
                        text += "\setft{%s%s%s}{%s}" % (
                            c,
                            p,
                            d,
                            strftime("%H:%M:%S", gmtime(self.data[zone]['feeding_times'][i][d] / 5)),
                        )
                        text += "\setft{%s%s%sv}{%s}" % (c, p, d, self.data[zone]['visits'][i][d])
                        text += "\setft{%s%s%snum}{%s}" % (
                            c,
                            p,
                            d,
                            strftime("%H:%M:%S", gmtime(self.data[zone]['num_df_feeding'][i][d] / 5)),
                        )
            text_file = open(f"%s/feedingtime_{zone}.tex" % (self.dir_tex_object), "w")
            text_file.write(text)
            text_file.close()


def feeding_times_start_end_dict():
    if not os.path.exists(config.SERVER_FEEDING_TIMES_FILE):
        warnings.warn(
            f"File {config.SERVER_FEEDING_TIMES_FILE} not found, thus feeding times will be calculated over all provided batches, if this is not intended please check the path in scripts/env.sh"
        )
        return None
    else:
        ft_df = pd.read_csv(
            config.SERVER_FEEDING_TIMES_FILE,
            usecols=[FT_DATE, FT_START, FT_END],
            sep=config.SERVER_FEEDING_TIMES_SEP,
        )
        ft_df = ft_df[~ft_df[FT_START].isna()]
        start_end = dict(
            [
                (
                    ''.join(d.split("-")),
                    (get_seconds_from_time(s), get_seconds_from_time(e)),
                )
                for (d, s, e) in zip(ft_df[FT_DATE], ft_df[FT_START], ft_df[FT_END])
            ]
        )
        return start_end
