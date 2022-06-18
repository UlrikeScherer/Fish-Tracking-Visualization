import pandas as pd
import os
import numpy as np
from time import gmtime, strftime
from src.config import BACK, FRONT, BLOCK, DATA_results
from src.utile import get_days_in_order,get_all_days_of_context, get_camera_pos_keys
from src.visualisation import Trajectory
from src.transformation import pixel_to_cm

class FeedingTrajectory(Trajectory):

    is_feeding = True
    dir_data_feeding = "%s/%s/feeding"%(DATA_results,BLOCK)
    fir_tex_feeding = "tex/files"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.PATCHES = self.get_feeding_patches()
        self.set_feeding_box(is_back=False)
        self.set_feeding_box(is_back=True)
        self.feeding_times = []
        self.visits = []
        self.reset_data()

    def reset_data(self):
        self.feeding_times = [dict() for i in range(self.N_fishes)]
        self.visits = [dict() for i in range(self.N_fishes)]

    def set_feeding_box(self, is_back=False):
        F = self.fig_back if is_back else self.fig_front
        box_line = F.ax.plot([0,0], [0,0], "y--")

    def subplot_function(self, batch, date, directory, name, fish_id, time_span="batch: 1,   00:00:00 - 00:30:00", is_back=False):
        F = self.fig_back if is_back else self.fig_front

        F.ax.set_title(time_span,fontsize=10)

        if batch.x.array[-1] <= -1:
            batch.drop(batch.tail(1).index)

        batchxy = pixel_to_cm(batch[["xpx", "ypx"]].to_numpy())
        F.line.set_data(*batchxy.T)

        feeding_b, box = self.feeding_data(batch, fish_id) # feeding_b: array of data frames that are inside the feeding box. 
        feeding_size = feeding_b.shape[0] # size of the feeding box gives us the time spent in the box in number of frames.
        # The next line identifies the indices of feeding_b array where the fish swims from in to out of the box in the next frame
        index_visits = []
        n_entries = 0
        # a case distinction has to be made: when the are no visits to the feeding box index_visits is empty.
        if feeding_size > 0:
            index_swim_in = np.where(feeding_b.FRAME[1:].array-feeding_b.FRAME[:-1].array != 1)[0]
            index_visits = [0, *(index_swim_in+1), feeding_size-1] # The first visit to the box clearly happens at index 0 of feeding_b and the last visit ends at the last index of feeding_b
            n_entries = len(index_visits)-1 # -1 for the last out index

        fb = pixel_to_cm(feeding_b[["xpx", "ypx"]].to_numpy()).T
        lines = F.ax.get_lines()
        # UPDATE BOX
        box_cm = pixel_to_cm(box)
        lines[1].set_data(*box_cm.T)

        lines = lines[2:]

        for i,l in enumerate(lines):
            if i < n_entries:
                s,e = index_visits[i], index_visits[i+1]
                l.set_data(*fb[:,s:e])
            else:
                l.remove()
        for i in range(len(lines),n_entries):
            s,e = index_visits[i], index_visits[i+1]
            line_feed = F.ax.plot(*fb[:,s:e], "r-", alpha=0.7, solid_capstyle="projecting", markersize=0.2)

        text_l = [" ","#Visits: %s"%(n_entries), r"$\Delta$ Feeding: %s"%(strftime("%M:%S", gmtime(feeding_size/5)))]
        remove_text = F.meta_text_for_plot(text_l=text_l)

        #ax.draw_artist(ax.patch)
        #ax.draw_artist(line)
        self.update_feeding_and_visits(fish_id, date, feeding_size, n_entries)

        if self.write_fig:
            F.write_figure(directory, name)
        remove_text()
        return F.fig

    def update_feeding_and_visits(self,fish_id, date, feeding_size, visits):
        if date not in self.feeding_times[fish_id]:
            self.feeding_times[fish_id][date] = 0
            self.visits[fish_id][date] = 0
        self.feeding_times[fish_id][date] += feeding_size
        self.visits[fish_id][date] += visits

    def feeding_data_to_csv(self):
        fish_names = get_camera_pos_keys(is_feeding=self.is_feeding)
        days = get_all_days_of_context(is_feeding=self.is_feeding)
        df_feeding = pd.DataFrame(columns=[fish_names], index=days)
        df_visits = pd.DataFrame(columns=[fish_names], index=days)
        for i, fn in enumerate(fish_names):
            for d in days:
                if d in self.feeding_times[i]:
                    df_feeding.loc[d, fn]=self.feeding_times[i][d]
                    df_visits.loc[d, fn]=self.visits[i][d]

        os.makedirs(self.dir_data_feeding, exist_ok=True)
        df_feeding.to_csv("%s/%s.csv"%(self.dir_data_feeding, "feeding_times"))
        df_visits.to_csv("%s/%s.csv"%(self.dir_data_feeding, "visits"))

    def feeding_data_to_tex(self):
        text = '''\newcommand\ftlist{}\newcommand\setft[2]{\csdef{ft#1}{#2}}\newcommand\getft[1]{\csuse{ft#1}}'''.replace('\n', '\\n').replace('\f', '\\f')

        for i, (c,p) in enumerate(self.fish2camera):
            days = get_days_in_order(is_feeding=self.is_feeding, camera=c, is_back=p==BACK)
            for d in days:
                if d in self.feeding_times[i]:
                    text += "\setft{%s%s%s}{%s}"%(c,p,d,strftime("%H:%M:%S", gmtime(self.feeding_times[i][d]/5)))
                    text += "\setft{%s%s%sv}{%s}"%(c,p,d,self.visits[i][d])
        text_file = open("%s/%s_feedingtime.tex"%(self.dir_data_feeding, BLOCK), "w")
        text_file.write(text)
        text_file.close()

    def feeding_data(self, data, fish_id):
        return get_feeding_box(data,*self.PATCHES[fish_id])

    def get_feeding_patches(self):
        patches = pd.read_csv("data/feeding_patch_coords.csv", delimiter=";")
        return dict([(i, find_cords(*self.fish2camera[i], patches)) for i in range(self.N_fishes)])

def find_cords(camera_id, position, csv):
    f1= csv["camera_id"]==int(camera_id)
    f2= csv["front_or_back"]==position
    cords = csv[f1 & f2][["TL_x", "TL_y", "TR_x", "TR_y"]]
    if cords.empty:
        raise Exception("camera: %s with position %s is not known"%(camera_id, position))
    return cords.to_numpy()[0]

def get_feeding_cords(data,camera_id, is_back):
    pos = BACK if is_back else FRONT
    patches = pd.read_csv("data/feeding_patch_coords.csv", delimiter=";")
    return get_feeding_box(data,*find_cords(camera_id, pos, patches))

def get_feeding_box(data, TL_x, TL_y, TR_x, TR_y):
    scale = 2
    if abs(TL_x - TR_x) < abs(TL_y - TR_y):
        # FRONT
        p_len = abs(TL_y - TR_y)*scale
        f1 = data["xpx"] > TL_x - p_len
        f2 = data["ypx"] < TL_y + p_len
        f3 = data["ypx"] > TR_y - p_len
        TL_y+=p_len
        TR_y-=p_len
        box = np.array([(TL_x, TL_y), (TL_x-p_len, TL_y), (TR_x-p_len, TR_y), (TR_x, TR_y), (TL_x, TL_y)])
    else:
        # BACK
        p_len = abs(TL_x - TR_x)*scale
        f1 = data["ypx"] > TR_y - p_len
        f2 = data["xpx"] > TL_x - p_len
        f3 = data["xpx"] < TR_x + p_len
        TL_x-=p_len
        TR_x+=p_len
        box = np.array([(TL_x, TL_y), (TL_x, TL_y-p_len), (TR_x, TR_y-p_len), (TR_x, TR_y), (TL_x, TL_y)])

    feeding = data[f1 & f2 & f3]
    return feeding, box
