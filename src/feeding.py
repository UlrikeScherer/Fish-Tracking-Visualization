import pandas as pd
import os
import numpy as np
from time import gmtime, strftime
from src.utile import BACK, FRONT, get_days_in_order, get_fish_ids, BLOCK
from src.metrics import DATA_results
from src.visualisation import Trajectory

class FeedingTrajectory(Trajectory):

    is_feeding = True
    dir_data_feeding = "%s/%s/feeding"%(DATA_results,BLOCK)

    def __init__(self, y_max = 54, **kwargs):
        super().__init__(y_max=y_max, **kwargs)
        self.PATCHES = self.get_feeding_patches()
        self.set_feeding_box(is_back=False)
        self.set_feeding_box(is_back=True)
        self.feeding_times = []
        self.visits = []

    def reset_data(self):
        self.feeding_times = [dict() for i in range(self.N_fishes)]
        self.visits = [dict() for i in range(self.N_fishes)]

    def set_feeding_box(self, is_back=False):
        (fig, ax, line) = self.fig_obj_back if is_back else self.fig_obj_front
        box_line = ax.plot([0,0], [0,0], "y--")
        return (fig, ax, line)

    def subplot_function(self, batch, date, directory, name, fish_id, time_span="batch: 1,   00:00:00 - 00:30:00", is_back=False):
        (fig, ax, line) = self.fig_obj_back if is_back else self.fig_obj_front
        ax.set_title(time_span,fontsize=10)

        if batch.x.array[-1] <= -1:
            batch.drop(batch.tail(1).index)
        batchxy = pixel_to_cm(batch[["xpx", "ypx"]].to_numpy().T).T
        line.set_data(*batchxy)

        feeding_b, box = self.feeding_data(batch, fish_id)
        feeding_size = feeding_b.shape[0]
        index_swim_in = np.where(feeding_b.FRAME[1:].array-feeding_b.FRAME[:-1].array != 1)[0]
        index_visits = [0, *(index_swim_in+1), feeding_size-1]
        entries = len(index_visits)
        fb = pixel_to_cm(feeding_b[["xpx", "ypx"]].to_numpy().T).T   
        lines = ax.get_lines()
        # UPDATE BOX
        box_cm = pixel_to_cm(box.T)
        lines[1].set_data(*box_cm.T)

        lines = lines[2:]

        for i,l in enumerate(lines):
            if i < entries-1:
                s,e = index_visits[i], index_visits[i+1]
                l.set_data(*fb[:,s:e])
            else:
                l.remove()
        for i in range(len(lines),entries-1):
            s,e = index_visits[i], index_visits[i+1]
            line_feed = ax.plot(*fb[:,s:e], "r-", alpha=0.7, solid_capstyle="projecting", markersize=0.2)
        
        text_l = [" ","#Visits: %s"%(entries-1), r"$\Delta$ Feeding: %s"%(strftime("%M:%S", gmtime(feeding_size/5)))]
        remove_text = self.meta_text_for_plot(ax, text_l=text_l, is_back=is_back)
        
        #ax.draw_artist(ax.patch)
        #ax.draw_artist(line)
        self.update_feeding_and_visits(fish_id, date, feeding_size, entries-1)

        if self.write_fig:
            self.write_figure(fig, directory, name)
        remove_text()
        return fig

    def update_feeding_and_visits(self,fish_id, date, feeding_size, visits):
        if date not in self.feeding_times[fish_id]: 
            self.feeding_times[fish_id][date] = 0
            self.visits[fish_id][date] = 0
        self.feeding_times[fish_id][date] += feeding_size
        self.visits[fish_id][date] += visits

    def feeding_data_to_csv(self):
        fish_names = get_fish_ids()
        days = get_days_in_order(is_feeding=self.is_feeding)
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

        days = get_days_in_order(is_feeding=self.is_feeding)

        for i, (c,p) in enumerate(self.fish2camera):
            for d in days:
                if d in self.feeding_times[i]:
                    text += "\setft{%s%s%s}{%s}"%(c,p,d,strftime("%H:%M:%S", gmtime(self.feeding_times[i][d]/5)))
                    text += "\setft{%s%s%sv}{%s}"%(c,p,d,self.visits[i][d])
        text_file = open("tex/%s_feedingtime.tex"%BLOCK, "w")
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

def rotation(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

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

def pixel_to_cm(pixels):
    R = rotation(-np.pi/4)
    t = [0.02326606,0.02326606]# 0.01541363]
    T = np.diag(t)
    trans_cm = np.array([19.86765585, -1.16965425])
    return (T@R@pixels).T - trans_cm

def pixel_to_cm2(patch, batch):
    px, cm = batch[["xpx", "ypx"]].to_numpy(), batch[[ "x", "y"]].to_numpy()
    i_start = 0
    i_end = 0
    for i in range(i_start,len(cm)):
        if np.all(np.abs(cm[i]-cm[i_start])>10): 
            i_end = i
            break
    # ---------------
    R, t,S = find_rot_tras(px, cm)
    return (R@patch).T + t

def find_rot_tras(px, cm):
    """
    @params: px, cm: np.array 2x2 row-wise
    returns: R rotation-translation matrix to map from pixels to centimeter coordinates. 
    """
    n = px.shape[0]
    px_dist = abs(px[1]-px[0])
    cm_dist = abs(cm[-1]-cm[0])
    fraq = cm_dist/px_dist
    #px = px * fraq
    px_center = px.sum(axis=0)*(1/px.shape[0])
    cm_center = cm.sum(axis=0)*(1/cm.shape[0])
    
    H = (px-px_center).T@(cm-cm_center)
    U, S, V = np.linalg.svd(H)
    R = V@U.T
    px_t = (R@px.T).T
    px_dist_t = abs(px_t[-1]-px_t[0])
    T = np.diag((cm_dist/px_dist_t))
    if np.linalg.det(R) < 0:
        U,S,V = np.linalg.svd(R)    # multiply 3rd column of V by -1
        V[:,-1]=V[:,-1]*-1
        R = V @ U.T
        
    trans = cm_center - T@(R@px_center)
    return T@R, trans, S


