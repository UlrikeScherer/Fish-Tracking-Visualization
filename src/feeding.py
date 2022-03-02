import pandas as pd
import os
import numpy as np
from time import gmtime, strftime
from src.utile import fish2camera
from src.visualisation import meta_text_for_plot

def find_cords(camera_id, position, csv):
    f1= csv["camera_id"]==int(camera_id)
    f2= csv["front_or_back"]==position
    cords = csv[f1 & f2][["TL_x", "TL_y", "TR_x", "TR_y"]]
    if cords.empty:
        raise Exception("camera: %s with position %s is not known"%(camera_id, position))
    return cords.to_numpy()[0]

def get_feeding_patches():
    patches = pd.read_csv("data/feeding_patch_coords.csv", delimiter=";")
    return dict([(i, find_cords(*fish2camera[i], patches)) for i in range(len(fish2camera))])

def rotation(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    
def feeding_data(data, fish_id):
    p = get_feeding_patches()
    TL_x, TL_y, TR_x, TR_y = p[fish_id]
    scale = 2
    
    if abs(TL_x - TR_x) < abs(TL_y - TR_y):
        # FRONT 
        p_len = abs(TL_y - TR_y)*scale
        f1 = data["xpx"] > TL_x - p_len
        f2 = data["ypx"] < TL_y + p_len
        f3 = data["ypx"] > TR_y - p_len
        TL_y+=p_len
        TR_y-=p_len
        box = np.array([(TL_x, TL_y), (TL_x-p_len, TL_y), (TR_x-p_len, TR_y), (TR_x, TR_y)])
    else: 
        # BACK 
        p_len = abs(TL_x - TR_x)*scale
        f1 = data["ypx"] > TR_y - p_len
        f2 = data["xpx"] > TL_x - p_len
        f3 = data["xpx"] < TR_x + p_len
        TL_x-=p_len
        TR_x+=p_len
        box = np.array([(TL_x, TL_y), (TL_x, TL_y-p_len), (TR_x, TR_y-p_len), (TR_x, TR_y)])
        
    feeding = data[f1 & f2 & f3]
    return feeding, box

def subplot_feeding(fig_obj, batch, fish_id, date, directory, name, time_span="batch: 1,   00:00:00 - 00:30:00", is_back=False):
    (fig, ax, line) = fig_obj
    #low, up = 0, len(batch.x)-1
    ax.set_title(time_span,fontsize=10)

    if batch.x.array[-1] <= -1:
        batch.drop(batch.tail(1).index)
    batchxy = pixel_to_cm(batch[["xpx", "ypx"]].to_numpy().T).T
    #line.set_data(batch.x.array, batch.y.array)
    line.set_data(*batchxy)
    feeding_b, box = feeding_data(batch, fish_id)
    feeding_size = feeding_b.size
    index_swim_in = np.where(feeding_b.FRAME[1:].array-feeding_b.FRAME[:-1].array != 1)[0]
    index_visits = [0, *index_swim_in, feeding_size-1]
    entries = len(index_swim_in)
    fb = pixel_to_cm(feeding_b[["xpx", "ypx"]].to_numpy().T).T   
    
    for i in range(entries-1):
        s,e = index_swim_in[i], index_swim_in[i+1]
        line_feed = ax.plot(*fb, "r-o", alpha=0.7, solid_capstyle="projecting", markersize=0.2)
    
    box_cm = pixel_to_cm(box.T)
    print(box,box.T, box_cm)
    box_line = ax.plot(*box_cm.T, "y--")
    
    text_l = [" ","#Visits: %s"%(entries+1), r"$\Delta$ Feeding: %s"%(strftime("%H:%M:%S", gmtime(feeding_size/5)))]
    remove_text = meta_text_for_plot(ax, text_l=text_l, is_back=is_back)
    
    #ax.draw_artist(ax.patch)
    #ax.draw_artist(line)
    
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    fig.savefig("{}/{}.pdf".format(directory,name),bbox_inches='tight')
    remove_text()
    return fig

def pixel_to_cm(pixels):
    R = rotation(-np.pi/4)
    T = np.array([[0.02326606, 0.],[0.,0.01541363]])
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

    
