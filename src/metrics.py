import numpy as np
from src.utile import S_LIMIT
from methods import avg_and_sum_angles # import cython functions for faster for-loops. 

def mean_sd(steps):
    mean = np.mean(steps)
    sd = np.std(steps)
    return mean, sd

def num_of_spikes(steps):
    return np.sum(steps > S_LIMIT)

def calc_length_of_steps(df):
    ysq = (df.y.array[1:] - df.y.array[:-1])**2
    xsq = (df.x.array[1:] - df.x.array[:-1])**2
    c=np.sqrt(ysq + xsq)
    return c

def calc_step_per_frame(df):
    """ This function calculates the eucleadian step length in centimerters per FRAME, this is useful as a speed measument after the removal of erroneous data points."""
    ysq = (df.y.array[1:] - df.y.array[:-1])**2
    xsq = (df.x.array[1:] - df.x.array[:-1])**2
    frame_dist = df.FRAME.array[1:] - df.FRAME.array[:-1]
    c=np.sqrt(ysq + xsq)/frame_dist
    return c

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def determinant(v,w):
    """ Determinant of two vectors. """
    return v[0]*w[1]-v[1]*w[0]

def direction_angle(v,w):
    """ Return the angle between v,w anti clockwise from direction v to m. """
    cos = np.dot(v,w)
    r = np.arccos(np.clip(cos, -1, 1))
    det = determinant(v,w)
    if det < 0: 
        return r
    else:
        return -r
    
def angle(v,w):
    cos = np.dot(v,w)
    return np.arccos(np.clip(cos, -1, 1))

def sum_of_angles(df):
    y = (df.ypx.array[1:] - df.ypx.array[:-1])
    x = (df.xpx.array[1:] - df.xpx.array[:-1])
    if len(x) == 0:
        return 0
    u = unit_vector([x[0],y[0]])
    sum_alpha = 0
    for i in range(1,y.size):
        v = unit_vector([x[i],y[i]])
        if np.any(np.isnan(v)):
            continue
        if np.any(np.isnan(u)):
            u = v
            continue
        alpha = direction_angle(u,v)
        sum_alpha += alpha
        u = v
    return sum_alpha

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
