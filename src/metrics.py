import numpy as np
from src.utile import S_LIMIT

def mean_sd(steps):
    mean = np.mean(steps)
    sd = np.std(steps)
    return mean, sd

def num_of_spikes(steps):
    return np.where(steps>S_LIMIT)[0].size

def calc_length_of_steps(df):
    ysq = (df.y.array[1:] - df.y.array[:-1])**2
    xsq = (df.x.array[1:] - df.x.array[:-1])**2
    c=np.sqrt(ysq + xsq)
    return c

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.array([np.nan, np.nan])
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
    for i in range(2,y.size):
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
    pos_y = y_lim[0] + (y_lim[1] - y_lim[0]) * 0.1 
    pos_x1 = x_lim[0] + (x_lim[1] - x_lim[0]) * 0.03
    pos_x2 = x_lim[0] + (x_lim[1] - x_lim[0]) * 0.78
    if is_back: 
        pos_y = y_lim[1] - ((y_lim[1] - y_lim[0]) * 0.25)
    steps = calc_length_of_steps(batch)
    sum_alpha = sum_of_angles(batch)
    mean, sd = mean_sd(steps)
    spiks = num_of_spikes(steps)
    N = len(steps)
    text1 = ax.text(pos_x1,pos_y, r"$\sum \alpha_i : %.f$"%sum_alpha + "\n" + r"$\mu + \sigma: %.2f + %.2f$"%(mean, sd))
    text2 = ax.text(pos_x2,pos_y,r"$N: {}$".format(N) + "\n" + r"$ \# spiks: {}$".format(spiks))
    return lambda: (text1.remove(), text2.remove())
