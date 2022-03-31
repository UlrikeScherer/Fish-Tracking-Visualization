import numpy as np

def rotation(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

def px2cm():
    return 0.02326606

def pixel_to_cm(pixels): 
    """
    @params: pixels (Nx2)
    returns: cm (Nx2)
    """
    R = rotation(np.pi/4)
    t = [0.02326606,0.02326606]# 0.01541363]
    T = np.diag(t)
    trans_cm = np.array([19.86765585, -1.16965425])
    return (pixels@R@T) - trans_cm

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


