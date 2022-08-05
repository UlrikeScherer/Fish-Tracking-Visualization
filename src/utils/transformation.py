import numpy as np

def normalize_origin_of_compartment(data, area, is_back):
    if is_back:
        origin1 = area[0,0], area[1,1]
        new_area = (area-origin1)
        origin2 = new_area[2,0],new_area[3,1]
        new_area = -new_area+origin2
        data = -data+origin1+origin2
    else:
        origin1 = area[1,0], area[0,1]
        new_area = area - origin1
        data = data - origin1
    return data, new_area

def rotation(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def px2cm(a):
    return a * 0.02326606


def pixel_to_cm(pixels):
    """
    @params: pixels (Nx2)
    returns: cm (Nx2)
    """
    R = rotation(np.pi / 4)
    t = [0.02326606, 0.02326606]  # 0.01541363]
    T = np.diag(t)
    trans_cm = np.array([19.86765585, -1.16965425])
    return (pixels @ R @ T) - trans_cm
