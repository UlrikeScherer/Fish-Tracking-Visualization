import glob, json, os
import matplotlib.pyplot as plt
import numpy as np
from src.config import FRONT, BACK, BLOCK, DATA_DIR, BLOCK1, BLOCK2, ROOT_LOCAL
from .utile import get_camera_pos_keys

# AREA CONFIG
AREA_CONFIG = "%s/area_config" % ROOT_LOCAL
# area block 1
area_block1_back = "%s/areas back 10Sep2021" % AREA_CONFIG
area_block1_front = "%s/areas front 10Sep2021" % AREA_CONFIG
# area block 2
area_block2_front_02nov = "%s/areas front 02Nov2021" % AREA_CONFIG
area_block2_back_02nov = "%s/areas back 02Nov2021" % AREA_CONFIG
area_block2_front_21nov = "%s/areas front 21Nov2021" % AREA_CONFIG
area_block2_back_21nov = "%s/areas back 21Nov2021" % AREA_CONFIG

nov21 = "20211121_060000"
nov02 = "20211102_060000"

CALIBRATION_DIST_CM = 83.0

def get_area_functions():  # retruns a function to deliver the area for key and day
    if BLOCK == BLOCK1:
        area = read_area_data_from_json()
        return lambda key, day=None: area[key]
    elif BLOCK == BLOCK2:
        area1 = read_area_data_from_json(day=nov02)
        area2 = read_area_data_from_json(day=nov21)
        return lambda key, day: area1[key] if day < nov21 else area2[key]

def get_calibration_functions():
    if BLOCK == BLOCK1:
        f = open(f"{DATA_DIR}/{BLOCK}_calibration.json", "r")
        calibration = json.load(f)
        return lambda cam, day=None: calibration[cam]
    elif BLOCK == BLOCK2:
        f1 = open(f"{DATA_DIR}/{BLOCK}_calibration{nov02}.json", "r")
        calibration1 = json.load(f1)
        f2 = open(f"{DATA_DIR}/{BLOCK}_calibration{nov21}.json", "r")
        calibration2 = json.load(f2)
        return lambda cam, day: calibration1[cam] if day < nov21 else calibration2[cam]
    else:
        raise Exception("{} is not a valid block".format(BLOCK))  


def read_area_data_from_json(day=None):
    if day is None:
        suffix = ""
    elif day < nov21:
        suffix = nov02
    else:
        suffix = nov21

    with open("{}/{}_area_data{}.json".format(DATA_DIR, BLOCK, suffix), "r") as infile:
        area_data = json.load(infile)
        for k in area_data.keys():
            area_data[k] = np.array(area_data[k])
        return area_data


def get_area_path(
    day=None,
):  # TODO distinguish between different area configs for block2
    if BLOCK == BLOCK1:
        return {BACK: area_block1_back, FRONT: area_block1_front}
    elif BLOCK == BLOCK2:
        if day is None:
            raise Exception("define kwargs day for block2")
        if day == nov02:
            return {BACK: area_block2_back_02nov, FRONT: area_block2_front_02nov}
        elif day == nov21:
            return {BACK: area_block2_back_21nov, FRONT: area_block2_front_21nov}
        else:
            raise Exception("invalide day")
    else:
        raise Exception("something super weird happened")


def get_areas(day=None):
    area_paths = get_area_path(day=day)
    area_data = dict()
    example_dict = {FRONT: np.array([]), BACK: np.array([])}
    for p, path in (area_paths).items():
        files_a = glob.glob("%s/*.csv" % (path), recursive=True)
        if len(files_a) == 0:
            raise Exception("no files found in %s" % path)
        for f in files_a:
            c = os.path.basename(f)[:8]
            if c.isnumeric():
                file_read = open(f, "r")
                key = "%s_%s" % (c, p)
                for line in file_read.readlines():
                    if "Last" in line:
                        poly = [ll.split(",") for ll in line.split("#")[2].split(";")]
                        data_a = np.array(poly).astype(np.float64)
                        if key not in area_data or area_data[key].size > data_a.size:
                            area_data[key] = data_a
                            continue
                if area_data[key].shape[0] == 5 and len(example_dict[p]) == 0:
                    example_dict[p] = area_data[key]

    for k, v in area_data.items():
        if v.shape[0] != 5:
            area_data[k] = update_area(example_dict[k.split("_")[1]], v)
            if area_data[k] is None:
                del area_data[k]

    missing_areas = [c for c in get_camera_pos_keys() if c not in area_data.keys()]
    if len(missing_areas) > 0:
        print("Missing Areas:", missing_areas)
    for m_k in missing_areas:
        area_data[m_k] = example_dict[m_k.split("_")[1]]

    write_area_data_to_json(area_data, suffix=day)
    for k, v in list(area_data.items()):
        plt.plot(v[:, 0], v[:, 1], "-o")
    return area_data


def update_area(example, area):
    indices = []
    for i, p in enumerate(example):
        x = (area[:, 0] - p[0]) ** 2
        y = (area[:, 1] - p[1]) ** 2
        idx = np.argmin(x + y)
        if idx in indices:
            return None
        indices.append(idx)
    return area[indices]


def write_area_data_to_json(area_data, suffix=None):
    if suffix is None:
        suffix = ""
    area_d = dict(zip(area_data.keys(), map(lambda v: v.tolist(), area_data.values())))
    with open("{}/{}_area_data{}.json".format(DATA_DIR, BLOCK, suffix), "w") as outfile:
        json.dump(area_d, outfile, indent=2)

def get_line_starting_with(file, matchstr="Last"):
    file_read = open(file, "r")
    for line in file_read.readlines():
        if matchstr == line[:len(matchstr)]:
            return line
        
def compute_calibrations(day=None):  
    calibration = dict()
    for position,path in get_area_path(day).items():
        files_a = glob.glob("%s/*.csv" % (path), recursive=True)
        if len(files_a) == 0:
            raise Exception("no files found in %s" % (path))
        for file in files_a:
            c = os.path.basename(file)[:8]
            if c.isnumeric():
                if c not in calibration:
                    cal = np.array([list(map(lambda x: int(x), coord.split(","))) if "," in coord else None for coord in get_line_starting_with(file).split("#")[1].split(";")])
                    calibration[c] = CALIBRATION_DIST_CM/np.mean(np.sqrt(np.sum((cal[:2]-cal[2:])**2, axis=1)))
    suffix = day if day else ""       
    with open(f"{DATA_DIR}/{BLOCK}_calibration{suffix}.json", "w") as f:
        json.dump(calibration, f, indent=2)
    return calibration