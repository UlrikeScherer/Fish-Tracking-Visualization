import glob, json, os
import matplotlib.pyplot as plt
import numpy as np
from fishproviz.config import FRONT, BACK, CONFIG_DATA, CALIBRATION_DIST_CM,DEFAULT_CALIBRATION, CONFIG_DATA, area_back, area_front
from .utile import get_camera_pos_keys

def get_area_functions():  # retruns a function to deliver the area for key and day
    try: 
        area = read_area_data_from_json()
        return lambda key, day=None: area[key]
    except Exception as e:
        print(e, " program will run without area data")
        return lambda key, day=None: None

def get_calibration_functions():
    calibration_file = f"{CONFIG_DATA}/calibration.json"
    if not os.path.exists(calibration_file):
        try: 
            calibration = compute_calibrations()
        except Exception as e:
            print(e, "will use default calibration of %s px/cm" % ( DEFAULT_CALIBRATION))
            return lambda cam: DEFAULT_CALIBRATION
    else:
        f = open(calibration_file, "r")
        calibration = json.load(f)
    return lambda cam: calibration[cam.split("_")[0]]

def read_area_data_from_json():
    if not os.path.exists("{}/area_data.json".format(CONFIG_DATA)):
        return get_areas()
    with open("{}/area_data.json".format(CONFIG_DATA), "r") as infile:
        area_data = json.load(infile)
        for k in area_data.keys():
            area_data[k] = np.array(area_data[k])
        return area_data

def get_areas():
    area_data = dict()
    example_dict = {FRONT: np.array([]), BACK: np.array([])}
    for p, path in zip([FRONT,BACK], [area_front, area_back]):
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

    write_area_data_to_json(area_data)
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


def write_area_data_to_json(area_data):
    area_d = dict(zip(area_data.keys(), map(lambda v: v.tolist(), area_data.values())))
    with open("{}/area_data.json".format(CONFIG_DATA), "w") as outfile:
        json.dump(area_d, outfile, indent=2)

def get_line_starting_with(file, matchstr="Last"):
    file_read = open(file, "r")
    for line in file_read.readlines():
        if matchstr == line[:len(matchstr)]:
            return line
        
def compute_calibrations():  
    calibration = dict()
    for path in [area_front, area_back]:
        files_a = glob.glob("%s/*.csv" % (path), recursive=True)
        if len(files_a) == 0:
            raise Exception("No files found in the directory: %s" % (path))
        for file in files_a:
            c = os.path.basename(file)[:8]
            if c.isnumeric():
                if c not in calibration:
                    cal = np.array([list(map(lambda x: int(x), coord.split(","))) if "," in coord else None for coord in get_line_starting_with(file).split("#")[1].split(";")])
                    calibration[c] = CALIBRATION_DIST_CM/np.mean(np.sqrt(np.sum((cal[:2]-cal[2:])**2, axis=1)))      
    with open(f"{CONFIG_DATA}/calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)
    return calibration
