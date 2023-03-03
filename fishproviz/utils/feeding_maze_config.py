
import glob
import json
from collections import defaultdict
import os
from fishproviz.config import MAZE_FILE, CONFIG_DATA, PATH_RECORDINGS
MAZE = "maze"
FP_1 = "FP_1"
FP_2 = "FP_2"


def read_maze_data_from_json(project_path=CONFIG_DATA):
    """Reads the maze data from the json-file and returns a dictionary with the
    data for each fish.
    """
    if not os.path.isfile(f"{project_path}/{MAZE_FILE}"):
        print("No maze_data.json file found in %s" % project_path)
        return read_maze_data_from_server(PATH_RECORDINGS, project_path)
    with open(f"{project_path}/{MAZE_FILE}", "r") as f:
        maze_dict = json.load(f)
    return maze_dict

def read_maze_data_from_server(path_recordings, project_path):
    """Reads the maze data from the server and returns a dictionary with the
    data for each fish.
    """
    if not os.path.isdir(path_recordings):
        raise Exception("Path to recordings does not exist: %s" % path_recordings)
    print("Reading maze data from server!")
    files = glob.glob(path_recordings+"/**/*.annotations.json", recursive=True)
    if len(files) == 0:
        raise Exception("No annotations.json files found in %s" % path_recordings)
    maze_dict = defaultdict(lambda: defaultdict(dict))
    for f in sorted(files):
        f_split = f.split("/")
        cam = f_split[-3]
        day=f_split[-2].split(".")[0]
        with open(f, "r") as fp: 
            jf = json.load(fp)
            for d in jf:
                if "back" in d["comment"].lower():
                    pos_str="back"
                elif "front" in d["comment"].lower():
                    pos_str="front"
                else: 
                    print("missing position in comment: %s for cam: %s"%(d["comment"], cam + " " + day))
                    print(f)
                    continue
                cam_pos="%s_%s"%(cam,pos_str)
                if d["type"]=="ellipse":
                    if MAZE in maze_dict[cam_pos][day]:
                        print("Warning!:", f,d)
                        continue
                    maze_dict[cam_pos][day][MAZE]=d
                elif d["type"]=="label":
                    if FP_1 not in maze_dict[cam_pos][day]:
                        maze_dict[cam_pos][day][FP_1]=d
                    elif FP_2 not in maze_dict[cam_pos][day]:
                        maze_dict[cam_pos][day][FP_2]=d
    with open(f"{project_path}/{MAZE_FILE}", "w") as f:
        json.dump(maze_dict,f)
    return maze_dict