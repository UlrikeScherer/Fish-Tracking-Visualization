import glob, json
import matplotlib.pyplot as plt
import numpy as np
from src.utile import FRONT, BACK, ROOT, BLOCK, DATA_DIR

def read_area_data_from_json():
    with open("{}/{}_area_data.json".format(DATA_DIR,BLOCK), "r") as infile:
        area_data = json.load(infile)
        for k in area_data.keys():
            area_data[k]=np.array(area_data[k])
        return area_data

def get_areas():
    files_a = glob.glob("%s/area*%s/*.csv"%(ROOT_LOCAL, BLOCK), recursive=True)
    area_data = dict()
    example_dict = {FRONT:[],BACK:[]}
    for f in files_a:
        file_read = open(f, "r")
        c = os.path.basename(f)[:8]
        if BACK in f:
            p = BACK
        else: 
            p = FRONT
        key = "%s_%s"%(c,p)
        for l in file_read.readlines():
            if "Last" in l:
                poly = [ll.split(",") for ll in l.split("#")[2].split(";")]
                area_data[key]=np.array(poly).astype(np.float64)
        if area_data[key].shape[0]==5 and len(example_dict[p])==0:
            example_dict[p]=area_data[key]
    
    for k,v in area_data.items():
        if v.shape[0]!=5:
            area_data[k] = update_area(example_dict[k.split("_")[1]], v)
            
    missing_areas = [c for c in get_camera_pos_keys() if c not in area_data.keys()]
    if len(missing_areas)>0:
        print("Missing Areas:", missing_areas)
    for m_k in missing_areas:
        area_data[m_k]=example_dict[m_k.split("_")[1]]
        
    write_area_data_to_json(area_data)
    for k,v in list(area_data.items()):
        plt.plot(v[:,0], v[:,1], "-o")
    return area_data

def update_area(example, area):
    new_area = np.zeros((5,2))
    for i,p in enumerate(example):
        x = (area[:,0]-p[0])**2
        y = (area[:,1]-p[1])**2
        idx = np.argmin(x+y)
        new_area[i]=area[idx]
    return new_area

def write_area_data_to_json(area_data):
    area_d = dict(zip(area_data.keys(), map(lambda v: v.tolist(),area_data.values())))
    with open("{}/{}_area_data.json".format(DATA_DIR,BLOCK), "w") as outfile:
        json.dump(area_d, outfile, indent=2)

