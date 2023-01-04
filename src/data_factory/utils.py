import glob
import math
import os
import numpy as np
from src.config import DIR_CSV_LOCAL, DIR_CSV_LOCAL, projectPath
import motionmapperpy as mmpy

def pointsInCircum(r,n=100):
    return [(math.cos(2*math.pi/n*x)*r,math.sin(2*math.pi/n*x)*r) for x in range(0,n+1)]

def runs_of_ones_array(bits):
    # make sure all runs of ones are well-bounded
    bounded = np.hstack(([0], bits, [0]))
    # get 1 at run starts and -1 at run ends
    difs = np.diff(bounded)
    run_starts, = np.where(difs > 0)
    run_ends, = np.where(difs < 0)
    return run_ends - run_starts, run_starts, run_ends

def combine_ones_and_zeros(ones, zeros, th, size):
    block = 0
    hits = 0
    records=list()
    j,i = 0,0
    while i < ones.shape[0]:
        if block >= size:
            if (hits/block)>=th:
                records.append({'idx':(j,i), 'score':(hits/block)})
                block,hits,j = 0,0,i
            else:
                block -= (ones[j] + zeros[j])
                hits -= ones[j]
                j+=1
        if block == 0:
            block, hits = ones[i], ones[i]
            i+=1
        elif block < size:
            block += (ones[i] + zeros[i-1])
            hits += ones[i]
            i+=1
    return records

def get_cluster_sequences(clusters, cluster_ids=range(1,6), sw=6*60*5, th=0.6):
    records = dict(zip(cluster_ids, [list() for i in range(len(cluster_ids))]))
    for cid in cluster_ids:
        bits = clusters==cid
        n_ones, rs,re = runs_of_ones_array(bits)
        n_zeros = rs[1:] - re[:-1]
        matches = combine_ones_and_zeros(n_ones, n_zeros, th, sw)
        matches.sort(key=lambda x: x["score"], reverse=True)
        results = [(rs[m['idx'][0]],re[m['idx'][1]], m['score']) for m in matches]
        records[cid].extend(results)
    return records

def create_subset_data(k=25):
    import glob, random, shutil, os
    list_of_files = random.choices(glob.glob(f"{DIR_CSV_LOCAL}/{DIR_CSV_LOCAL}/*/*/*/*.csv"), k=k)

    for file in list_of_files:
        dist = file.replace(DIR_CSV_LOCAL.split("/")[-1],"FE_tracks_subset", 1)
        dist = "/".join(dist.split("/")[:-1])
        os.makedirs(dist ,exist_ok=True)
        shutil.copy(file, dist)

def get_individuals_keys(parameters, block=""):
    files = glob.glob(parameters.projectPath+f"/Projections/{block}*_pcaModes.mat")
    return sorted(list(set(map(lambda f: "_".join(f.split("/")[-1].split("_")[:3]),files))))

def get_days(parameters, prefix=""):
    files = glob.glob(parameters.projectPath+f"/Projections/{prefix}*_pcaModes.mat")
    return sorted(list(set(map(lambda f: "_".join(f.split("/")[-1].split("_")[3:5]),files))))

def set_parameters(parameters=None): 
    parameters = mmpy.setRunParameters(parameters)
    parameters.pcaModes = 3
    parameters.samplingFreq = 5
    parameters.maxF = 2.5
    parameters.minF = 0.01
    parameters.omega0 = 5
    parameters.numProcessors = 16
    parameters.method="UMAP"
    parameters.kmeans = 10
    parameters.kmeans_list = [5, 7, 10, 20, 50, 100]
    parameters.projectPath = projectPath
    os.makedirs(parameters.projectPath,exist_ok=True)
    mmpy.createProjectDirectory(parameters.projectPath)
    return parameters