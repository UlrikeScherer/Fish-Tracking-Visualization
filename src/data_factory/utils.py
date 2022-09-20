import math
import numpy as np
from src.config import ROOT_LOCAL, DIR_CSV_LOCAL

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

# slow function 
""" 
def find_cluster_seqences(clusters, cluster_ids=range(1,6), sw=5*60*5, th=0.9):
    couter = dict(zip(cluster_ids, [sum(clusters[:sw]==cid) for cid in cluster_ids]))
    record = dict()
    i = sw
    while i < clusters.shape[0]:
        couter[clusters[i]] += 1
        couter[clusters[i-sw]] -= 1
        if couter[clusters[i]]/sw > th:
            record[(i-sw,i)]=clusters[i]
            couter = dict(zip(cluster_ids, [sum(clusters[i:i+sw]==cid) for cid in cluster_ids]))
            i+=sw
        else:
            i+=1
    return record
"""

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

def get_cluster_seqences(clusters, cluster_ids=range(1,6), sw=6*60*5, th=0.6):
    records = dict(zip(cluster_ids, [list() for i in range(len(cluster_ids))]))
    for cid in cluster_ids:
        bits = clusters==cid
        n_ones, rs,re = runs_of_ones_array(bits)
        n_zeros = rs[1:] - re[:-1]
        matches = combine_ones_and_zeros(n_ones, n_zeros, th, sw)
        matches.sort(key=lambda x: x["score"], reverse=True)
        records[cid].extend([(rs[m['idx'][0]],re[m['idx'][1]], m['score']) for m in matches[:15]])
    return records

def create_subset_data(k=25):
    import glob, random, shutil
    list_of_files = random.choices(glob.glob(f"{ROOT_LOCAL}/{DIR_CSV_LOCAL}/*/*/*/*.csv"), k=k)

    for file in list_of_files:
        dist = file.replace(DIR_CSV_LOCAL.split("/")[-1],"FE_tracks_subset", 1)
        dist = "/".join(dist.split("/")[:-1])
        os.makedirs(dist ,exist_ok=True)
        shutil.copy(file, dist)


