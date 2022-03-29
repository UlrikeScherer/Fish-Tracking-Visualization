import os, sys, re, glob

from src.utile import *

def check_foldersystem(path):
    LOG_msg = ["For path: %s"%path]
    for c in [name for name in os.listdir(path) if len(name)==8 and name.isnumeric()]:
        for d in [name for name in os.listdir("{}/{}".format(path,c)) if name[:8].isnumeric()]:
            if "{}.{}".format(d[:15],c) != d:
                LOG_msg.append("Day %s is not in the correct format"%d)
            files = glob.glob("{}/{}/{}/*.csv".format(path, c, d))
            files = [os.path.basename(f) for f in files]
            wrote_folder = False
            if "_1550" in d: 
                if len(files) != 4:
                    wrote_folder = True
                    LOG_msg.append("In folder %s the number of csv files is unequal the expected number 4, it is %d instead"%(
                    "{}/{}/{}".format(path, c, d), len(files))
                              )
            elif len(files) != 15: 
                wrote_folder = True
                LOG_msg.append("In folder %s the number of csv files is unequal the expected number 15, it is %d instead"%(
                    "{}/{}/{}".format(path, c, d), len(files))
                              )
            if not wrote_folder:
                LOG_msg.append("In folder %s has the correct number of csv files: "%"{}/{}/{}".format(path, c, d))
            files.sort()
            i = 0

            for f in files: 
                pattern = re.compile("{}_{}.{}_{:06d}_\d*-\d*-\d*T\d*_\d*_\d*_\d*.csv".format(c,d[:15],c,i))
                if pattern.match(f) is None:
                    LOG_msg.append("files: %s has corrupted name or is duplicate."%(f))
                else: i+=1
                    
    return LOG_msg

def main():
    path = '/Volumes/data/loopbio_data/FE_(fingerprint_experiment)_SepDec2021/FE_tracks_retracked/FE_tracks_060000_block1_retracked'
    back = 'FE_060000_tracks_block1_back_retracked'
    front = '/FE_060000_tracks_block1_front_retracked'
    paths_to_check = ["%s/%s"%(path, back),"%s/%s"%(path, front)]
    LOG = list()
    for p in paths_to_check:
        LOG.extend(check_foldersystem(p))
    f = open("log-path-validation.txt", "w")
    f.writelines("\n".join(LOG))

if __name__ == '__main__':
    main()