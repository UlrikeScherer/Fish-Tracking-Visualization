import argparse
import os, re, glob
import time
from numpy import any
from fishproviz.config import DIR_CSV_LOCAL
from fishproviz.utils import utile

def check_foldersystem(path, n_files=15, delete=False):
    '''
    asserts correctness of nested directories in the current working directory,
    implicitly ignores `_no_fish` folders, expects 4 files for the first day,
    deletes duplicates if `delete`-flag is set
    
    params:
        path: str
        n_files: int
        delete: boolean int (0 == False, 1 == True)
    returns: 
        log-message as str
    '''
    LOG_msg = ["For path: %s" % path]

    for camera_dir in [name for name in os.listdir(path) if len(name) == 8 and name.isnumeric()]:
        day_dir_list = [
            name for name in os.listdir("{}/{}".format(path, camera_dir)) if name[:8].isnumeric()
        ]
        days_unique = set()
        for day_dir in day_dir_list:
            working_dir = f'{path}/{camera_dir}/{day_dir}'
            if day_dir[:8] not in days_unique:
                days_unique.add(day_dir[:8])
            else:
                LOG_msg.append(
                    f"Duplicate day {day_dir} in folder: {path}/{camera_dir}/"
                )
            files = glob.glob(f'{working_dir}/*.csv')
            files = [os.path.basename(f) for f in files]
            wrote_folder = False

            # ignore no fish folders
            if "_no_fish" in day_dir:
                if len(files) > 0:
                    LOG_msg.append("Folders with no_fish suffix should be empty!")
                else:
                    continue
            # normal case
            else:
                if len(files) != n_files:
                    wrote_folder = True
                    LOG_msg.append(
                        f"In folder {working_dir} the number of csv files is unequal the expected number {n_files}, it is {len(files)} instead"
                    )
                msg, duplicate_f_list, _correct_f = utile.filter_files(camera_dir, day_dir, files, n_files)

            # if the are any complains add them to the LOG list
            if len(msg) > 0:
                if not wrote_folder:
                    LOG_msg.append(
                        f"In folder {working_dir} has the correct number of csv files: "
                    )
                # if the delete flag is set and they are duplicates ==> remove them
                if delete and len(duplicate_f_list) > 0:
                    LOG_msg.append("----DELETING DUPLICATES----")
                    for duplicate_f in duplicate_f_list:
                        os.remove(f'{working_dir}/{duplicate_f}')
                LOG_msg.extend(msg)
    return LOG_msg


def main(delete=False, n_files=15, path=DIR_CSV_LOCAL):
    """
    @params:
    This main has two optional arguments:
    delete: if set to 1, the duplicates will be deleted
    n_files: defines how many files to expect in each folder, default is 15, for feeding use 8, for a log file that is more clear.
    """
    # select the path here and use the %s to replace "front" or "back"
    position = ["front", "back"]
    if not os.path.exists(path):
        raise ValueError("Path %s does not exist" % path)
    else:
        PATHS = [
            "%s/%s" % (path, dir_p)
            for dir_p in os.listdir(path)
            if dir_p[0] != "." and any([p in dir_p for p in position])
        ]

    if len(PATHS) < 2:
        raise ValueError("Path %s does not contain enough folders" % path)
    # TODO: log immediately into log-file
    LOG = list()
    for p in PATHS:  # validating files for front and back position
        LOG.append(p.upper() + "-" * 100 + "\n")
        LOG.extend(check_foldersystem(p, n_files=n_files, delete=delete))
    f = open("log-path-validation.txt", "w")
    f.writelines("\n".join(LOG))
    print("LOG: see log-path-validation.txt, %d errors and warnings found" % len(LOG))
    return len(LOG) == 0


if __name__ == "__main__":
    tstart = time.time()
    parser = argparse.ArgumentParser(
        prog="python3 path_validation.py",
        description="This program validate the file structure below the directory %s or if provided with a PATH argument the corresponding directory."
        % DIR_CSV_LOCAL,
        epilog="Example of use: python3 path_validation.py --delete --n_files 15 --path /Volumes/Extreme_SSD/FE_tracks",
    )
    parser.add_argument(
        "--delete",
        action="store_false",
        help="If set, the duplicates will be deleted.",
    )
    parser.add_argument(
        "--n_files",
        type=int,
        default=15,
        help="Number of files to expect in each folder, default is 15, for feeding use 8, for a log file that is cleaner.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=DIR_CSV_LOCAL,
        help="Path to the directory that contains the folders front and back, default is %s." % DIR_CSV_LOCAL,
    )
    args = parser.parse_args()
    main(**args.__dict__)
    tend = time.time()
    print("Running time:", tend - tstart, "sec.")
