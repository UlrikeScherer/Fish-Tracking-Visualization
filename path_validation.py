import argparse
import os, glob
import time
from numpy import any
from fishproviz.config import DIR_CSV_LOCAL
from fishproviz.utils import utile, logger

Logger = logger.create_logger(
    logger_name="path_validation",
    log_level_stream=20,
    log_level_file=10,
)


def check_foldersystem(path, n_files=15, delete=False):
    """
    asserts correctness of nested directories in the current working directory,
    implicitly ignores `_no_fish` folders, expects 4 files for the first day,
    deletes duplicates if `delete`-flag is set

    params:
        path: str
        n_files: int
        delete: boolean int (0 == False, 1 == True)
    """
    Logger.debug(f"For path: {path}")

    for camera_dir in [name for name in os.listdir(path) if len(name) == 8 and name.isnumeric()]:
        day_dir_list = [name for name in os.listdir("{}/{}".format(path, camera_dir)) if name[:8].isnumeric()]
        days_unique = set()
        for day_dir in day_dir_list:
            working_dir = f"{path}/{camera_dir}/{day_dir}"
            if day_dir[:8] not in days_unique:
                days_unique.add(day_dir[:8])
            else:
                Logger.debug(f"Duplicate day {day_dir} in folder: {path}/{camera_dir}/")
            files = sorted(glob.glob(f"{working_dir}/*.csv"))

            # ignore no fish folders
            if "_no_fish" in day_dir:
                if len(files) > 0:
                    Logger.debug("Folders with no_fish suffix should be empty!")
                continue
            check_day_folder(working_dir, camera_dir, day_dir.split(".")[0], files, n_files, delete)


def check_day_folder(working_dir: str, camera: str, day: str, files: list[str], n_files: int, delete: bool) -> None:
    """Logs file count, unmatched names, duplicate keys and missing keys of one day folder; with `delete`, keeps the last-sorted file per duplicate key."""
    if len(files) != n_files:
        Logger.debug(f"In folder {working_dir} the number of csv files is unequal the expected number {n_files}, it is {len(files)} instead")
    by_key, unmatched = utile.group_batch_files(camera, day, files)
    missing = ["{:06d}".format(i) for i in range(n_files) if "{:06d}".format(i) not in by_key]
    if missing:
        Logger.debug(f"In folder {working_dir} the following batch keys are missing: {' '.join(missing)}")
    if unmatched:
        Logger.debug(f"In folder {working_dir} the following file names match no batch name, maybe wrong folder:\n\t" + "\n\t".join(unmatched))
    duplicates = [f for key_files in by_key.values() if len(key_files) > 1 for f in sorted(key_files)[:-1]]
    if duplicates:
        Logger.debug(f"In folder {working_dir} the following files are duplicates of a later file with the same batch key:\n\t" + "\n\t".join(duplicates))
        if delete:
            Logger.debug("----DELETING DUPLICATES----")
            for duplicate_f in duplicates:
                os.remove(duplicate_f)


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
        PATHS = ["%s/%s" % (path, dir_p) for dir_p in os.listdir(path) if dir_p[0] != "." and any([p in dir_p for p in position])]

    if len(PATHS) < 2:
        raise ValueError("Path %s does not contain enough folders" % path)
    for p in PATHS:  # validating files for front and back position
        Logger.debug(p.upper() + "-" * 100 + "\n")

        check_foldersystem(p, n_files=n_files, delete=delete)
    Logger.info(f"LOG: see log-file {Logger.filepath}, {Logger.debug.counter} errors and warnings found.")


if __name__ == "__main__":
    tstart = time.time()
    parser = argparse.ArgumentParser(
        prog="python3 path_validation.py",
        description="This program validate the file structure below the directory %s or if provided with a PATH argument the corresponding directory." % DIR_CSV_LOCAL,
        epilog="Example of use: python3 path_validation.py --delete --n_files 15 --path /Volumes/Extreme_SSD/FE_tracks",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
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
    Logger.info("Starting")
    main(**args.__dict__)
    tend = time.time()
    Logger.info(f"Running time: {tend - tstart} sec.")
