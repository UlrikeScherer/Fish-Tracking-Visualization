import os, sys, re, glob
import time


def check_foldersystem(path, n_files=15, delete=0):
    LOG_msg = ["For path: %s" % path]
    for c in [name for name in os.listdir(path) if len(name) == 8 and name.isnumeric()]:
        day_dirs = [
            name for name in os.listdir("{}/{}".format(path, c)) if name[:8].isnumeric()
        ]
        days_unique = set()
        for d in day_dirs:
            if d[:8] not in days_unique:
                days_unique.add(d[:8])
            else:
                LOG_msg.append("Duplicate day {} in folder: {}/{}/".format(d, path, c))
            # if "{}.{}".format(d[:15],c) != d:
            #    LOG_msg.append("Day %s is not in the correct format"%d) # this message can be ignored
            files = glob.glob("{}/{}/{}/*.csv".format(path, c, d))
            files = [os.path.basename(f) for f in files]
            wrote_folder = False

            # ignore no fish folders
            if "_no_fish" in d:
                if len(files) > 0:
                    LOG_msg.append("Folders with no_fish suffix should be empty!")
                else:
                    continue
            # expect only 4 files in the folder for the first day
            if "_1550" in d:
                if len(files) != 4:
                    wrote_folder = True
                    LOG_msg.append(
                        "In folder %s the number of csv files is unequal the expected number 4, it is %d instead"
                        % ("{}/{}/{}".format(path, c, d), len(files))
                    )
                msg, duplicate_f = filter_files(c, d, files, 4)

            # normal case
            else:
                if len(files) != n_files:
                    wrote_folder = True
                    LOG_msg.append(
                        "In folder %s the number of csv files is unequal the expected number %d, it is %d instead"
                        % ("{}/{}/{}".format(path, c, d), n_files, len(files))
                    )
                msg, duplicate_f, correct_f = filter_files(c, d, files, n_files)

            # if the are any complains add the to the LOG list
            if len(msg) > 0:
                if not wrote_folder:
                    LOG_msg.append(
                        "In folder %s has the correct number of csv files: "
                        % "{}/{}/{}".format(path, c, d)
                    )
                # if the delete flag is set and they are duplicates ==> remove them
                if delete and len(duplicate_f) > 0:
                    LOG_msg.append("----DELETING DUPLICATES----")
                    for df in duplicate_f:
                        os.remove("{}/{}/{}/{}".format(path, c, d, df))

                LOG_msg.extend(msg)

    return LOG_msg


def filter_files(c, d, files, n_files):
    """
    @params:
    c: camera_id
    d: folder name of a day
    files: list of files that are to be filtered
    n_files: number of files to expect.
    @Returns: LOG, duplicate_f, correct_f
    LOG: a list of LOG messages
    duplicate_f: a list of all duplicates occurring
    correct_f: dict of the correct files for keys i in 0,...,n_files-1
    """
    LOG = []
    missing_numbers = []
    duplicate_f = []
    correct_f = dict()
    for i in range(n_files):
        key_i = "{:06d}".format(i)
        pattern = re.compile(
            ".*{}_{}.{}_{}_\d*-\d*-\d*T\d*_\d*_\d*_\d*.csv".format(c, d[:15], c, key_i)
        )
        i_f = [f for f in files if pattern.match(f) is not None]
        if len(i_f) > 1:
            i_f.sort()
            duplicate_f.extend(i_f[:-1])
            correct_f[key_i] = i_f[-1]
        elif len(i_f) == 0:
            missing_numbers.append(key_i)
        else:
            correct_f[key_i] = i_f[-1]

    pattern_general = re.compile(
        ".*{}_{}.{}_\d*_\d*-\d*-\d*T\d*_\d*_\d*_\d*.csv".format(c, d[:15], c)
    )
    corrupted_f = [f for f in files if pattern_general.match(f) is None]

    if len(missing_numbers) > 0:
        LOG.append(
            "The following files are missing: \n \t\t{}".format(
                " ".join(missing_numbers)
            )
        )
    if len(duplicate_f) > 0:
        LOG.append(
            "The following files are duplicates: \n\t{}".format(
                "\n\t".join(duplicate_f)
            )
        )
    if len(corrupted_f) > 0:
        LOG.append(
            "The following file names are corrupted, maybe wrong folder: \n\t{}".format(
                "\n\t".join(corrupted_f)
            )
        )
    return LOG, duplicate_f, correct_f


def main(delete=0, n_files=15):
    """
    This main has two optional arguments
    delete: 0 or 1 indecation if files that are duplicates should be removed
    n_files: defines how many files to expect in each folder, default is 15, for feeding use 8, for a log file that is more clear.
    """
    # past your path here as path1 with %s indicating front or back
    # path1 = '/Volumes/Extreme_SSD/FE_tracks_140000/FE_tracks_140000_block2/FE_tracks_140000_block2_%s'
    # path1 = "/Volumes/Extreme_SSD/FE_tracks_140000/FE_tracks_140000_block2/FE_tracks_140000_block2_%s"
    path1 = "/Volumes/Extreme_SSD/FE_tracks/FE_block2_060000_%s_final"
    # path1 = '/Volumes/data/loopbio_data/FE_(fingerprint_experiment)_SepDec2021/FE_tracks_retracked/FE_tracks_060000_block1_retracked/FE_060000_tracks_block1_%s_retracked'

    # select the path here and use the %s to replace "front" or "back"
    PATH = path1
    position = ["front", "back"]
    LOG = list()
    for p in position:  # validating files for front and back position
        LOG.append(p.upper() + "-" * 100 + "\n")
        LOG.extend(check_foldersystem(PATH % p, n_files=n_files, delete=delete))
    f = open("log-path-validation.txt", "w")
    f.writelines("\n".join(LOG))


if __name__ == "__main__":
    tstart = time.time()
    main(**dict((arg.split("=")[0], int(arg.split("=")[1])) for arg in sys.argv[1:]))
    tend = time.time()
    print("Running time:", tend - tstart, "sec.")
