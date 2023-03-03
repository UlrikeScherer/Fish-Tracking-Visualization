import pandas as pd
import calendar
import locale
locale.setlocale(locale.LC_TIME, 'C')
block1_remove = ["block1_23442333_back", "block1_23520264_front", "block2_23520266_front"]

def filter_ids(row, fid):
    #print(row)
    block = str(int(row[0]))
    c_p = str(row[1])
    return f"block{block}" in fid and c_p in fid

def date2dateid(datestr):
    ds = datestr.split(" ")
    day = int(ds[0][:2])
    month = list(calendar.month_abbr).index(ds[1])
    year = int("20"+ds[2][:2])
    return "%s%02d%02d"%(year, month, day)

def get_excluded_days(fish_ids):
    missing = pd.read_csv("data/FE_missing_trajectories_overview.csv", sep=";")
    missing_days = missing[(missing[missing.columns[1]]>0) & 
                       ((missing[missing.columns[0]]=="1") | (missing[missing.columns[0]]=="2"))]
    mdays = missing_days
    exclude = dict()
    for row in mdays.iterrows():
        flt_ids=list(filter(lambda fid: filter_ids(row[1], fid),fish_ids))
        if len(flt_ids)>0:
            exclude["%s_%s"%(flt_ids[0],date2dateid(row[1]["day"]))]=["%06d"%i for i in range(15) if (row[1]["track_%02d"%i] == 1)]
    return exclude