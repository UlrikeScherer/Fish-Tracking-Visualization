import numpy as np
from src.utile import csv_of_the_day, get_fish2camera_map, S_LIMIT, BACK
from src.metrics import calc_step_per_frame

def global_mean(fish_ids, days):
    s,n = global_activity(fish_ids, days)
    return s.sum()/n.sum()

def global_mean_by_day(fish_ids, days):
    s,n = global_activity(fish_ids, days)

def global_activity(fish_ids, days):
    sum_means = 0
    num_N=0
    fish2camera=get_fish2camera_map()
    distance_fish_day = [list() for i in range(len(fish_ids))]
    N_steps = [list() for i in range(len(fish_ids))]
    for i,(c, pos) in enumerate(fish2camera[fish_ids]):
        is_back = BACK == pos
        for j, d in enumerate(days):
            batches = csv_of_the_day(c, d, is_back, drop_out_of_scope=True)
            if len(batches)==0:
                continue
            sum_steps = 0
            n_steps = 0
            for b in batches:
                steps = calc_step_per_frame(b)
                steps = steps[steps < 3*S_LIMIT]
                sum_steps += steps.sum()
                n_steps += len(steps)
            if n_steps>0:
                distance_fish_day[i].append(sum_steps)
                N_steps[i].append(n_steps)
            
    return distance_fish_day, N_steps
    
def global_sd(fish_ids, days, mean):
    sum_sd = 0
    num_N=0
    fish2camera=get_fish2camera_map()
    for i,(c, pos) in enumerate(fish2camera[fish_ids]):
        is_back = BACK == pos
        for j, d in enumerate(days):
                batches = csv_of_the_day(c, d, is_back, drop_out_of_scope=True)
                for b in batches:
                    steps = calc_step_per_frame(b)
                    x = np.abs(steps - mean)**2
                    x_sum = x.sum()
                    if type(x_sum)!=np.float64:
                        print(m, b, d, c)
                        continue
                    else:
                        sum_sd += x_sum
                        num_N += len(x)
    return np.sqrt(sum_sd/num_N)
