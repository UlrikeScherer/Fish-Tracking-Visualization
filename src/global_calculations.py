import numpy as np
from src.utile import csv_of_the_day

def global_mean(cams, days):
    sum_means = 0
    num_N=0
    for c in cams:
        for d in days:
            for is_back in [True, False]:
                batches = csv_of_the_day(c, d, is_back, drop_out_of_scope=True)
                for b in batches:
                    steps = calc_length_of_steps(b)
                    sum_steps = steps.sum()
                    if type(sum_steps)!=np.float64:
                        print(m, b, d, c)
                        continue
                    else:
                        sum_means += sum_steps
                        num_N += len(steps)
    return sum_means/(num_N)
    
def global_sd(cams, days, mean):
    sum_sd = 0
    num_N=0
    for c in cams:
        for d in days:
            for is_back in [True, False]:
                batches = csv_of_the_day(c, d, is_back, drop_out_of_scope=True)
                for b in batches:
                    steps = calc_length_of_steps(b)
                    x = np.abs(steps - mean)**2
                    x_sum = x.sum()
                    if type(x_sum)!=np.float64:
                        print(m, b, d, c)
                        continue
                    else:
                        sum_sd += x_sum
                        num_N += len(x)
    return np.sqrt(sum_sd/num_N)