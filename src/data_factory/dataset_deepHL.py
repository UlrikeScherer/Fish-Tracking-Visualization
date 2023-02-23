import os
import numpy as np
import pandas as pd
from src.data_factory.processing import load_trajectory_data_concat
from src.data_factory.utils import get_days, get_individuals_keys, set_parameters, split_into_batches
from src.config import BACK
from src.utils.transformation import px2cm

def dataset_prep(parameters, fish_ids, batch_size_minutes=5):
    """
    This function prepares a dataset for deepHL model. It takes the data from the fish_ids and days and saves the data
    in the dataset folder.
    :param parameters: parameters dictionary
    :param fish_ids: list of fish ids only use 2 ids 
    :param batch_size_minutes: size of the batch in minutes
    """
    for fk in fish_ids[:2]:
        os.makedirs(f"{parameters.projectPath}/datasetDH/{fk}",exist_ok=True)
        days = get_days(parameters, prefix=fk)
        for i, day in enumerate(days[:4]):
            f1 = load_trajectory_data_concat(parameters, fk, day)
            if f1 is None: 
                continue
            data = np.concatenate((px2cm(f1["positions"]),f1["projections"][:,2][:,np.newaxis]),axis=1)
            times, datas = split_into_batches(f1['df_time_index'], data, batch_size_minutes)
            for j,(d,t) in enumerate(zip(datas, times)):
                f1_df = pd.DataFrame(d, columns=["x", "y","dist"], index=t)
                f1_df.index.name = '# time'
                f1_df.to_csv(f"{parameters.projectPath}/datasetDH/{fk}/{i}{j}.csv", index=True)


if __name__ == "__main__":
    parameters = set_parameters()
    fish_ids = get_individuals_keys(parameters)
    dataset_prep(parameters, [fish_ids[i] for i in [2,5]], batch_size_minutes=5)
