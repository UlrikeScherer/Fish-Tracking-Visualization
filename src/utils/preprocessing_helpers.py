import numpy as np
import pandas as pd
from src.config import ROOT_LOCAL
from src.utils import get_fish_ids


def update_livehistory_means_and_std(m_s):
    path, filename = ROOT_LOCAL, "DevEx_fingerprint_activity_lifehistory.csv"
    fileupdate = "DevEx_fingerprint_activity_lifehistory_update.csv"
    columns_m_s = ["mean_activity", "std_activity", "fish_id"]
    info_df = pd.read_csv("%s/%s" % (path, filename), delimiter=";")
    block1_m_s = pd.read_csv("%s/%s" % (path, fileupdate), delimiter=";")
    block1_m_s = block1_m_s[block1_m_s["block"] == 1][columns_m_s].to_numpy()
    m_s_df = np.concatenate((m_s, get_fish_ids()[:, :1]), axis=1)
    m_s_df = pd.DataFrame(
        np.concatenate((m_s_df, block1_m_s), axis=0), columns=columns_m_s
    )
    df_update = pd.merge(info_df, m_s_df, on="fish_id", how="left")
    df_update.to_csv("%s/%s" % (path, fileupdate), sep=";")
