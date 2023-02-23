from src.config import BLOCK1, BLOCK2
import numpy as np
from src.data_factory.utils import get_days, set_parameters, get_individuals_keys
from src.data_factory.processing import load_trajectory_data
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


def repeatability_lmm(data, groups="id_cat", formula="step ~ (1 | id_cat) + block_cat"):
    lmm = smf.mixedlm(formula, data, groups=data[groups])
    lmm_fit = lmm.fit()
    v_within = np.var([lmm_fit.random_effects[i][0] for i in lmm_fit.random_effects.keys()])
    repeatability = 1 - (lmm_fit.scale /( lmm_fit.scale + v_within))
    #print(lmm_fit.summary(),"v_g: ", lmm_fit.scale,"v_r: ", v_within, "rep:", repeatability)
    return repeatability, lmm_fit

def repeatability(step_list):
    weights = np.array(list(map(len,step_list)))
    weights = weights/weights.sum()
    step_std =np.array(list(map(np.var,step_list)))
    step_mean = np.array(list(map(np.mean,step_list)))
    V_w = step_std.mean() #weighted_avg_and_std(step_std, weights)[0]
    #V_t = np.concatenate(step_list).var()
    V_g = step_mean.var() #weighted_avg_and_std(step_mean, weights)[1]
    R=V_g/(V_g+V_w)
    return R #,V_w,V_g

def run_repeatability():
    parameters = set_parameters()
    days_b1 = get_days(parameters, prefix=BLOCK1)
    days_b2 = get_days(parameters, prefix=BLOCK2)
    fish_keys = get_individuals_keys(parameters)
    start = 6 * 5 * 60**2
    end = 14 * 5 * 60**2
    step = 60**2 *5 # 1 hour
    rr = np.zeros((max(len(days_b1), len(days_b2)), 8+1))
    for i,(d1,d2) in list(enumerate(zip(days_b1, days_b2))):
        data1 = load_trajectory_data(parameters=parameters,fk="",day=d1)
        data2 = load_trajectory_data(parameters=parameters,fk="",day=d2)
        datadf = pd.concat((df_table(data1, BLOCK1, fish_keys), df_table(data2, BLOCK2, fish_keys)))
        for j,s in enumerate(range(start, end, step)):
            flt_time = (datadf["time"] > s) & (datadf["time"] < s+step)
            rr[i,j] = repeatability_lmm(datadf[flt_time])
        rr[i,8] = repeatability_lmm(datadf)
        #data = data1+data2
        #step_list = [data[i]["projections"][:,0] for i in range(len(data))]
        #rr[i] = repeatability(step_list)
    pd.DataFrame(rr, columns=["h%s"%r for r in range(0,9)]).to_csv(parameters.projectPath+"/repeatability_lmm_by_h.csv",sep=";")
    return rr

def df_table(data,block, fish_keys):
    return pd.concat([pd.DataFrame({"step":d["projections"][:,0],"time":d["df_time_index"].flatten()}) for d in data],keys=[fish_keys.index(block+"_"+d["fish_key"][0][0]) for d in data]).reset_index()


def get_melted_table(data_avg_step):
    melted = pd.melt(data_avg_step, id_vars=["block1","block2","minutes"],value_vars=data_avg_step.columns[3:], var_name="id", value_name="step")
    split_cols = melted['id'].str.split('_', expand=True)
    split_cols.columns = ['block', 'cam_id', 'pos']
    split_cols['id_cat'] = melted['id'].astype('category').cat.codes
    #combine the melted dataframe with the split columns
    result = pd.concat([melted, split_cols], axis=1)
    result['block_cat'] = result['block'].astype('category').cat.codes
    return result.dropna()

def get_repeatability_dxdy(parameters, data_avg_step):
    result = get_melted_table(data_avg_step)
    days_b1 = get_days(parameters=parameters,prefix=BLOCK1)
    days_b2 = get_days(parameters=parameters,prefix=BLOCK2)
    n_days = len(days_b1)
    rep_M = np.zeros((n_days, n_days))
    for i in range(n_days):
        b1d1,b2d1 = days_b1[i],days_b2[i]
        for j in range(i,n_days):
            b1d2, b2d2 = days_b1[j],days_b2[j]
            copy_results = result.query(
                "block1 == @b1d1 or block1==@b1d2 or block2 == @b2d1 or block2==@b2d2"
            ).copy().dropna()
            rep_M[i,j] = repeatability_lmm(copy_results, group="id_cat")
    return rep_M

def plot_repeatability(df_r):
    fig, ax = plt.subplots(figsize=(7,3.5))
    x = np.arange(8)
    data = df_r.to_numpy()[:,1:]
    for i in range(data.shape[0]):
        ax.scatter((x+(i*8)), data[i,:8])
    ax.plot(range(4,data.shape[0]*8,8), data[:,8], label="daily")
    ax.set_ylabel("repeatability")
    ax.set_xlabel("hours")
    ax.legend()
    return fig

if __name__ == "__main__":
    rr = run_repeatability()
    print(rr)