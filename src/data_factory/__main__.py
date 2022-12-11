
import motionmapperpy as mmpy
import hdf5storage
import numpy as np
import os

from .utils import get_days, get_individuals_keys, set_parameters
from .plasticity import cluster_entropy_plot, plot_index_columns
from .processing import get_regions_for_fish_key, load_clusters_concat, load_zVals_concat
from .plotting import plot_transition_rates 
from src.utils import get_camera_names, get_all_days_of_context, get_camera_pos_keys
from src.utils.tank_area_config import get_area_functions

def main_factory(parameters, n_clusters = [5,10]):
    cameras = get_camera_names(is_back=True)
    DAYS = get_days(parameters)
    fish_ids = get_individuals_keys(parameters)
    area_f = get_area_functions()
    for n_c in n_clusters:
        parameters.kmeans = n_c
        wshed_path = '%s/%s/zVals_wShed_groups_%s.mat'%(parameters.projectPath, parameters.method,5)
        if not os.path.exists(wshed_path):
            startsigma=1.0
            mmpy.findWatershedRegions(parameters, minimum_regions=n_c, startsigma=startsigma, pThreshold=[0.33, 0.67], saveplot=True, endident = '*_pcaModes.mat')
        wshedfile = hdf5storage.loadmat(wshed_path)
        get_clusters_func_wshed = lambda fk, d: get_regions_for_fish_key(wshedfile,fk,d)
        for flag in [True, False]:
            cluster_entropy_plot(parameters,get_clusters_func_wshed,fish_ids,parameters.kmeans,
                    name=f"cluster_entropy_wshed",by_the_hour=flag)
        plot_transition_rates(get_regions_for_fish_key(wshedfile), 
                    filename=parameters.projectPath+"/overall_kmeans_%s.pdf"%n_c)
        plot_transition_rates(load_zVals_concat(parameters)["kmeans_clusters"]+1, 
                    filename=parameters.projectPath+"/overall_kmeans_%.pdf"%n_c, cluster_remap=None)

def for_all_cluster_entrophy(parameters, fish_ids):
    get_clusters_func = lambda fk,d: load_clusters_concat(parameters, fk, d)
    for k in parameters.kmeans_list:
        parameters.kmeans=k
        wshedfile = hdf5storage.loadmat('%s/%s/zVals_wShed_groups_%s.mat'%(parameters.projectPath, parameters.method,parameters.kmeans))
        get_clusters_func_wshed = lambda fk, d: get_regions_for_fish_key(wshedfile,fk,d)
        for flag in [True, False]:
            for func, c_type in zip([get_clusters_func_wshed, get_clusters_func],["wshed", "kmeans"]):
                cluster_entropy_plot(parameters,func,fish_ids,parameters.kmeans,
                        name=f"cluster_entropy_{c_type}",by_the_hour=flag, forall=True, fit_degree=2)

if __name__ == "__main__":
    parameters = set_parameters()
    for_all_cluster_entrophy(parameters, get_individuals_keys(parameters))
    #main_factory(parameters)