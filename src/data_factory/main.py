
import motionmapperpy as mmpy
import hdf5storage
import numpy as np
import os
from .plasticity import cluster_entropy_plot
from .processing import get_regions_for_fish_key, load_zVals_concat
from .plotting import plot_transition_rates 
from src.utils import get_camera_names, get_all_days_of_context, get_camera_pos_keys
from src.utils.tank_area_config import get_area_functions

def main_factory(parameters, n_clusters = [5,10]):
    cameras = get_camera_names(is_back=True)
    days = get_all_days_of_context()
    fish_keys = get_camera_pos_keys()
    area_f = get_area_functions()
    for n_c in n_clusters:
        wshed_path = '%s/%s/zVals_wShed_groups_%s.mat'%(parameters.projectPath, parameters.method,5)
        if not os.path.exists(wshed_path):
            startsigma=1.0
            mmpy.findWatershedRegions(parameters, minimum_regions=n_c, startsigma=startsigma, pThreshold=[0.33, 0.67], saveplot=True, endident = '*_pcaModes.mat')
        wshedfile = hdf5storage.loadmat(wshed_path)
        
        cluster_entropy_plot(fish_keys,n_c,name="cluster_entropy_for_days_data_kmean_%s"%n_c, 
                     by_the_hour=True, kmean_clusters=True)
        plot_transition_rates(get_regions_for_fish_key(wshedfile), 
                    filename=parameters.projectPath+"/overall_kmeans_%s.pdf"%n_c)
        plot_transition_rates(load_zVals_concat(parameters)["kmeans_clusters"]+1, 
                    filename=parameters.projectPath+"/overall_kmeans_%.pdf"%n_c, cluster_remap=None)
        