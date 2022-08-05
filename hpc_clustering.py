from cuml.manifold import TSNE
from cuml import PCA, KMeans
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
import numpy as np
import time, sys
from scipy.spatial.distance import cdist

# local imports
from src.config import N_FISHES
from src.utils import get_all_days_of_context, get_camera_pos_keys
from src.clustering.transitions_cluster import (
    cluster_distribution_over_days,
    plot_transitions_individuality_development, 
)
from src.clustering.clustering import (
    clustering,
    get_results_filepath,
    load_traces,
    calculate_traces,
    normalize_data_metrics,
    traces_as_numpy,
    get_metrics_for_traces,
    plot_lines_for_cluster,
    plot_components,
    single_plot_components,
    fish_individuality_tsne,
    fish_development_tsne,
)
from src.visualizations.pca_plots import bar_plot_pca, bar_plot_pca_loadings, elbow_method_kmeans


def execute_clustering(trace_size, *n_clusters):
    try:
        traces_all, nSs = load_traces(trace_size)
        print("traces loaded for size: %s" % trace_size)
    except FileNotFoundError:
        print("Calculates all traces ...")
        days = get_all_days_of_context()
        traces_all, nSs = calculate_traces(
            list(range(N_FISHES)), days, trace_size, write_to_file=True
        )

    pca = PCA()
    traces_np = normalize_data_metrics(traces_as_numpy(traces_all))
    pca_traces = pca.fit_transform(traces_np)
    elbow_method_kmeans(
        pca_traces, 14, get_results_filepath(trace_size, "Elbow_Method"), model=KMeans
    )
    bar_plot_pca(pca, trace_size)
    bar_plot_pca_loadings(pca, trace_size)
    tsne_model = init_tsne_model(perplexity=30, N=pca_traces.shape[0])
    X_embedded = tsne_model.fit_transform(pca_traces)
    for n_c in n_clusters:
        KM = KMeans(n_clusters=n_c, random_state=12)
        clusters = clustering(pca_traces, n_c, model=KM, rating_feature=traces_np[:, 0])
        plot_lines_for_cluster(
            traces_np,
            nSs,
            clusters,
            n_c,
            trace_size,
            limit=30,
            fig_name="cluster_characteristics_%d" % (n_c),
        )
        plot_components(
            pca_traces,
            X_embedded,
            clusters,
            file_name=get_results_filepath(trace_size, "pca_tsne_%d" % (n_c)),
        )
        single_plot_components(
            X_embedded,
            clusters,
            file_name=get_results_filepath(trace_size, "tsne_%d" % (n_c)),
        )

    # INDIVIDUALITY AND DEVELOPMENT
    fish_keys = get_camera_pos_keys()
    days = get_all_days_of_context()
    fish_individuality_tsne(
        fish_keys, X_embedded, traces_all, clusters, n_c, trace_size
    )
    cluster_distribution_over_days(traces_all, clusters, n_c, days, trace_size)
    for fk in fish_keys:
        fish_development_tsne(
            fk, days[6::7], X_embedded, traces_all, clusters, n_c, trace_size
        )
        cluster_distribution_over_days(
            traces_all, clusters, n_c, days, trace_size, fish_key=fk
        )

    # TRANSITIONS
    plot_transitions_individuality_development(
        fish_keys, traces_all, X_embedded, clusters, n_c, trace_size
    )


def init_tsne_model(perplexity, N, **kwargs):
    skl_kw = dict(
        n_components=2,
        perplexity=perplexity,
        n_neighbors=max(102 - 0.0012 * N, 30),
        learning_rate_method=None,
        early_exaggeration=(N > 10000) * 12 + 12.0,
        exaggeration_iter=500,
        learning_rate=max(N / 3, 200),
        n_iter=3000,
        n_iter_without_progress=300,
        min_grad_norm=1e-07,
        metric="euclidean",
        init="random",
        verbose=0,
        random_state=2,  # None,
        method="barnes_hut",  # 'fft'
        angle=0.5,
        square_distances="legacy",
    )
    skl_kw.update(kwargs)
    tsne_model = TSNE(**skl_kw)
    print(tsne_model.get_params())
    return tsne_model


if __name__ == "__main__":
    tstart = time.time()
    execute_clustering(*[int(arg) for arg in sys.argv[1:]])
    tend = time.time()
    print("Running time:", tend - tstart, "sec.")
