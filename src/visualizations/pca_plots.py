import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from scipy.spatial.distance import cdist

from src.clustering.clustering import get_metrics_for_traces, get_results_filepath

def bar_plot_pca(pca, trace_size):
    y = pca.explained_variance_ratio_
    plt.bar(["PC%d" % (i + 1) for i in range(len(y))], y)
    plt.savefig(
        get_results_filepath(trace_size, "PCA_explained_variance_ratio"),
        bbox_inches="tight",
    )
    plt.close()


def bar_plot_pca_loadings(pca, trace_size, number_of_components=4, names=None):
    if names is None:
        names = get_metrics_for_traces()[1]
    print(names)
    df_pca = pd.DataFrame(dict(zip(names, abs(pca.components_.T))))
    pca_np = df_pca.to_numpy()
    fig, axs = plt.subplots(
        nrows=1,
        ncols=number_of_components,
        figsize=(number_of_components * 4, 1 * 4),
        sharey="row",
    )
    for i in range(number_of_components):
        axs[i].bar(df_pca.columns, pca_np[i], color=cm.get_cmap("tab20")(range(9)))
        axs[i].set_title("PC%d ratio: %.2f" % (i + 1, pca.explained_variance_ratio_[i]))
        plt.setp(axs[i].get_xticklabels(), rotation=45, ha="right")

    fig.savefig(get_results_filepath(trace_size, "PCA_Loadings"), bbox_inches="tight")
    plt.close(fig)


def elbow_method_kmeans(X, max_k, fig_name=None, model=KMeans):
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1, max_k + 1)
    for k in K:
        # Building and fitting the model
        kmeanModel = model(n_clusters=k).fit(X)
        kmeanModel.fit(X)

        distortions.append(
            sum(np.min(cdist(X, kmeanModel.cluster_centers_, "euclidean"), axis=1))
            / X.shape[0]
        )
        inertias.append(kmeanModel.inertia_)

        mapping1[k] = (
            sum(np.min(cdist(X, kmeanModel.cluster_centers_, "euclidean"), axis=1))
            / X.shape[0]
        )
        mapping2[k] = kmeanModel.inertia_
    # print and show
    for key, val in mapping1.items():
        print(f"{key} : {val}")
    plt.plot(K, distortions, "bx-")
    plt.xlabel("Values of K")
    plt.ylabel("Distortion")
    plt.title("The Elbow Method using Distortion")
    plt.legend()
    if fig_name is None:
        plt.show()
    else:
        plt.savefig(fig_name)
        plt.close()