from cuml.manifold import TSNE
from cuml import PCA, KMeans
from matplotlib import cm
from src.config import N_FISHES
from src.utile import get_all_days_of_context
import numpy as np
import time, sys
from scipy.spatial.distance import cdist
from src.clustering import *

def execute_clustering(trace_size, n_clusters):
    try:
        traces_all, nSs = load_traces(trace_size)
        print("traces loaded for size: %s"%trace_size)
    except:
        print("Calculates all traces ...") 
        days = get_all_days_of_context()
        traces_all, nSs = calculate_traces(list(range(N_FISHES)), days, trace_size, write_to_file=True)

    pca = PCA()
    traces_np = normalize_data_metrics(traces_as_numpy(traces_all))
    pca_traces = pca.fit_transform(traces_np)
    KM = KMeans(n_clusters=n_clusters)
    clusters = KM.fit_predict(pca_traces)
    elbow_method_kmeans(pca_traces, 14, get_results_filepath(trace_size,"Elbow_Method"))
    plot_lines_for_cluster(traces_np, nSs, clusters, n_clusters, trace_size, limit=30, fig_name="cluster_characteristics_%d"%(n_clusters))
    bar_plot_pca(pca, trace_size)
    bar_plot_pca_loadings(pca, trace_size)
    
    tsne_model = init_tsne_model(perplexity=30,N = pca_traces.shape[0])
    X_embedded = tsne_model.fit_transform(pca_traces)
    plot_components(pca_traces, X_embedded, clusters, file_name=get_results_filepath(trace_size, "pca_tsne_%d"%(n_clusters)))
        
def bar_plot_pca(pca, trace_size):
    y = pca.explained_variance_ratio_
    plt.bar(["PC%d"%(i+1) for i in range(len(y))],y)
    plt.savefig(get_results_filepath(trace_size, "PCA_explained_variance_ratio"), bbox_inches = "tight")
    plt.close()

def bar_plot_pca_loadings(pca,trace_size,number_of_components=4):
    names = get_metrics_for_traces()[1]
    print(names)
    df_pca = pd.DataFrame(dict(zip(names, abs(pca.components_.T))))
    pca_np = df_pca.to_numpy()
    fig, axs = plt.subplots(nrows=1,ncols=number_of_components, figsize=(number_of_components*4,1*4), sharey="row")
    for i in range(number_of_components):
        axs[i].bar(df_pca.columns,pca_np[i], color=cm.get_cmap("tab20")(range(9)))
        axs[i].set_title("PC%d ratio: %.2f"%(i+1, pca.explained_variance_ratio_[i]))
        plt.setp(axs[i].get_xticklabels(), rotation=45, ha="right")

    fig.savefig(get_results_filepath(trace_size,"PCA_Loadings"),bbox_inches = "tight")
    plt.close(fig)

def elbow_method_kmeans(X, max_k, fig_name=None):
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1, max_k+1)
    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)

        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                            'euclidean'), axis=1)) / X.shape[0])
        inertias.append(kmeanModel.inertia_)

        mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                       'euclidean'), axis=1)) / X.shape[0]
        mapping2[k] = kmeanModel.inertia_
    ### print and show 
    for key, val in mapping1.items():
        print(f'{key} : {val}')
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.legend()
    if fig_name is None:
        plt.show()
    else:
        plt.savefig(fig_name)
        plt.close()

def init_tsne_model(perplexity,N,**kwargs):
    skl_kw = dict(
        n_components=2,
        perplexity=perplexity,
        n_neighbors=max(102-0.0012*N,30),
        learning_rate_method=None,
        early_exaggeration=(N>10000)*12+12.0,
        exaggeration_iter=500,
        learning_rate=max(N/3, 200),
        n_iter=3000,
        n_iter_without_progress=300,
        min_grad_norm=1e-07,
        metric='euclidean',
        init='random',
        verbose=0,
        random_state=1,#None,
        method= 'barnes_hut',#'fft'
        angle=0.5,
        square_distances='legacy'
    )
    skl_kw.update(kwargs)
    tsne_model = TSNE(**skl_kw)
    print(tsne_model.get_params())
    return tsne_model

if __name__ == '__main__':
    tstart = time.time()
    execute_clustering(**dict((arg.split('=')[0],int(arg.split('=')[1])) for arg in sys.argv[1:]))
    tend = time.time()
    print("Running time:", tend-tstart, "sec.")
