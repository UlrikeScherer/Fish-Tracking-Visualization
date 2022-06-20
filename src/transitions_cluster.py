from graph_tool.draw import Graph, GraphView, graph_draw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.config import CAM_POS, DAY
from src.clustering import get_results_filepath
from src.utile import get_all_days_of_context, get_date_string


def transition_rates_for_fish(fish_key,clusters, traces_all, normalize=0):
    fish_filter = traces_all[CAM_POS] == fish_key
    clusters_f=clusters[fish_filter]
    #cl, counts = np.unique(clusters_f, return_counts=True)
    #print(counts/clusters.shape[0])
    transitions = transition_rates(clusters_f, normalize=normalize)
    return transitions

def transition_rates(clusters, normalize=0):
    transitions = pd.crosstab(pd.Series(clusters[:-1],name='from'),pd.Series(clusters[1:],name='to'),
                              normalize=0, dropna=True)
    return transitions

def transition_rates_over_all(fish_keys, clusters, traces_all, trace_size):
    tra_r = transition_rates_for_fish(fish_keys[0], clusters, traces_all, normalize=False)
    for f_key in fish_keys[1:]:
        tra_r = tra_r + transition_rates_for_fish(f_key, clusters, traces_all, normalize=False)
    return tra_r.div(tra_r.sum(axis=1), axis=0)

def stationary_distribution(marcov_matrix):    
    P=marcov_matrix
    print(P.shape)
    A=np.append(P.T-np.identity(P.shape[0]),np.ones((1,P.shape[0])),axis=0)
    b=np.array([*[0 for i in range(P.shape[0])],1]).T
    sta_dist = np.linalg.solve((A.T).dot(A), (A.T).dot(b))
    return sta_dist

def draw_transition_graph(t, n_clusters, positions=None, ax=None, flip_y=True, output=None, ink_scale=1, vweight_scale=250, eweight_scale=10, vertex_pen_width=0.05,**kwargs):
    g = Graph(directed=True)
    t_np = t.to_numpy()
    edges = [(t.index[i],t.columns[j],t_np[i,j]*eweight_scale, max(t_np[i,j]*3*eweight_scale, eweight_scale/2), "%.2f"%t_np[i,j]) for (i,j) in np.transpose(t_np.nonzero())]
    eweight = g.new_ep("double")
    elabel = g.new_ep("string")
    emarker = g.new_ep("double")
    vweight = g.new_vp("double")
    vcolor = g.new_vp("vector<double>")
    pos = g.new_vp("vector<double>")
    eprop = g.add_edge_list(edges, eprops=[eweight,emarker,elabel])
    vweight.a[t.index]=stationary_distribution(t_np)*vweight_scale
    for v in g.vertices():
        vidx = g.vertex_index[v]
        X_c_mean = positions[vidx]
        pos[v] = [X_c_mean[0], -X_c_mean[1] if flip_y else X_c_mean[1]]
        vcolor[v] = plt.get_cmap("tab10", n_clusters)(vidx)
        if vidx not in t.index: 
            vcolor[v]=(1,1,1,0)
        
    gd = graph_draw(g,pos=pos,
                    fit_view=0.8,ink_scale=ink_scale,fit_view_ink=True,
                    **kwargs,
                    vertex_size=vweight,
                    vertex_text=g.vertex_index,
           vertex_fill_color=vcolor,
            vertex_pen_width=vertex_pen_width,
            vertex_font_size=vweight,
           eprops={"pen_width":eweight, "text":elabel, "font_size":emarker,
                   "marker_size":emarker,
                   "color":[0.179, 0.203,0.210, 0.7]},
           adjust_aspect=True, output=output,mplfig=ax)
    return g

def iter_fish_subset(fish_keys, table, *arrays):
    assert(is_of_equal_size(*arrays,table))
    return map(lambda fk: (fk,*flt_on_key(fk, CAM_POS, table, *arrays)), fish_keys)

def flt_on_key(value, table_key, table, *arrays):
    flt=table[table_key]==value
    return (X[flt] for X in arrays)

def is_of_equal_size(*arrays):
    return np.alltrue([a.shape[0] == arrays[0].shape[0] for a in arrays])

def plot_transitions_individuality_develpoment(fish_keys, table, X_embedded, clusters,n_clusters, trace_size):
    days=get_all_days_of_context()
    #TRANSITION 
    t = transition_rates_over_all(fish_keys, clusters, table, trace_size)
    cluster_means = [np.mean(X_embedded[clusters==idx],axis=0) for idx in range(n_clusters)]
    draw_transition_graph(t,n_clusters, positions=cluster_means,flip_y=True, output=get_results_filepath(trace_size, "all_transitions_c%s"%t.shape[0]))
    
    for fk,X,C,table in iter_fish_subset(fish_keys, table, X_embedded, clusters, table):
        t = transition_rates(C)
        fpath = get_results_filepath(trace_size,"ft_%d_%s"%(n_clusters, fk), subfolder="individualty")
        g = draw_transition_graph(t,n_clusters, positions=cluster_means,flip_y=True,output=fpath)
        day_before = "0"
        for i,day in enumerate(days[6::7]):
            flt_d = (table["DAY"]>day_before) & (table["DAY"] <= day)
            C_d = C[flt_d]
            t_d = transition_rates(C_d)
            if t_d.shape[0] == 0: continue
            if t_d.shape[0]==3: print(fk)
            day_before = day
            weekpath = get_results_filepath(trace_size,"ft_c%d_%s_week%d"%(n_clusters, fk, i), subfolder="development/%s"%fk, format="png")
            g = draw_transition_graph(t_d,n_clusters, positions=cluster_means,flip_y=True,output=weekpath)
            
#### --------- CLUSTER DISTRIBUTION ----------------- ####
def get_cluster_distribution(clusters, n_clusters):
    dist=np.zeros(n_clusters)
    uni, counts = np.unique(clusters, return_counts=True)
    dist[uni]=counts/clusters.shape[0]
    return dist

def cluster_distribution_over_days(table, clusters, n_clusters, days, trace_size, fish_key=None):
    y = []
    name="%d_clusters_distribution_over_days"%n_clusters
    subfolder="development"
    
    if fish_key is not None:
        clusters, table = flt_on_key(fish_key, CAM_POS, table,clusters, table)
        name="%s_%s"%(name, fish_key)
        subfolder="%s/%s"%(subfolder,fish_key)
        
    for d in days:
        flt = table[DAY]==d
        dist = get_cluster_distribution(clusters[flt], n_clusters)
        y.append(dist)
    df = pd.DataFrame(y, index=[get_date_string(d) for d in days], columns=["cluster %d"%c for c in range(n_clusters)])
    df.plot(kind = 'bar', stacked = True, colormap="tab10", figsize=(10,5), rot=45, position=-1)
    plt.tight_layout()
    filepath = get_results_filepath(trace_size,name,subfolder=subfolder)
    plt.savefig(filepath)
    plt.close()
            

