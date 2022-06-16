from graph_tool.draw import Graph, GraphView, graph_draw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.config import CAM_POS


def transition_rates_for_fish(fish_key,clusters,traces_all):
    fish_filter = traces_all[CAM_POS] == fish_key
    clusters_f=clusters[fish_filter]
    #cl, counts = np.unique(clusters_f, return_counts=True)
    #print(counts/clusters.shape[0])
    transitions = transition_rates(clusters_f)
    return transitions

def transition_rates(clusters):
    transitions = pd.crosstab(pd.Series(clusters[:-1],name='from'),pd.Series(clusters[1:],name='to'),
                              normalize=0, dropna=True)
    return transitions

def transition_rates_over_all(fish_keys, clusters,n_clusters, traces_all, trace_size):
    tra_r = transition_rates(fish_keys[0], clusters,n_clusters, traces_all, trace_size, normalize=False)
    for f_key in fish_keys[1:]:
        tra_r = tra_r + transition_rates(f_key, clusters,n_clusters, traces_all, trace_size, normalize=False)
    return tra_r.div(tra_r.sum(axis=1), axis=0)

def stationary_distribution(marcov_matrix):    
    P=marcov_matrix
    print(P.shape)
    A=np.append(P.T-np.identity(P.shape[0]),np.ones((1,P.shape[0])),axis=0)
    b=np.array([*[0 for i in range(P.shape[0])],1]).T
    sta_dist = np.linalg.solve((A.T).dot(A), (A.T).dot(b))
    return sta_dist

def draw_transition_graph(t, X_em, clusters,ax=None, flip_y=True, output=None, ink_scale=1, vweight_scale=250, eweight_scale=10, vertex_pen_width=0.5,**kwargs):
    g = Graph(directed=True)
    t_np = t.to_numpy()
    edges = [(i,j,t_np[i,j]*eweight_scale, max(t_np[i,j]*3*eweight_scale, eweight_scale/2), "%.2f"%t_np[i,j]) for (i,j) in np.transpose(t_np.nonzero())]
    eweight = g.new_ep("double")
    elabel = g.new_ep("string")
    emarker = g.new_ep("double")
    vweight = g.new_vp("double")
    vcolor = g.new_vp("vector<double>")
    pos = g.new_vp("vector<double>")
    eprop = g.add_edge_list(edges, eprops=[eweight,emarker,elabel])
    vweight.a=stationary_distribution(t_np)*vweight_scale
    for v in g.vertices():
        vidx = g.vertex_index[v]
        X_c_mean = np.mean(X_em[clusters==vidx],axis=0)
        pos[v] = [X_c_mean[0], -X_c_mean[1] if flip_y else X_c_mean[1]]
        vcolor[v] = plt.get_cmap("tab10", t_np.shape[0])(g.vertex_index[v])
        
    gd = graph_draw(g,pos=pos,
                    fit_view=0.8,ink_scale=ink_scale,fit_view_ink=True,
                    **kwargs,
                    vertex_size=vweight,
                    vertex_text=g.vertex_index,
           vertex_fill_color=vcolor,
            vertex_pen_width=vertex_pen_width,
           eprops={"pen_width":eweight, "text":elabel, "font_size":emarker,
                   "marker_size":emarker,
                   "color":[0.179, 0.203,0.210, 0.7]},
           adjust_aspect=True, output=output,mplfig=ax)
    return g