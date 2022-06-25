import plotly.express as px
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
import numpy as np
from src.clustering import *

def get_convex_hull(points): 
    hull = ConvexHull(points)
    cycle = points[hull.vertices]
    cycle = np.concatenate([cycle, cycle[:1]])
    return cycle

colors_go = ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
             'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
             'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
             'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
             'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
             'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
             'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
             'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
             'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
             'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
             'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
             'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
             'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
             'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
             'ylorrd']
mycolors=['reds', 'greens', 'blues']
colorscales = [
                [[0, 'rgb(255, 191,0)'], [1, 'rgb(255, 191,100)']],
                [[0, 'rgb(0,200,0)'], [1, 'rgb(0,255,100)']],
                [[0, 'rgb(200,0,0)'], [1, 'rgb(255,0,100)']],
                [[0, 'rgb(235, 52, 232)'], [1, 'rgb(235, 52, 232)']],
                [[0, 'rgb(235, 140, 52)'], [1, 'rgb(235, 140, 52)']],
                [[0, 'rgb(128, 52, 235)'], [1, 'rgb(128, 52, 235)']],
                [[0, 'rgb(143, 37, 123)'], [1, 'rgb(143, 37, 123)']]
              ]
bg_color='rgba(255,255,255,1)'
white2blue=[[0, bg_color], [1, 'rgb(32, 87, 179)']]

def draw_multi_density_TSNE(X_embedded, clusters, number_of_clusters, fig_name="multi_density_TSNE", start=20, end=100, size=15, width=3):
    x = X_embedded[:,0]
    y = X_embedded[:,1]
    res = [go.Histogram2dContour( x = x, y = y, contours=dict(coloring="fill", showlines=False), colorscale=white2blue, showscale=False)]
    for i in range(number_of_clusters):
        h2d = go.Histogram2dContour(opacity=0.7,showscale=False, line=dict(width=width),contours=dict(coloring="lines", start=start, end=end, size=size),
                                    x = x[clusters==i], y = y[clusters==i], autocolorscale=False,
                                    colorscale=colorscales[i])
        res.append(h2d)
    fig = go.Figure(res)
    fig.update_layout( height = 600, width = 600, showlegend=False, paper_bgcolor=bg_color, plot_bgcolor=bg_color, margin_t=0, margin_b=0, margin_l=0, margin_r=0)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.write_image("%s.pdf"%fig_name)
    return fig

def draw_density_TSNE(X_embedded, clusters, number_of_clusters, fig_name="density_TSNE"):
    x = X_embedded[:,0]
    y = X_embedded[:,1]
    res = go.Histogram2dContour(x = x, y = y, colorscale = white2blue, showscale=False,contours=dict(coloring="fill", showlines=False))
    hulls = [go.Scatter( line = {"width": 5}, **dict(zip(["x","y"],get_convex_hull(X_embedded[clusters==i]).T))) for i in range(number_of_clusters)]
    fig = go.Figure(hulls + [res])
    fig.update_layout( height = 600, width = 600, showlegend=False, paper_bgcolor=bg_color, plot_bgcolor=bg_color, margin_t=0, margin_b=0, margin_l=0, margin_r=0)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.write_image("%s.pdf"%fig_name)
    return fig

def draw_raw_density_TSNE(X_embedded, fig_name="raw_density_TSNE"):
    x = X_embedded[:,0]
    y = X_embedded[:,1]
    res = go.Histogram2dContour(x = x, y = y, colorscale = white2blue, showscale=False,contours=dict(coloring="fill", showlines=False))
    fig = go.Figure([res])
    fig.update_layout( height = 600, width = 600, showlegend=False, paper_bgcolor=bg_color, plot_bgcolor=bg_color, margin_t=0, margin_b=0, margin_l=0, margin_r=0)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.write_image("%s.pdf"%fig_name)
    return fig

def plot_lines_from_locations(X_embedded, samples, radius=2, bins=10, limit=10, fig_name="line_clusters.pdf"):
    set_N, centers = set_of_neighbourhoods(X_embedded, samples, radius=radius, bins=bins)
    nrows=2
    len_N = int(len(set_N)/nrows)

    fig, axs = plt.subplots(nrows=nrows,ncols=len_N, figsize=(len_N*4,nrows*4),sharex=True, sharey=True)
    c_i=0
    for l_key, ax in zip(set_N.keys(),np.concatenate(axs)):
        plot_lines_cumsum(set_N[l_key], ax=ax, title="cluster: %d, %s"%(c_i,l_key), limit=limit)
        c_i+=1
    fig.savefig(fig_name)
    return np.array(centers)