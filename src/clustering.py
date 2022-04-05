from src.transformation import rotation, px2cm
from methods import avg_turning_direction, calc_steps, tortuosity_of_chunk 
from src.metrics import entropy_for_chunk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def rotate_trace(trace):
    alph = np.arctan2(*trace[0])
    return np.dot(trace,rotation(-alph))

def transfrom_to_traces(batch, trace_size):
    setX = batch[["xpx", "ypx"]].to_numpy()
    setX = setX[1:]-setX[:-1]
    lenX = setX.shape[0]
    sizeSet = int(lenX/trace_size)
    newSet = np.zeros((sizeSet,trace_size,2))
    for i in range(sizeSet-1):
        newSet[i,:,:] = rotate_trace(setX[i*trace_size:(i+1)*trace_size])
    X = np.reshape(newSet, (sizeSet, trace_size*2))
    return X, newSet

def transfrom_to_traces2(batch, trace_size):
    setX = batch[["xpx", "ypx"]].to_numpy()
    #setX = setX[1:]-setX[:-1]
    steps = calc_steps(setX)
    angels = avg_turning_direction(setX)
    #print(len(steps), len(angels))
    lenX = setX.shape[0]
    sizeSet = int(lenX/trace_size)
    newSet = np.zeros((sizeSet,trace_size,2))
    for i in range(sizeSet):
        newSet[i,:,0] = steps[i*trace_size:(i+1)*trace_size]
        if len(angels)<=i: continue
        newSet[i,:,1] = angels[i*trace_size:(i+1)*trace_size]
    X = np.reshape(newSet, (sizeSet, trace_size*2))
    return X, transfrom_to_traces(batch, trace_size)[1]

def transfrom_to_traces_metric_based(batch, trace_size):
    setX = batch[["xpx", "ypx"]].to_numpy()
    steps = calc_steps(setX)
    angels = avg_turning_direction(setX)
    
    lenX = setX.shape[0]
    sizeSet = int(lenX/trace_size)
    newSet = np.zeros((sizeSet,4))
    traceSet = np.zeros((sizeSet,trace_size,2))
    
    for i in range(sizeSet):
        newSet[i,0] = np.mean(steps[i*trace_size:(i+1)*trace_size])
        if len(angels)>i:
            newSet[i,1] = sum(angels[i*trace_size:(i+1)*trace_size])/len(angels)
        newSet[i,2] = np.mean(tortuosity_of_chunk(setX[i*trace_size:(i+1)*trace_size]))
        newSet[i,3], _ = entropy_for_chunk(setX[i*trace_size:(i+1)*trace_size])
        if np.isnan(newSet[i,3]): newSet[i,3] = 0.0
    return newSet, transfrom_to_traces(batch, trace_size)[1]

def transfrom_to_traces_seq_metrics(batch, trace_size, step):
    setX = batch[["xpx", "ypx"]].to_numpy()
    steps = calc_steps(setX)
    angels = avg_turning_direction(setX)
    
    lenX = setX.shape[0]
    setSize = int(lenX/step)
    newSet = np.zeros(())
    
    #for i in enumerate(range(0,setSize,step)):
        
        
def normalize_data(traces):
    d_std, d_mean = np.std(traces[:,0::2]), np.mean(traces[:,0::2])
    a_std, a_mean = np.std(traces[:,1::2]), np.mean(traces[:,1::2])
    traces[:,0::2]=(traces[:,0::2]-d_mean)/d_std
    traces[:,1::2]=(traces[:,1::2]-a_mean)/a_std
    return traces

def normalize_data_metrics(traces):
    for i in range(traces.shape[1]):
        d_std, d_mean = np.std(traces[:,i]), np.mean(traces[:,i])
        traces[:,i]=(traces[:,i]-d_mean)/d_std
    return traces

def TSNE_vis(X_embedded):
    cmap=["Blues", "Reds"]
    p = sns.kdeplot(x=X_embedded[:,0], y=X_embedded[:,1], cmap=cmap[0], shade=True, thresh=0)
    print(p)
    #plt.scatter(X_embedded[i*half:(i+1)*half,0], X_embedded[i*half:(i+1)*half, 1], marker="o", label=i, alpha=0.5)
    plt.savefig("TSNE.pdf")
    return p

def TSNE_vis_2(X_embedded, centers=None, clusters=None, fig_name="TSNE_vis"):
    x = X_embedded[:,0]
    y = X_embedded[:,1]
    res = go.Histogram2dContour(x = x, y = y, colorscale = 'Blues')
    fig = go.Figure(res)
    if centers is not None: 
        fig.add_trace(go.Scatter(
            x = centers[:,0], y = centers[:,1],
            xaxis = 'x', yaxis = 'y',
            mode = 'markers',
            marker = dict(
                color = 'rgba(255, 255, 102,0.5)',
                size = 20
            ),
            text=["cluster %d, n: %d"%(i,n) for (i,n) in enumerate(centers[:,2])]
        ))
        
    fig.add_trace(go.Scatter(
        x = x, y = y,
        xaxis = 'x', yaxis = 'y',
        mode = 'markers',
        marker = dict(
            cmax=clusters.max(),
            cmin=0,
            color=clusters,
            colorbar=dict(
                title="Clusters"
            ),
            colorscale="Viridis",
            #color = 'rgba(0,0,0,0.1)',
            size = 3,
            opacity=0.5
        )
    ))
    
    fig.update_layout( height = 600, width = 600 )
    fig.write_image("%s.pdf"%fig_name)
    return fig

def get_neighbourhood_selection(X_embedded, nSs, c_x=None, c_y=None, radius=1):
    if c_x is None and c_y is None:
        hist, x,y = np.histogram2d(X_embedded[:,0], X_embedded[:,1], bins=10)
        max_x, max_y = int(hist.argmax()/10),hist.argmax()%10
        c_x, c_y = (x[max_x]+x[max_x+1])/2, (y[max_y]+y[max_y+1])/2
    find = np.nonzero((np.abs(X_embedded[:,0]-c_x)+np.abs(X_embedded[:,1]-c_y)) <= radius)
    return nSs[find[0]]

def set_of_neighbourhoods(X_embedded, nSs, radius=1, bins=15):
    hist, x,y = np.histogram2d(X_embedded[:,0], X_embedded[:,1], bins=bins)
    len_x = len(x)
    neighbourhoods = dict()
    centers = list()
    for (max_x,max_y) in zip(range(len_x),hist.argsort()[:,-1]):
        c_x, c_y = (x[max_x]+x[max_x+1])/2, (y[max_y]+y[max_y+1])/2
        centers.append([c_x, c_y,hist[max_x, max_y]])
        neighbourhoods["x:%.2f, y:%.2f, n:%d"%(c_x,c_y, hist[max_x, max_y])] = get_neighbourhood_selection(X_embedded, nSs, c_x, c_y, radius=radius)
    return neighbourhoods, centers

def plot_lines_angle(lines_to_plot, limit=20):
    for line in lines_to_plot[:limit]:
        angels = np.cumsum(line[:,1])
        x = line[:,0]*np.cos(angels)
        y = line[:,0]*np.sin(angels)
        plt.plot(np.cumsum(x), np.cumsum(y))
    plt.savefig("lines_exp.pdf")
    
def plot_lines_cumsum(lines_to_plot, limit=20, ax=None, title="x:, y: "):
    if ax is not None: ax.set_title(title)
    for line in lines_to_plot[:limit]:
        line = np.cumsum(line*px2cm(),axis=0)
        if ax is None:
            plt.plot(line[:,0], line[:,1])
        else: 
            ax.plot(line[:,0], line[:,1])
            
    plt.savefig("lines_exp.pdf")
    