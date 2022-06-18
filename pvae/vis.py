# visualisation helpers for data

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import torch

import networkx as nx

sns.set()

def array_plot(points, filepath):
    data = points[0]
    period = len(points) + 1
    a = np.zeros((period*data.shape[0], data.shape[1]))
    a[period*np.array(range(data.shape[0])),:] = data
    if period > 2:
        recon = points[1]
        a[period*np.array(range(data.shape[0]))+1,:] = recon
    ax = sns.heatmap(a, linewidth=0.5, vmin=-1, vmax=1, cmap=sns.color_palette('RdBu_r', 100))
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.clf()

# ============== for simulation only ===================
def visualize_embeddings(trained_emb, edges, model_type, loss, diameter, thr):
    """ 
    plot embeddings, along with the hard boundary

    :param trained_emb: the training embeddings 
    :param edges: the edge list
    :param model_type: the name of the model 
    :param loss: the loss at a particular epoch,
    :param diameter: the diameter of the embedding
    :param thr: the threshold for hard boundary 
    :return a fig 
    """
    # build graph        
    g = nx.Graph()
    g.add_edges_from(edges)

    # node colors 
    # ! fixed for now
    node_colors = []
    for node in range(g.number_of_nodes()):
        path_length = nx.shortest_path_length(g, node, 0)
        node_colors.append(1 - path_length / 10)

    # read node embeddings 
    fig = Figure(figsize=(6, 6))
    ax = fig.gca()
        
    nx.draw(
        g, 
        pos=dict(zip(range(g.number_of_nodes()), trained_emb)),
        node_size=15,
        width=0.1,
        node_color=node_colors,
        cmap='Blues',
        ax=ax
    )
    ax.scatter([trained_emb[0][0]], [trained_emb[0][1]], color='red')
    ax.set_title('{} \n loss: {:.4f}, diameter: {:.2f}'.format(model_type, loss, diameter))

    # visualize hard boundary 
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(thr * np.cos(t), thr * np.sin(t), linewidth=1, color='darkred')

    return fig

def convert_fig_to_array(fig):
    """ convert a matplotlib fig to numpy array (H, W, C) """
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img_arr = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return img_arr
