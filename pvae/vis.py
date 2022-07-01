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
def visualize_embeddings(trained_emb, edges, model_type, loss, diameter, thr, distortion):
    """ 
    plot embeddings, along with the hard boundary

    :param trained_emb: the training embeddings 
    :param edges: the edge list
    :param model_type: the name of the model 
    :param loss: the loss at a particular epoch,
    :param diameter: the diameter of the embedding
    :param thr: the threshold for hard boundary 
    :param distortion: the training distortion 
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
    ax.set_title('{} \n loss: {:.4f}, distortion: {:.4f}, diameter: {:.4f}'.format(model_type, loss, distortion, diameter))

    # visualize hard boundary 
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(thr * np.cos(t), thr * np.sin(t), linewidth=1, color='darkred')

    return fig

def visualize_train_test_embeddings(
        model_type,
        train_emb, train_edges, train_loss, train_diameter, train_distortion,
        test_emb, test_edges, test_loss, test_diameter, test_distoriton,
        thr
    ):
    """ 
    plot train test embeddings, along with the hard boundary
    """
    # build graph        
    train_g = nx.Graph()
    train_g.add_edges_from(train_edges)
    test_g = nx.Graph()
    test_g.add_edges_from(test_edges)

    # node colors 
    # ! fixed for now
    train_node_colors = []
    train_root = np.min(train_g.nodes())  # smallest index as root 
    # print(train_edges)
    # print(train_g.nodes())
    for node in (list(train_g.nodes())):
        path_length = nx.shortest_path_length(train_g, node, train_root)
        train_node_colors.append(1 - path_length / 10)
    test_node_colors = []
    test_root = np.min(test_g.nodes())
    for node in (list(test_g.nodes())):
        path_length = nx.shortest_path_length(test_g, node, test_root)
        test_node_colors.append(1 - path_length / 10)

    # read node embeddings 
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    # ax = fig.gca()
    draw_kwargs = {'node_size': 15, 'width': 0.1, 'cmap': 'Blues'}

    # draw train 
    train_nodes_idx = sorted(list(train_g.nodes()))
    nx.draw(
        train_g, 
        pos=dict(zip(train_nodes_idx, train_emb)),
        node_color=train_node_colors,
        ax=ax[0],
        **draw_kwargs
    )
    ax[0].scatter([train_emb[0][0]], [train_emb[0][1]], color='red')
    ax[0].set_title('train {} \n loss: {:.4f}, distortion: {:.4f}, diameter: {:.2f}'.format(model_type, train_loss, train_distortion, train_diameter))

    # visualize hard boundary 
    t = np.linspace(0, 2 * np.pi, 100)
    ax[0].plot(thr * np.cos(t), thr * np.sin(t), linewidth=1, color='darkred')

    # draw test 
    test_nodes_idx = sorted(list(test_g.nodes()))
    nx.draw(
        test_g, 
        pos=dict(zip(test_nodes_idx, test_emb)),
        node_color=test_node_colors,
        ax=ax[1],
        **draw_kwargs
    )
    ax[1].scatter([test_emb[0][0]], [test_emb[0][1]], color='red')
    ax[1].set_title('test {} \n loss: {:.4f}, distorion: {:.4f}, diameter: {:.2f}'.format(model_type, test_loss, test_distoriton, test_diameter))

    # visualize hard boundary 
    ax[1].plot(thr * np.cos(t), thr * np.sin(t), linewidth=1, color='darkred')   

    return fig

def convert_fig_to_array(fig):
    """ convert a matplotlib fig to numpy array (H, W, C) """
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img_arr = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return img_arr
