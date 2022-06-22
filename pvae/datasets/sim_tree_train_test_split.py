"""
split sim trees into training set and testing set 
"""

# load package 
import os 
import numpy as np
import pandas as pd
import networkx as nx  
from itertools import combinations

import matplotlib.pyplot as plt 


# ===================== IO +=======================
def load_sim_tree(path: str):
    """ load all sim tree data """
    with open(os.path.join(path, 'sim_tree_points.npy'), 'rb') as f:
        nodes_positions = np.load(f)
    with open(os.path.join(path, 'sim_tree_edges.npy'), 'rb') as f:
        edges = np.load(f)
    with open(os.path.join(path, 'sim_tree_dist_mat.npy'), 'rb') as f:
        dist_mat = np.load(f)
    return nodes_positions, edges, dist_mat

def save_sim_tree(path: str, suffix: str, nodes_positions, edges, dist_mat):
    """ 
    save all sim tree data 
    :param suffix: specify train/test code index
    """
    with open(os.path.join(path, f'sim_tree_points_{suffix}.npy'), 'wb') as f:
        np.save(f, nodes_positions)
    with open(os.path.join(path, f'sim_tree_edges_{suffix}.npy'), 'wb') as f:
        np.save(f, edges)
    with open(os.path.join(path, f'sim_tree_dist_mat_{suffix}.npy'), 'wb') as f:
        np.save(f, dist_mat)


# =================== tree ========================
def sample_from_sim_tree(
        nodes_positions:np.ndarray, 
        edges:np.ndarray, 
        dist_mat:np.ndarray, 
        num_sample_points:np.ndarray
    ):
    """ 
    sample points from tree edges, linear interpolation for distance metric
    :param num_sample_points: the total number of points to sample 
    """
    # construct a graph 
    g = nx.Graph()
    g.add_edges_from(edges)
    attr_dict = {}
    for (n1_idx, n2_idx) in edges:
        dist = dist_mat[n1_idx, n2_idx]
        attr_dict[(n1_idx, n2_idx)] = dist 
    nx.set_edge_attributes(g, attr_dict, 'dist')

    cur_num_points = len(nodes_positions)
    sampled_point_idx = cur_num_points


    # sample from edge 
    new_sampled_positions = []
    edges_copy = edges.copy()
    np.random.shuffle(edges_copy)
    edge_idx = 0
    cur_num_new_points = 0
    max_points_per_edge = num_sample_points * 3 // cur_num_points  # tunnable 
    while edge_idx < cur_num_points and cur_num_new_points < num_sample_points:
        (n1_idx, n2_idx) = edges_copy[edge_idx]
        edge_dist = dist_mat[n1_idx, n2_idx]
        n1, n2 = nodes_positions[n1_idx], nodes_positions[n2_idx]

        # determine points positions 
        num_points_cur_edge = min(np.random.randint(0, max_points_per_edge) + 1, num_sample_points - cur_num_new_points)
        cur_num_new_points += num_points_cur_edge
        sampled_positions_proportion = np.random.uniform(0.1, 0.9, (num_points_cur_edge, 1))
        sampled_positions_proportion.sort(axis=0)
        direction_vector = n2 - n1 
        sampled_points = n1 + sampled_positions_proportion @ np.array([direction_vector])

        # append to graph 
        g.remove_edge(n1_idx, n2_idx)
        for sampled_point in sampled_points:
            new_sampled_positions.append(sampled_point)
        proportion_list = [0, *sampled_positions_proportion, 1]
        diff_proportion_list = np.diff(proportion_list)
        sampled_points = [n1, *sampled_points, n2]  # append to front and end 
        sampled_points_idx = [n1_idx, *range(sampled_point_idx, sampled_point_idx + num_points_cur_edge), n2_idx]
        sampled_point_idx += num_points_cur_edge
        for i in range(1, len(sampled_points)):
            g.add_edge(
                sampled_points_idx[i - 1],
                sampled_points_idx[i],
                dist=diff_proportion_list[i - 1] * edge_dist
            )
            

        # move to next edge 
        edge_idx += 1
    
    new_sampled_positions = np.vstack(new_sampled_positions)
    # new_edge_list = np.array(new_edge_list)

    # ensure connectedness 
    leaves_candidate = [idx for idx in g.nodes() if len(list(g.neighbors(idx))) == 1 and list(g.neighbors(idx))[0] < cur_num_points]
    while leaves_candidate: 
        for leaf in leaves_candidate:
            # print(leaf)
            g.remove_edge(leaf, list(g.neighbors(leaf))[0])
        # update leaf 
        leaves_candidate = [idx for idx in g.nodes() if len(list(g.neighbors(idx))) == 1 and list(g.neighbors(idx))[0] < cur_num_points]

    new_edges = np.array(list(g.edges()))

    # compute distance matrix 
    unique_idx = np.unique(new_edges.flatten())
    unique_idx.sort()
    dist_mat_shape = len(unique_idx)

    # conxtruct matrix
    new_dist_mat = np.zeros((dist_mat_shape, dist_mat_shape))
    triu_indices = np.triu_indices_from(new_dist_mat, k=1)

    for (n1_idx, n2_idx), r_idx, c_idx in zip(combinations(unique_idx, 2), *triu_indices):
        new_dist_mat[r_idx, c_idx] = nx.shortest_path_length(g, n1_idx, n2_idx, weight='dist')
    new_dist_mat = new_dist_mat + new_dist_mat.T   # symmetrize 

    # get new points 
    total_posisitons = np.vstack((nodes_positions, new_sampled_positions))
    total_sampled_positions = total_posisitons[unique_idx]

    return total_sampled_positions, new_edges, new_dist_mat


def visualize_train_test(
        nodes_positions, 
        edges, 
        train_nodes_positions, 
        test_nodes_positions
    ):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].scatter(nodes_positions[:, 0], nodes_positions[:, 1])
    ax[1].scatter(nodes_positions[:, 0], nodes_positions[:, 1])
    for point_1_idx, point_2_idx in edges:
        point_1 = nodes_positions[point_1_idx]
        point_2 = nodes_positions[point_2_idx]
        
        x_values = [point_1[0], point_2[0]]
        y_values = [point_1[1], point_2[1]]

        ax[0].plot(x_values, y_values, 'grey', linewidth=0.5)
        ax[1].plot(x_values, y_values, 'grey', linewidth=0.5)
    
    # add train 
    ax[0].scatter(train_nodes_positions[:, 0], train_nodes_positions[:, 1], c='tab:red')
    ax[1].scatter(test_nodes_positions[:, 0], test_nodes_positions[:, 1], c='tab:green')

    ax[0].set_title('train')
    ax[1].set_title('test')

