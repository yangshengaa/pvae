# load packages 
import os 
import random
import pickle
from csv import reader
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt 

import networkx as nx 

from sklearn.neighbors import NearestNeighbors

import torch
import torch.utils.data


def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = list(lines)
    return dataset


class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, csv_name):
        filename = 'data/{}.csv'.format(csv_name)
        dataset = np.array(load_csv(filename))
        dataset = dataset[1:, :]
        self.images = dataset[:, 0:-1].astype(np.float)
        self.latents = dataset[:, [-1]]
        self.latents = self.latents.astype(np.int)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.Tensor(self.images[idx, :])
        latent = torch.Tensor(self.latents[idx])
        return (image, latent)


class SyntheticDataset(torch.utils.data.Dataset):
    '''
    Implementation of a synthetic dataset by hierarchical diffusion. 
    Args:
    :param int dim: dimension of the input sample
    :param int depth: depth of the tree; the root corresponds to the depth 0
    :param int :numberOfChildren: Number of children of each node in the tree
    :param int :numberOfsiblings: Number of noisy observations obtained from the nodes of the tree
    :param float sigma_children: noise
    :param int param: integer by which :math:`\\sigma_children` is divided at each deeper level of the tree
    '''
    def __init__(self, dim, depth, numberOfChildren=2, sigma_children=1, param=1, numberOfsiblings=1, factor_sibling=10):
        self.dim = int(dim)
        self.root = np.zeros(self.dim)
        self.depth = int(depth)
        self.sigma_children = sigma_children
        self.factor_sibling = factor_sibling
        self.param = param
        self.numberOfChildren = int(numberOfChildren)
        self.numberOfsiblings = int(numberOfsiblings)  

        self.origin_data, self.origin_labels, self.data, self.labels = self.bst()

        # Normalise data (0 mean, 1 std)
        self.data -= np.mean(self.data, axis=0, keepdims=True)
        self.data /= np.std(self.data, axis=0, keepdims=True)

    def __len__(self):
        '''
        this method returns the total number of samples/nodes
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Generates one sample
        '''
        data, labels = self.data[idx], self.labels[idx]
        return torch.Tensor(data), torch.Tensor(labels)

    def get_children(self, parent_value, parent_label, current_depth, offspring=True):
        '''
        :param 1d-array parent_value
        :param 1d-array parent_label
        :param int current_depth
        :param  Boolean offspring: if True the parent node gives birth to numberOfChildren nodes
                                    if False the parent node gives birth to numberOfsiblings noisy observations
        :return: list of 2-tuples containing the value and label of each child of a parent node
        :rtype: list of length numberOfChildren
        '''
        if offspring:
            numberOfChildren = self.numberOfChildren
            sigma = self.sigma_children / (self.param ** current_depth)
        else:
            numberOfChildren = self.numberOfsiblings
            sigma = self.sigma_children / (self.factor_sibling*(self.param ** current_depth))
        children = []
        for i in range (numberOfChildren):
            child_value = parent_value + np.random.randn(self.dim) * np.sqrt(sigma)
            child_label = np.copy(parent_label)
            if offspring: 
                child_label[current_depth] = i + 1
            else:
                child_label[current_depth] = -i - 1
            children.append((child_value, child_label))
        return children

    def bst(self):
        '''
        This method generates all the nodes of a level before going to the next level
        '''
        queue = [(self.root, np.zeros(self.depth+1), 0)]
        visited = []
        labels_visited = []
        values_clones = []
        labels_clones = []
        while len(queue) > 0:
            current_node, current_label, current_depth = queue.pop(0)
            visited.append(current_node)
            labels_visited.append(current_label)
            if current_depth < self.depth:
                children = self.get_children(current_node, current_label, current_depth)
                for child in children:
                    queue.append((child[0], child[1], current_depth + 1)) 
            if current_depth <= self.depth:
                clones = self.get_children(current_node, current_label, current_depth, False)
                for clone in clones:
                    values_clones.append(clone[0])
                    labels_clones.append(clone[1])
        length = int(((self.numberOfChildren) ** (self.depth + 1) - 1) / (self.numberOfChildren - 1))
        length_leaves = int(self.numberOfChildren**self.depth)
        images = np.concatenate([i for i in visited]).reshape(length, self.dim)
        labels_visited = np.concatenate([i for i in labels_visited]).reshape(length, self.depth+1)[:,:self.depth]
        values_clones = np.concatenate([i for i in values_clones]).reshape(self.numberOfsiblings*length, self.dim)
        labels_clones = np.concatenate([i for i in labels_clones]).reshape(self.numberOfsiblings*length, self.depth+1)
        return images, labels_visited, values_clones, labels_clones

# =========== for simulation use only ===============
class SyntheticTreeDistortionDataSet(torch.utils.data.Dataset):
    """
    K-NN based
    explore distortion of tree like dataset using a hyperbolic embeddings 

    construct a tree-like data for simulation. The procedure is as follows:
        - randomly throw points 
        - build mst connecting these points 
        - randomly sample data points from its edge
        - add gaussian noise 
        - build KNN, precompute shortest path distances 
    
    :param n: the dimension 
    :param num_start_points: number of starting points to connect 
    :param num_points: the number of points to generate at the end 
    :param k: number of neighbors for connection
    :param centroids: the starting centroids to start with. If none, generate from raw
    :param is_noiseless: True if the generated tree data is noiseless 
    :param noise_level: a scalar controling the noise level 
    :return two dictionaries: 
        - Dict[idx, np.array] recording the original data points and 
        - Dict[(idx, idx), int] recording the shortest path distance 
    """

    def __init__(
        self, 
        n: int,
        num_start_points: int = 6,
        num_points: int = 500,
        k: int=5,
        centroids: np.ndarray=None,
        is_noiseless: bool=False,
        noise_level: float=1.
    ) -> None:
        self.n = n
        self.num_start_points = num_start_points
        self.num_points = num_points
        self.k = k
        self.centroids = centroids 
        self.is_noiseless = is_noiseless
        self.noise_level = noise_level

        # construct tree 
        self.sim_data_points_dict, self.shortest_path_dict, self.dist_mat = self.make_tree_data()

        # unpack 
        label_data_tuple = list(self.sim_data_points_dict.items())
        self.data = np.vstack([x[1] for x in label_data_tuple])
        self.labels = np.array([[x[0] for x in label_data_tuple]]).T
    
    # * the following two are meaningless
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return torch.Tensor(data), torch.Tensor(label)

    # ------- util -------
    def kruskal(self, start_points_array_dict: Dict):
        """ construct an mst, return edges """
        # compute pairwise Euclidean distance
        start_points_dist_dict = {}
        start_points_indices = start_points_array_dict.keys()
        idx_combinations = combinations(start_points_indices, 2)

        for idx_1, idx_2 in idx_combinations:
            start_point_1, start_point_2 = start_points_array_dict[idx_1], start_points_array_dict[idx_2]
            cur_dist = np.linalg.norm(start_point_1 - start_point_2)
            start_points_dist_dict[(idx_1, idx_2)] = cur_dist

        # sort in increasing order
        dist_sorted_tuple = sorted(start_points_dist_dict.items(), key=lambda x: x[1])

        # construct an empty graph
        g = nx.Graph()
        g.add_nodes_from(start_points_indices)
        for (idx_1, idx_2), cur_dist in dist_sorted_tuple:
            if not nx.has_path(g, idx_1, idx_2):
                g.add_edge(idx_1, idx_2)

        edges = g.edges
        return edges

    def make_tree_data(self):
        """ construct tree like data """
        # randomly create starting points from within a unit cube 
        if self.centroids is None: 
            start_points_array = np.random.uniform(
                low=-1, high=1, size=(self.num_start_points, self.n)
            )
            self.centroids = start_points_array  # for manual examination only
        else:
            start_points_array = self.centroids

        start_points_array_dict = dict(zip(range(start_points_array.shape[0]), start_points_array))

        # construct MST
        edges = self.kruskal(start_points_array_dict)
        self.edges = edges

        # generate noise
        if self.is_noiseless:
            scale = 0
            gaussian_noise = np.zeros((self.num_points, self.n))
        else:
            scale = self.noise_level * abs(np.min(np.max(start_points_array, axis=0))) / 10 / self.n  # larger dimension, smaller noise 
            gaussian_noise = np.random.normal(scale=scale, size=(self.num_points, self.n))

        # select edges and select branching point
        num_points_per_edge = self.num_points // len(edges)
        remainder = self.num_points - num_points_per_edge * len(edges)
        num_points_by_edge = [num_points_per_edge] * len(edges)
        num_points_by_edge[np.random.choice(range(len(edges)))] += remainder

        selected_branching_points = []
        for cur_num_points, (idx_1, idx_2) in zip(num_points_by_edge, edges):
            v1, v2 = start_points_array_dict[idx_1], start_points_array_dict[idx_2]
            random_prop = np.random.uniform(
                low=-scale,
                high=1+scale,
                size=(cur_num_points, 1)
            )
            random_branching_points = random_prop * (v2 - v1) + v1
            selected_branching_points.append(random_branching_points)

        # concat
        selected_branching_points = np.vstack(selected_branching_points)
        sim_data_points = selected_branching_points + gaussian_noise  # add noise
        sim_data_points_dict = dict(zip(range(self.num_points), sim_data_points))

        # KNN
        nbrs = NearestNeighbors(n_neighbors=self.k+1, n_jobs=-1).fit(sim_data_points)
        adj_matrix = nbrs.kneighbors_graph(sim_data_points).toarray() - np.eye(self.num_points)  # remove self-loops

        # construct graph
        g = nx.from_numpy_array(adj_matrix)
        if not nx.is_connected(g):  # redo randomization if not connected
            print('Randomized dataset not connected, redo randomization')
            return self.make_tree_data()

        # record shortest path distances
        shortest_path_dict = {}
        sim_data_indices = combinations(range(self.num_points), 2)

        # path dist along an edge 
        for idx_1, idx_2 in g.edges:
            data_1, data_2 = sim_data_points_dict[idx_1], sim_data_points_dict[idx_2]
            cur_dist = np.linalg.norm(data_1 - data_2)
            g.edges[idx_1, idx_2]['dist'] = cur_dist  # passed in as attributes
        self.g = g

        # create shortest path dict and dist matrix 
        dist_mat = torch.zeros(self.num_points, self.num_points, requires_grad=False)
        for idx_1, idx_2 in sim_data_indices:
            cur_shortest_dist = nx.shortest_path_length(g, idx_1, idx_2, weight='dist')
            shortest_path_dict[(idx_1, idx_2)] = torch.tensor(cur_shortest_dist)

            dist_mat[idx_1, idx_2] = cur_shortest_dist
            dist_mat[idx_2, idx_1] = cur_shortest_dist
        
        return sim_data_points_dict, shortest_path_dict, dist_mat

class SyntheticTreeDistortionDataSetPermute(torch.utils.data.Dataset):
    """ 
    no K-NN used. Two types of permutations: 
    - add_noise: add noise to nodes, connect the noise points to the node;
    - change_edge: change the edge connecting a leaf node to another parent who has the same grand-parent. 

    Need to input data points locations and edges to start with 
    """
    def __init__(
        self, 
        nodes_positions: np.ndarray, 
        edges: List[Tuple[int, int]],
        to_permute: bool=False,
        permute_type: str='add_noise',
        proportion_permute: float=0.3,
        use_path_length: bool=False,
        max_degree: int=3
    ) -> None:
        """ 
        :param nodes_positions: the euclidean positions of the data points. Assuming indexed by 0 to len(.) - 1
        :param edges: the list of edges to connect 
        :param to_permute: false not to permute, true otherwise 
        :param permute_type: either "add_noise" or "change_edge"
        :param proportion_permute: proportion of points to permute. 
        :param use_path_length: whether to use path length. False to use euclidean distances, otherwise length 1
        :param max_degree: a add_noise specific parameter. Number of noise points added for a fixed point follows uniform([1, max_degree])
        """
        super().__init__()
        self.nodes_positions = nodes_positions
        self.edges = list(edges)  # deep copy 
        self.to_permute = to_permute 
        self.permute_type = permute_type 
        self.proportion_permute = proportion_permute 
        self.use_path_length = use_path_length
        self.max_degree = max_degree 

        self.n = self.nodes_positions.shape[1]  # number of dimensions

        # construct a graph 
        self.g = self.construct_graph()

        # permute if necessary 
        if self.to_permute:
            self.permute()

        # compute pairwise distance 
        self.dist_mat = self.get_dist_mat()
    

    def construct_graph(self) -> nx.DiGraph:
        """ 
        construct a graph according to node positions and edges. Edges are weighted by euclidean distance 'dist' 
        :return a graph (tree)
        """
        g = nx.DiGraph()  # a directed one to start with
        g.add_edges_from(self.edges)

        # weight by dist 
        for node_1, node_2 in self.edges:
            cur_dist = np.linalg.norm(self.nodes_positions[node_1] - self.nodes_positions[node_2])
            g.edges[node_1, node_2]['dist'] = cur_dist
        
        return g
    

    def permute(self):
        """ permute nodes or edges according to the rules above """
        if self.permute_type == 'add_noise':
            self.add_noise()
        elif self.permute_type == 'change_edge':
            self.change_edge()
        else:
            raise NotImplementedError("Permutation Type only supports add_noise and change_edge")


    def add_noise(self):
        """ add noise """
        # prepare 
        num_nodes = self.g.number_of_nodes()
        num_nodes_to_permute = int(num_nodes * self.proportion_permute)
        selected_node_indices = np.random.choice(range(num_nodes), size=num_nodes_to_permute, replace=False)
        num_noise_per_node = np.random.randint(low=0, high=self.max_degree, size=num_nodes_to_permute) + 1

        # add gaussian noise 
        cur_noise_idx = num_nodes
        for selected_node_idx, num_noise_cur_node in zip(selected_node_indices, num_noise_per_node):
            selected_node = self.nodes_positions[selected_node_idx]
            gaussian_noises = np.random.normal(scale=0.5, size=(num_noise_cur_node, self.n))  # ! tune noise 
            noise_points = selected_node + gaussian_noises

            # add to nodes 
            self.nodes_positions = np.append(self.nodes_positions, noise_points, axis=0)

            # add to graph, serially add after num_nodes
            for noise_poiint in noise_points:
                self.edges.append((selected_node_idx, cur_noise_idx))
                self.g.add_edge(selected_node_idx, cur_noise_idx, dist=np.linalg.norm(selected_node - noise_poiint)) 
                cur_noise_idx += 1


    def change_edge(self):
        """ change edge, assuming that each node has only one parent """
        # prepare 
        num_nodes = self.g.number_of_nodes()
        num_nodes_to_permute = int(num_nodes * self.proportion_permute)
        selected_node_indices = np.random.choice(range(num_nodes), size=num_nodes_to_permute, replace=False)

        # change edges
        for selected_node_idx in selected_node_indices:
            parent_node_list = list(self.g.predecessors(selected_node_idx))

            # if no predecessor, skip 
            if len(parent_node_list) == 0:
                continue
            
            parent_node_idx = parent_node_list[0]  # assuming only one parent
            
            grandparent_node_list = list(self.g.predecessors(parent_node_idx))

            # if no grandparent, also skip 
            if len(grandparent_node_list) == 0:
                continue
                
            grandparent_node_idx = grandparent_node_list[0]

            # select other parents 
            other_parents = set(self.g.successors(grandparent_node_idx)) - set(parent_node_list)
            # if that was the only parent, also skip 
            if len(other_parents) == 0:
                continue
            
            selected_other_parent_idx = random.sample(other_parents, 1)[0]

            # remove and insert edges 
            self.g.remove_edge(parent_node_idx, selected_node_idx)
            self.g.add_edge(
                selected_other_parent_idx, selected_node_idx,
                dist=np.linalg.norm(self.nodes_positions[selected_other_parent_idx] - self.nodes_positions[selected_node_idx])
            )

            # update edge list 
            self.edges.remove((parent_node_idx, selected_node_idx))
            self.edges.append((selected_other_parent_idx, selected_node_idx))


    def get_dist_mat(self):
        """ compute pairwise distance and put into a matrix """
        g_undirected = self.g.to_undirected()
        num_nodes = g_undirected.number_of_nodes()
        idx_list = combinations(range(num_nodes), 2)

        dist_mat = np.zeros((num_nodes, num_nodes))
        for idx_1, idx_2 in idx_list:
            if self.use_path_length:
                cur_dist = nx.shortest_path_length(g_undirected, idx_1, idx_2)
            else:
                cur_dist = nx.shortest_path_length(g_undirected, idx_1, idx_2, weight='dist')
            dist_mat[idx_1, idx_2] = cur_dist
            dist_mat[idx_2, idx_1] = cur_dist
        
        return dist_mat
    

    def visualize(self):
        """ plot the plots """
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.scatter(self.nodes_positions[:, 0], self.nodes_positions[:, 1])
        for point_1_idx, point_2_idx in self.edges:
            point_1 = self.nodes_positions[point_1_idx]
            point_2 = self.nodes_positions[point_2_idx]
            
            x_values = [point_1[0], point_2[0]]
            y_values = [point_1[1], point_2[1]]

            plt.plot(x_values, y_values, 'grey', linewidth=0.5)
        
        
    def save_to_folder(self, folder_name):
        """ save euclidean coordinates, edge lists, dist_mat to the same folder """
        with open(os.path.join(folder_name, 'sim_tree_points.npy'), 'wb') as f:
            np.save(f, self.nodes_positions)
        with open(os.path.join(folder_name, 'sim_tree_dist_mat.npy'), 'wb') as f:
            np.save(f, self.dist_mat)
        with open(os.path.join(folder_name, 'sim_tree_edges.npy'), 'wb') as f:
            np.save(f, np.array(self.edges))
 
class SyntheticTreeDistortionDataSetFromFile(torch.utils.data.Dataset):
    """alternative to the above, read from file """
    def __init__(
        self, 
        folder_name: str
    ) -> None:
        self.path = folder_name

        # construct tree 
        self.sim_data_points, self.shortest_path_mat = self.read_tree_data()

        self.data = self.sim_data_points
        self.labels = np.array([range(len(self.data))]).T

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return torch.Tensor(data), torch.Tensor(label)

    # ----- util ------
    def read_tree_data(self):
        """read from file"""
        # read data points 
        with open(os.path.join('data', self.path, 'sim_tree_points.npy'), 'rb') as f:
            sim_data_points = np.load(f)
        
        # read mat 
        with open(os.path.join('data', self.path, 'sim_tree_dist_mat.npy'), 'rb') as f:
            shortest_path_mat = np.load(f)
        shortest_path_mat = torch.Tensor(shortest_path_mat)

        # return sim_data_points_dict, shortest_path_dict, shortest_path_mat
        return sim_data_points, shortest_path_mat
