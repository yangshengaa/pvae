"""
concrete synthetic dataset method, including 
- parent class: handling IO and refresh statistics 
- subclasses: each of the dataset 
"""

# load packages 
import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from scipy.spatial.distance import cdist

import networkx as nx 

import warnings
warnings.filterwarnings('ignore')

# load file 
from .make_synthetic_dataset import SyntheticTreeDistortionDataSetPermute, SyntheticTreeDistortionDataSet

# data path 
DATA_PATH = 'data'

# ================ parent class =================
class ConcreteSyntheticDatasetParent:
    """ 
    parent of all concrete simulated dataset 
    
    call make_dataset, then call make
    """

    def __init__(self, index, **kwargs):
        
        self.folder_name = f'sim_tree_{index}'  # save to sim_tree_{index}
        folder_path = os.path.join(DATA_PATH, self.folder_name)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        self.kwargs = kwargs
        
    def make_dataset(self):
        """ concrete dataset making procedure """
        raise NotImplementedError()
    
    @staticmethod 
    def compute_stats(g, nodes_positions, dist_mat):
        """ compute the stats listed below """
        number_of_nodes = g.number_of_nodes()
        number_of_edges = g.number_of_edges()
        degree_arr = np.array([x[1] for x in list(g.degree)])
        max_degree = np.max(degree_arr)
        mean_degree = np.mean(degree_arr)
        std_degree = np.std(degree_arr)
        
        emb_dist_mat = ConcreteSyntheticDatasetParent._euclidean_distance(nodes_positions)
        average_distortion = ConcreteSyntheticDatasetParent._average_distortion(emb_dist_mat, dist_mat)
        individual_distortion = ConcreteSyntheticDatasetParent._individual_distortion(emb_dist_mat, dist_mat)
        max_distortion = ConcreteSyntheticDatasetParent._max_distortion(emb_dist_mat, dist_mat)
        contraction_std = ConcreteSyntheticDatasetParent._contraction_std(emb_dist_mat, dist_mat)
        expansion_std = ConcreteSyntheticDatasetParent._expansion_std(emb_dist_mat, dist_mat)

        col_names = [
            'number_of_nodes', 'number_of_edges', 'max_degree', 'mean_degree', 'std_degree',
            'average_distortion', 'individual_distortion', 'max_distortion', 'contraction_std', 'expansion_std'
        ]
        entries = [[
            number_of_nodes, number_of_edges, max_degree, mean_degree, std_degree,
            average_distortion, individual_distortion, max_distortion, contraction_std, expansion_std
        ]]
        cur_stats_df = pd.DataFrame(entries, columns=col_names)
        return cur_stats_df
    
    def refresh_stats(self):
        """ 
        update stats on each dataset, including 
        - number of nodes
        - number of edges 
        - max degree 
        - mean degree 
        - std degree 
        - average distortion
        - individual distortion 
        - max distortion 
        - contraction std 
        - expansion std 
        """
        nodes_positions = self.dataset.nodes_positions
        dist_mat = self.dataset.dist_mat 
        g = self.dataset.g

        # compute stats 
        cur_stats_df = self.compute_stats(g, nodes_positions, dist_mat)
        cur_stats_df.index = [self.folder_name]

        # read and append 
        stats_file_name = os.path.join(DATA_PATH, 'data_stats.csv')
        try:
            stats_df = pd.read_csv(stats_file_name, index_col=0)
            # new_stats_df = pd.concat([stats_df, cur_stats_df])
            stats_df.loc[self.folder_name] = cur_stats_df.loc[self.folder_name]

            # sort on names
            dataset_names = list(stats_df.index)
            dataset_names_sorted = sorted(dataset_names, key=lambda x: int(x.split('_')[2]))
            stats_df_sorted = stats_df.loc[dataset_names_sorted]

            # drop duplicates
            stats_df_sorted = stats_df_sorted[~stats_df_sorted.index.duplicated(keep='last')]

            # write to file 
            stats_df_sorted.to_csv(stats_file_name)
        except:  # if no file exists yet 
            cur_stats_df.to_csv(stats_file_name)

    def make(self):
        """ make dataset and store all necessary files """
        # save dataset
        self.dataset.save_to_folder(os.path.join(DATA_PATH, self.folder_name))
        # save vis 
        self.dataset.visualize()
        plt.savefig(os.path.join(DATA_PATH, self.folder_name, 'sim_tree_vis.png'), dpi=100)
        # refresh stats
        self.refresh_stats()
    
    @staticmethod
    def force_refresh_all():
        """ refresh and update all currently available statistics """
        # load stats df 
        stats_df = pd.read_csv(os.path.join(DATA_PATH, 'data_stats.csv'), index_col=0)
        # loop through and compute 
        dataset_folders = [dir_name for dir_name in os.listdir(DATA_PATH) if 'sim_tree' in dir_name]
        new_stats_df_list = []
        for dataset_folder in dataset_folders: 
            dataset_path = os.path.join(DATA_PATH, dataset_folder)
            # compute only if all necessary ingredients are loaded
            try:
                with open(os.path.join(dataset_path, 'sim_tree_points.npy'), 'rb') as f:
                    nodes_positions = np.load(f)
                with open(os.path.join(dataset_path, 'sim_tree_edges.npy'), 'rb') as f:
                    edges = np.load(f)
                with open(os.path.join(dataset_path, 'sim_tree_dist_mat.npy'), 'rb') as f:
                    dist_mat = np.load(f)
                
                g = nx.Graph()
                g.add_edges_from(edges)

                # compute stats 
                cur_stats_df = ConcreteSyntheticDatasetParent.compute_stats(g, nodes_positions, dist_mat)
                cur_stats_df.index = [dataset_folder]

                # append to list 
                new_stats_df_list.append(cur_stats_df)

            except:
                pass 
        
        # put back and flush 
        new_stats_df = pd.concat([stats_df, *new_stats_df_list])
        index_list = list(new_stats_df.index)
        index_list_sorted = sorted(index_list, key=lambda x: int(x.split('_')[2]))
        new_stats_df_sorted = new_stats_df.loc[index_list_sorted]

        # drop duplicates 
        new_stats_df_sorted = new_stats_df_sorted[~new_stats_df_sorted.index.duplicated(keep='last')]
        new_stats_df_sorted.to_csv(os.path.join(DATA_PATH, 'data_stats.csv'))
    

    # ---------- utils ----------
    @staticmethod
    def _euclidean_distance(node_positions: np.ndarray):
        """ compute the euclidean distance """
        euc_dist_mat = cdist(node_positions, node_positions, metric='euclidean')
        return euc_dist_mat

    @staticmethod
    def _average_distortion(emb_dist: np.ndarray, real_dist: np.ndarray):
        """ compute the average distortion """
        n = emb_dist.shape[0]
        # mask diag
        mask_arr = np.zeros((n, n)) / (1 - np.eye(n))
        emb_dist_masked = emb_dist + mask_arr 
        real_dist_masked = real_dist + mask_arr 

        # compute distortion 
        contraction = real_dist_masked / emb_dist_masked
        expansion = emb_dist_masked / real_dist_masked
        distortion = np.nanmean(contraction) * np.nanmean(expansion)

        return distortion 

    @staticmethod 
    def _individual_distortion(emb_dist: np.ndarray, real_dist: np.ndarray):
        """ compute the individual distortion """
        n = emb_dist.shape[0]
        # mask diag
        mask_arr = np.zeros((n, n)) / (1 - np.eye(n))
        emb_dist_masked = emb_dist + mask_arr
        real_dist_masked = real_dist + mask_arr

        # compute distortion
        contraction = real_dist_masked / emb_dist_masked
        expansion = emb_dist_masked / real_dist_masked
        individual_contraction =  np.nansum(contraction, axis=1) / (n - 1)
        individual_expansion = np.nansum(expansion, axis=1) / (n - 1)
        individual_distortion = np.mean(individual_contraction * individual_expansion)

        return individual_distortion

    @staticmethod 
    def _max_distortion(emb_dist: np.ndarray, real_dist: np.ndarray):
        """ compute the max distortion """
        n = emb_dist.shape[0]
        # mask diag
        mask_arr = np.zeros((n, n)) / (1 - np.eye(n))
        emb_dist_masked = emb_dist + mask_arr
        real_dist_masked = real_dist + mask_arr

        # compute distortion
        contraction = real_dist_masked / emb_dist_masked
        expansion = emb_dist_masked / real_dist_masked
        max_contraction = np.nanmax(contraction)
        max_expansion = np.nanmax(expansion)
        max_distortion = max_contraction * max_expansion

        return max_distortion

    @staticmethod 
    def _contraction_std(emb_dist: np.ndarray, real_dist: np.ndarray):
        """ compute contraction """
        n = emb_dist.shape[0]
        # mask diag
        mask_arr = np.zeros((n, n)) / (1 - np.eye(n))
        emb_dist_masked = emb_dist + mask_arr
        real_dist_masked = real_dist + mask_arr

        # compute contraction std 
        contraction = real_dist_masked / emb_dist_masked
        contraction_std = np.nanstd(contraction.flatten())
    
        return contraction_std

    @staticmethod
    def _expansion_std(emb_dist: np.ndarray, real_dist: np.ndarray):
        """ compute contraction """
        n = emb_dist.shape[0]
        # mask diag
        mask_arr = np.zeros((n, n)) / (1 - np.eye(n))
        emb_dist_masked = emb_dist + mask_arr
        real_dist_masked = real_dist + mask_arr

        # compute contraction std
        expansion = emb_dist_masked / real_dist_masked
        expansion_std = np.nanstd(expansion.flatten())

        return expansion_std

    
# ===========================================================
# ========================= subclasses ======================
# ===========================================================

# ----------------------- dataset 0 -------------------------
class simple_case(ConcreteSyntheticDatasetParent):
    """ a simple four point dataset """
    def make_dataset(self):
        
        nodes_positions = np.array([[0, 0], [0.5, 0], [-1/4, np.sqrt(3) / 4], [-1/4, -np.sqrt(3) / 4]])
        edges = np.array([[0, 1], [0, 2], [0, 3]])

        self.dataset = SyntheticTreeDistortionDataSetPermute(
            nodes_positions, 
            edges, 
            to_permute=self.kwargs['to_permute'], 
            use_path_length=self.kwargs['use_path_length']
        )

class single_dandelion(ConcreteSyntheticDatasetParent):
    """ a single dandelion, centered at 0 """
    def make_dataset(self):
        num_nodes = 50

        edges = []
        center_idx_list = []
        nodes_positions = []

        # center 1 
        thetas = np.linspace(0, 2 * np.pi, num_nodes)
        rhos = np.random.uniform(low=7, high=10, size=num_nodes)

        center = np.array([0, 0])
        nodes = np.vstack((rhos * np.cos(thetas), rhos * np.sin(thetas))).T  + center

        cur_idx = 0
        center_idx = cur_idx
        center_idx_list.append(center_idx)

        nodes_positions.append(np.array([center]))

        for node in nodes:
            num_points_along_edge = np.random.randint(low=4, high=7)
            inner_points = np.linspace(center, node, num_points_along_edge, axis=0)

            nodes_positions.append(inner_points[1:]) # excluding 0 

            cur_idx += 1
            edges.append((center_idx, cur_idx))
            for i in range(1, len(inner_points) - 1):
                edges.append((cur_idx, cur_idx + 1))
                cur_idx += 1

        nodes_positions = np.vstack(nodes_positions)

        # make dataset
        self.dataset = SyntheticTreeDistortionDataSetPermute(
            nodes_positions, edges, 
            to_permute=self.kwargs['to_permute'], 
            proportion_permute=self.kwargs['proportion_permute'],
            max_degree=self.kwargs['max_degree']
        )


class dandelion_mixture(ConcreteSyntheticDatasetParent):
    """ mixture of dandelions """
    def make_dataset(self):

        # make a circle of points
        num_nodes = 25

        edges = []
        center_idx_list = []
        nodes_positions = []

        # center 1
        thetas = np.linspace(0, 2 * np.pi, num_nodes)
        rhos = np.random.uniform(low=2, high=5, size=num_nodes)

        center = np.array([6, 5])
        nodes = np.vstack((rhos * np.cos(thetas), rhos * np.sin(thetas))).T + center

        cur_idx = 0
        center_idx = cur_idx
        center_idx_list.append(center_idx)


        nodes_positions.append(np.array([center]))

        for node in nodes:
            num_points_along_edge = np.random.randint(low=4, high=7)
            inner_points = np.linspace(center, node, num_points_along_edge, axis=0)

            nodes_positions.append(inner_points[1:])  # excluding 0

            cur_idx += 1
            edges.append((center_idx, cur_idx))
            for i in range(1, len(inner_points) - 1):
                edges.append((cur_idx, cur_idx + 1))
                cur_idx += 1


        # center 2
        thetas = np.linspace(0, 2 * np.pi, num_nodes)
        rhos = np.random.uniform(low=2, high=5, size=num_nodes)

        center = np.array([-4, 4])
        nodes = np.vstack((rhos * np.cos(thetas), rhos * np.sin(thetas))).T + center

        cur_idx += 1
        center_idx = cur_idx
        center_idx_list.append(center_idx)

        nodes_positions.append(np.array([center]))

        for node in nodes:
            num_points_along_edge = np.random.randint(low=4, high=7)
            inner_points = np.linspace(center, node, num_points_along_edge, axis=0)

            nodes_positions.append(inner_points[1:])  # excluding 0

            cur_idx += 1
            edges.append((center_idx, cur_idx))
            for i in range(1, len(inner_points) - 1):
                edges.append((cur_idx, cur_idx + 1))
                cur_idx += 1


        # center 3
        thetas = np.linspace(0, 2 * np.pi, num_nodes)
        rhos = np.random.uniform(low=2, high=6, size=num_nodes)

        center = np.array([1, -8])
        nodes = np.vstack((rhos * np.cos(thetas), rhos * np.sin(thetas))).T + center

        cur_idx += 1
        center_idx = cur_idx
        center_idx_list.append(center_idx)


        nodes_positions.append(np.array([center]))

        for node in nodes:
            num_points_along_edge = np.random.randint(low=4, high=7)
            inner_points = np.linspace(center, node, num_points_along_edge, axis=0)

            nodes_positions.append(inner_points[1:])  # excluding 0

            cur_idx += 1
            edges.append((center_idx, cur_idx))
            for i in range(1, len(inner_points) - 1):
                edges.append((cur_idx, cur_idx + 1))
                cur_idx += 1

        # nodes_positions = np.vstack(nodes_positions)

        # center 4
        thetas = np.linspace(0, 2 * np.pi, num_nodes)
        rhos = np.random.uniform(low=2, high=8, size=num_nodes)

        center = np.array([-10, -12])
        nodes = np.vstack((rhos * np.cos(thetas), rhos * np.sin(thetas))).T + center

        cur_idx += 1
        center_idx = cur_idx
        center_idx_list.append(center_idx)


        nodes_positions.append(np.array([center]))

        for node in nodes:
            num_points_along_edge = np.random.randint(low=4, high=7)
            inner_points = np.linspace(center, node, num_points_along_edge, axis=0)

            nodes_positions.append(inner_points[1:])  # excluding 0

            cur_idx += 1
            edges.append((center_idx, cur_idx))
            for i in range(1, len(inner_points) - 1):
                edges.append((cur_idx, cur_idx + 1))
                cur_idx += 1

        # manually add nodes
        nodes_positions.append(np.array([
            [-6, -3],
            [-0, -1],
            [10, 0]
        ]))

        edges.append((cur_idx + 1, center_idx_list[2]))
        edges.append((cur_idx + 1, center_idx_list[3]))
        edges.append((cur_idx + 1, cur_idx + 2))
        edges.append((cur_idx + 2, cur_idx + 3))
        edges.append((cur_idx + 2, center_idx_list[1]))
        edges.append((cur_idx + 3, center_idx_list[0]))

        nodes_positions = np.vstack(nodes_positions)

        # test dataset 
        self.dataset = SyntheticTreeDistortionDataSetPermute(
            nodes_positions, edges, 
            to_permute=self.kwargs['to_permute'], 
            proportion_permute=self.kwargs['proprotion_permute'], 
            max_degree=self.kwargs['max_degree']
        )


class random_rootless_tree(ConcreteSyntheticDatasetParent):
    """ a rootless random tree, constructed using random scatter followed by kruskal """
    def make_dataset(self):

        # random data points
        dataset = SyntheticTreeDistortionDataSet(2, 300, 10)
        data = dataset.centroids * 10


        edges = dataset.edges

        self.dataset = SyntheticTreeDistortionDataSetPermute(
            data, edges, 
            to_permute=self.kwargs['to_permute'], 
            proportion_permute=self.kwargs['proprotion_permute'], 
            max_degree=self.kwargs['max_degree']
        )

class explicit_tree(ConcreteSyntheticDatasetParent):
    """ an explicit tree dataset """
    def make_dataset(self):
        tree_depth = 10
        root = np.array([0, tree_depth])
        max_depth = tree_depth

        cur_idx = 0
        nodes_positions = [root]
        parent_queue = [0]
        children_queue = []
        edges = []
        cur_depth_num_nodes = 1
        for depth in range(max_depth):
            num_children = np.random.poisson(1.5, size=cur_depth_num_nodes)
            children_positions = np.linspace(
                np.array([-tree_depth, tree_depth - depth - 1]), 
                np.array([tree_depth, tree_depth - depth - 1]), 
                sum(num_children) + 2
            )[1:-1]  # middle ones 
            cur_depth_num_nodes = len(children_positions)
            # print(num_children, children_positions)

            # put in queue 
            for children_position in children_positions: 
                nodes_positions.append([children_position])
                cur_idx += 1
                children_queue.append(cur_idx)
            
            # connect 
            for cur_num_children in num_children: 
                cur_parent = parent_queue.pop(0)
                for i in range(cur_num_children):
                    cur_children = children_queue.pop(0)
                    edges.append((cur_parent, cur_children))
                    parent_queue.append(cur_children)

        nodes_positions = np.vstack(nodes_positions)

        self.dataset = SyntheticTreeDistortionDataSetPermute(
            nodes_positions, edges, 
            to_permute=self.kwargs['to_permute'], 
            permute_type=self.kwargs['permute_type'], 
            use_path_length=self.kwargs['use_path_legnth'], 
            proportion_permute=self.kwargs['proportion_permute']
        )
