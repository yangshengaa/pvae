import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import DataLoader

import math
from sklearn.model_selection._split import _validate_shuffle_split
from .vae import VAE
from .ae import AE
from pvae.vis import array_plot

from pvae.distributions import RiemannianNormal, WrappedNormal
from torch.distributions import Normal
from pvae import manifolds
from .architectures import *
from pvae.datasets import SyntheticDataset, CSVDataset, SyntheticTreeDistortionDataSet


class Tabular(VAE):
    """ Derive a specific sub-class of a VAE for tabular data. """
    def __init__(self, params):
        c = nn.Parameter(params.c * torch.ones(1), requires_grad=False)
        manifold = getattr(manifolds, params.manifold)(params.latent_dim, c)
        super(Tabular, self).__init__(
            eval(params.prior),           # prior distribution
            eval(params.posterior),       # posterior distribution
            dist.Normal,                  # likelihood distribution
            eval('Enc' + params.enc)(manifold, params.data_size, getattr(nn, params.nl)(), params.num_hidden_layers, params.hidden_dim, params.prior_iso),
            eval('Dec' + params.dec)(manifold, params.data_size, getattr(nn, params.nl)(), params.num_hidden_layers, params.hidden_dim),
            params
        )
        self.manifold = manifold
        self._pz_mu = nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False)
        self._pz_logvar = nn.Parameter(torch.zeros(1, 1), requires_grad=params.learn_prior_std)
        self.modelName = 'Tabular'

    @property
    def pz_params(self):
        return self._pz_mu.mul(1), F.softplus(self._pz_logvar).div(math.log(2)).mul(self.prior_std), self.manifold

    def generate(self, runPath, epoch):
        N, K = 10, 1
        _, _, samples = super(Tabular, self).generate(N, K)
        array_plot([samples.data.cpu()], '{}/gen_samples_{:03d}.png'.format(runPath, epoch))

    def reconstruct(self, data, runPath, epoch):
        recon = super(Tabular, self).reconstruct(data)
        array_plot([data.data.cpu(), recon.data.cpu()], '{}/reconstruct_{:03d}.png'.format(runPath, epoch))


class Tree(Tabular):
    """ Derive a specific sub-class of a VAE for tree data. """
    def __init__(self, params):
        super(Tree, self).__init__(params)

    def getDataLoaders(self, batch_size, shuffle, device, *args):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        print('Load training data...')
        dataset = SyntheticDataset(*self.data_size, *map(lambda x: float(x), args))
        n_train, n_test = _validate_shuffle_split(len(dataset), test_size=None, train_size=0.7)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False, **kwargs)
        return train_loader, test_loader


class CSV(Tabular):
    """ Derive a specific sub-class of a VAE for tabular data loaded via a cvs file. """
    def __init__(self, params):
        super(CSV, self).__init__(params)

    def getDataLoaders(self, batch_size, shuffle, device, *args):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        print('Load training data...')
        dataset = CSVDataset(*args)
        n_train, n_test = _validate_shuffle_split(len(dataset), test_size=None, train_size=0.7)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False, **kwargs)
        return train_loader, test_loader

# ========== for simulation only ===========

class TabularAE(AE):
    """ tabular, but vanilla autoencoder version """
    def __init__(self, params):
        c = nn.Parameter(params.c * torch.ones(1), requires_grad=False)
        manifold = getattr(manifolds, params.manifold)(params.latent_dim, c)
        if 'Sim' in params.dec: # if in the simulation context, specify an extra argument for output dimension 
            super(TabularAE, self).__init__(
                eval('Enc' + params.enc)(manifold, params.data_size, getattr(nn, params.nl)(), params.num_hidden_layers, params.hidden_dim, params.prior_iso),
                eval('Dec' + params.dec)(manifold, params.data_size, getattr(nn, params.nl)(), params.num_hidden_layers, params.hidden_dim, params.output_dim),
                params
            )
        else:
            super(TabularAE, self).__init__(
                eval('Enc' + params.enc)(manifold, params.data_size, getattr(nn, params.nl)(), params.num_hidden_layers, params.hidden_dim, params.prior_iso),
                eval('Dec' + params.dec)(manifold, params.data_size, getattr(nn, params.nl)(), params.num_hidden_layers, params.hidden_dim),
                params
            )
        self.manifold = manifold
        self.modelName = 'TabularAE'

        
class SimTreeDistortion(TabularAE):
    """ derive a specific sub-class of a VAE for custom synthetic tree data for distortion exploration """
    def __init__(self, params):
        super(SimTreeDistortion, self).__init__(params)
    
    def getDataLoaders(self, batch_size, shuffle, device, *args):
        """ return an additional shortest path dict for loss tuning """
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        print('Load training data...')
        dataset = SyntheticTreeDistortionDataSet(*self.data_size, *map(lambda x: int(x), args))  # TODO: accept str for path
        n_train, n_test = _validate_shuffle_split(len(dataset), test_size=None, train_size=0.7)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])
        # full batch training to account for pairwise comparison 
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), drop_last=False, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), drop_last=False, shuffle=False, **kwargs)
        overall_loader = DataLoader(dataset, batch_size=len(dataset), drop_last=False, shuffle=False, **kwargs)  # for overall distortion
        return train_loader, test_loader, overall_loader, dataset.shortest_path_dict
