import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from numpy import prod
from pvae.utils import Constants
from pvae.ops.manifold_layers import (
    GeodesicLayer, GeodesicLayerExpMap0, GeodesicLayerSinhAlt, GeodesicLayerAlt,
    HyperbolicLayerWrapped, HyperbolicLayerWrappedAlt, HyperbolicLayerWrappedSinhAlt,
    MobiusLayer, LogZero, ExpZero
)

from pvae import manifolds

def extra_hidden_layer(hidden_dim, non_lin):
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), non_lin)


class EncLinear(nn.Module):
    """ Usual encoder """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, prior_iso, no_bn):
        super(EncLinear, self).__init__()
        self.manifold = manifold
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(nn.Linear(prod(data_size), hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, manifold.coord_dim)
        self.fc22 = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)

        # batch normalization
        if not no_bn:
            self.bn = nn.BatchNorm1d(manifold.coord_dim)
        else:
            self.bn = nn.Identity()

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-len(self.data_size)], -1))
        mu = self.fc21(e)          # flatten data
        mu = self.bn(mu)
        return mu, F.softplus(self.fc22(e)) + Constants.eta,  self.manifold


class DecLinear(nn.Module):
    """ Usual decoder """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecLinear, self).__init__()
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(nn.Linear(manifold.coord_dim, hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, prod(data_size))

    def forward(self, z):
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)  # reshape data
        return mu, torch.ones_like(mu)


class EncWrapped(nn.Module):
    """ Usual encoder followed by an exponential map """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, prior_iso, no_bn):
        super(EncWrapped, self).__init__()
        self.manifold = manifold
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(nn.Linear(prod(data_size), hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, manifold.coord_dim)
        self.fc22 = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)

        # batch norm
        if not no_bn:
            self.bn = nn.BatchNorm1d(manifold.coord_dim)
        else:
            self.bn = nn.Identity()

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-len(self.data_size)], -1))
        mu = self.fc21(e)          # flatten data
        mu = self.bn(mu)           # batchnorm before lifting 
        mu = self.manifold.expmap0(mu)
        return mu, F.softplus(self.fc22(e)) + Constants.eta,  self.manifold


class DecWrapped(nn.Module):
    """ Usual encoder preceded by a logarithm map """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecWrapped, self).__init__()
        self.data_size = data_size
        self.manifold = manifold
        modules = []
        modules.append(nn.Sequential(nn.Linear(manifold.coord_dim, hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, prod(data_size))

    def forward(self, z):
        z = self.manifold.logmap0(z)
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)  # reshape data
        return mu, torch.ones_like(mu)

class EncWrappedNaive(nn.Module):
    """ immediate lifting """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, prior_iso, no_bn):
        super(EncWrappedNaive, self).__init__()
        self.manifold = manifold
        self.data_size = data_size
        self.initalized = False

        # batch norm
        if not no_bn:
            self.bn = nn.BatchNorm1d(manifold.coord_dim)
        else:
            self.bn = nn.Identity()
    
    def forward(self, x):
        # x = x.view(*x.size()[:-len(self.data_size)], -1)
        # init 
        if not self.initalized:
            num_obs, dim = x.shape
            self.embeddings = nn.Embedding(num_obs, dim)
            self.index_tensor = torch.LongTensor(range(num_obs))
            self.embeddings.weight = nn.Parameter(x)  # init weights of embeddings 
            self.initalized = True 
        
        # forward 
        mu = self.embeddings(self.index_tensor)
        # print(mu)
        mu = self.bn(mu)
        mu = self.manifold.expmap0(mu)
        return mu, 0, 0


class EncWrappedAlt(nn.Module):
    """ alternative encoder of EncWrapped: MLP folled by an custom map """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, prior_iso, no_bn):
        super(EncWrappedAlt, self).__init__()
        self.manifold = manifold
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(nn.Linear(prod(data_size), hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, manifold.coord_dim)
        self.fc22 = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)

        # batch norm
        if not no_bn:
            self.bn = nn.BatchNorm1d(manifold.coord_dim)
        else:
            self.bn = nn.Identity()

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-len(self.data_size)], -1))
        mu = self.fc21(e)                   # flatten data
        mu = self.bn(mu)                    # batchnormalization before lifting 
        mu = self.manifold.direct_map(mu)   # the only difference with EncWrapped
        return mu, F.softplus(self.fc22(e)) + Constants.eta,  self.manifold


class EncWrappedSinhAlt(nn.Module):
    """ alternative encoder of EncWrapped: MLP followed by sinh and a direct map """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, prior_iso, no_bn):
        super(EncWrappedSinhAlt, self).__init__()
        self.manifold = manifold
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(nn.Linear(prod(data_size), hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, manifold.coord_dim)
        self.fc22 = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)

        # batch norm 
        if not no_bn:
            self.bn = nn.BatchNorm1d(manifold.coord_dim)
        else:
            self.bn = nn.Identity()

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-len(self.data_size)], -1))
        mu = self.fc21(e)                        # flatten data
        mu = self.manifold.sinh_direct_map(mu)   # the only difference with EncWrapped
        return mu, F.softplus(self.fc22(e)) + Constants.eta,  self.manifold


class EncMixture(nn.Module):
    """ mixing euclidean with hyperbolic """
    def __init__(
            self, 
            manifold_type, 
            data_size, non_lin, hidden_dims, num_hyperbolic_layers, latent_dim, 
            c,
            no_final_lift, lift_type,
            no_bn
        ):
        super(EncMixture, self).__init__()
        self.manifold_type = manifold_type
        self.data_size = data_size
        self.non_lin = non_lin
        self.c = c
        self.no_final_lift = no_final_lift
        self.lift_type = lift_type
        self.hidden_dims = hidden_dims
        self.dims_list = [prod(data_size), *self.hidden_dims, latent_dim]
        self.num_hyperbolic_layers = num_hyperbolic_layers
        self.num_euclidean_layers = len(hidden_dims) + 1 - self.num_hyperbolic_layers

        # construct two sets of layers 
        self.euclidean_layers, self.bridge_map, self.hyperbolic_layers = self.get_layers()

        # batchnormalization 
        if not no_bn:
            self.bn = nn.BatchNorm1d(self.dims_list[self.num_euclidean_layers])
        else:
            self.bn = nn.Identity()
    
    def get_layers(self):
        """ create euclidean and hyperbolic layers """
        k, l = self.num_euclidean_layers, self.num_hyperbolic_layers

        euclidean_layers_list, hyperbolic_layers_list = [], []

        # construct euclidean layers 
        for i in range(k):
            euclidean_layers_list.append(nn.Linear(self.dims_list[i], self.dims_list[i + 1]))
            if i < k - 1:
                # no non_lin at the final layer, or before bridge
                euclidean_layers_list.append(self.non_lin)  
        
        # construct bridging map 
        bridge_manifold = getattr(manifolds, self.manifold_type)(self.dims_list[k], self.c)
        if self.lift_type == 'expmap': 
            bridge_map = bridge_manifold.expmap0
        elif self.lift_type == 'direct':
            bridge_map = bridge_manifold.direct_map
        elif self.lift_type == 'sinh_direct':
            bridge_map = bridge_manifold.sinh_direct_map
        elif self.no_final_lift and l == 0:  # no hyperbolic layer and no lifting at the end
            bridge_map == nn.Identity()
        else:
            raise NotImplementedError(f'lifting type {self.lift_type} not supported')

        # select hyperbolic layer 
        if self.lift_type == 'expmap':
            hyperbolic_layer = HyperbolicLayerWrapped
        elif self.lift_type == 'direct':
            hyperbolic_layer = HyperbolicLayerWrappedAlt
        elif self.lift_type == 'sinh_direct':
            hyperbolic_layer = HyperbolicLayerWrappedSinhAlt
        # construct hyperbolic layers
        for i in range(k, k + l):
            cur_dim = self.dims_list[i]
            cur_manifold = getattr(manifolds, self.manifold_type)(cur_dim, self.c)
            hyperbolic_layers_list.append(hyperbolic_layer(cur_manifold.coord_dim, self.dims_list[i + 1], cur_manifold))
        
        # final packing 
        euclidean_layers = nn.Sequential(*euclidean_layers_list)
        hyperbolic_layers = nn.Sequential(*hyperbolic_layers_list)
        
        return euclidean_layers, bridge_map, hyperbolic_layers
    
    def forward(self, x):
        euclidean_out = self.euclidean_layers(x.view(*x.size()[:-len(self.data_size)], -1))
        euclidean_out = self.bn(euclidean_out) # bn before lifting 
        lifted_out = self.bridge_map(euclidean_out)
        hyperbolic_out = self.hyperbolic_layers(lifted_out)
        return hyperbolic_out, None, None


class DecGeo(nn.Module):
    """ First layer is a Hypergyroplane followed by usual decoder """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecGeo, self).__init__()
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(GeodesicLayer(manifold.coord_dim, hidden_dim, manifold), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, prod(data_size))

    def forward(self, z):
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)  # reshape data
        return mu, torch.ones_like(mu)


class EncMob(nn.Module):
    """ Last layer is a Mobius layers """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, prior_iso):
        super(EncMob, self).__init__()
        self.manifold = manifold
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(nn.Linear(prod(data_size), hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = MobiusLayer(hidden_dim, manifold.coord_dim, manifold)
        self.fc22 = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-len(self.data_size)], -1))            # flatten data
        mu = self.fc21(e)          # flatten data
        mu = self.manifold.expmap0(mu)
        return mu, F.softplus(self.fc22(e)) + Constants.eta,  self.manifold


class DecMob(nn.Module):
    """ First layer is a Mobius Matrix multiplication """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecMob, self).__init__()
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(MobiusLayer(manifold.coord_dim, hidden_dim, manifold), LogZero(manifold), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, prod(data_size))

    def forward(self, z):
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)  # reshape data
        return mu, torch.ones_like(mu)


class DecBernouilliWrapper(nn.Module):
    """ Wrapper for Bernoulli likelihood """
    def __init__(self, dec):
        super(DecBernouilliWrapper, self).__init__()
        self.dec = dec

    def forward(self, z):
        mu, _ = self.dec.forward(z)
        return torch.tensor(1.0).to(z.device), mu

# =============== for simulation only ================
# all accepts an additional argument: output-dim, to generalize to other output dimensions
class DecLinearSim(DecLinear):
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, output_dim):
        super().__init__(manifold, data_size, non_lin, num_hidden_layers, hidden_dim)
        self.output_dim = output_dim
        self.fc31 = nn.Linear(hidden_dim, prod(output_dim))  # replace the original decoder dim
    
    def forward(self, z):
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], self.output_dim)  # reshape data
        return mu, torch.ones_like(mu)

class DecWrappedSim(DecWrapped):
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, output_dim):
        super().__init__(manifold, data_size, non_lin, num_hidden_layers, hidden_dim)
        self.output_dim = output_dim 
        self.fc31 = nn.Linear(hidden_dim, prod(output_dim))
    
    def forward(self, z):
        z = self.manifold.logmap0(z)
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], self.output_dim)  # reshape data
        return mu, torch.ones_like(mu)

class DecGeoSim(DecGeo):
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, output_dim):
        super().__init__(manifold, data_size, non_lin, num_hidden_layers, hidden_dim)
        self.output_dim = output_dim
        self.fc31 = nn.Linear(hidden_dim, prod(output_dim))
    
    def forward(self, z):
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], self.output_dim)  # reshape data
        return mu, torch.ones_like(mu)

class DecMobSim(DecMob):
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, output_dim):
        super().__init__(manifold, data_size, non_lin, num_hidden_layers, hidden_dim)
        self.output_dim = output_dim 
        self.fc31 = nn.Linear(hidden_dim, prod(output_dim))
    
    def forward(self, z):
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], self.output_dim)  # reshape data
        return mu, torch.ones_like(mu)
