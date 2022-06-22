import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from pvae.manifolds import PoincareBall, Euclidean
from geoopt import ManifoldParameter

# init params 
INIT_MEAN = 0
INIT_STD = 0.5


class RiemannianLayer(nn.Module):
    def __init__(self, in_features, out_features, manifold, over_param, weight_norm):
        super(RiemannianLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold

        self._weight = Parameter(nn.init.normal_(torch.Tensor(out_features, in_features), mean=INIT_MEAN, std=INIT_STD))
        # self._weight = ManifoldParameter(nn.init.normal_(torch.Tensor(out_features, in_features), mean=INIT_MEAN, std=INIT_STD))
        self.over_param = over_param
        self.weight_norm = weight_norm
        if self.over_param:
            self._bias = ManifoldParameter(torch.Tensor(out_features, in_features), manifold=manifold)
        else:
            self._bias = Parameter(nn.init.normal_(torch.Tensor(out_features, 1), mean=INIT_MEAN, std=INIT_STD))
            # self._bias = ManifoldParameter(nn.init.normal_(torch.Tensor(out_features, 1), mean=INIT_MEAN, std=INIT_STD))
        self.reset_parameters()

    @property
    def weight(self):
        return self.manifold.transp0(self.bias, self._weight) # weight \in T_0 => weight \in T_bias

    @property
    def weight_expmap0(self):
        weight = self.manifold.expmap0(self._weight)
        return self.manifold.transp0(self.bias, weight)
    
    @property
    def weight_alt(self):
        weight = self.manifold.direct_map(self._weight)
        return self.manifold.transp0(self.bias, weight)
    
    @property
    def weight_sinhalt(self):
        weight = self.manifold.sinh_direct_map(self._weight)
        return self.manifold.transp0(self.bias, weight)

    @property
    def bias(self):
        if self.over_param:
            return self._bias
        else:
            return self.manifold.expmap0(self._weight * self._bias) # reparameterisation of a point on the manifold

    def reset_parameters(self):
        init.kaiming_normal_(self._weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self._weight)
        bound = 4 / math.sqrt(fan_in)
        init.uniform_(self._bias, -bound, bound)
        if self.over_param:
            with torch.no_grad(): self._bias.set_(self.manifold.expmap0(self._bias))


class GeodesicLayer(RiemannianLayer):
    def __init__(self, in_features, out_features, manifold, over_param=False, weight_norm=False):
        super(GeodesicLayer, self).__init__(in_features, out_features, manifold, over_param, weight_norm)

    def forward(self, input):
        input = input.unsqueeze(-2).expand(*input.shape[:-(len(input.shape) - 2)], self.out_features, self.in_features)
        res = self.manifold.normdist2plane(input, self.bias, self.weight, signed=True, norm=self.weight_norm)
        return res

# === variations of geodesic layers to bypass riemannian adam
class GeodesicLayerExpMap0(RiemannianLayer):
    def __init__(self, in_features, out_features, manifold, over_param=False, weight_norm=False):
        super(GeodesicLayerExpMap0, self).__init__(in_features, out_features, manifold, over_param, weight_norm)

    def forward(self, input):
        input = input.unsqueeze(-2).expand(-1, self.out_features, self.in_features)
        res = self.manifold.normdist2plane(input, self.bias, self.weight_expmap0, signed=True, norm=self.weight_norm)
        return res

class GeodesicLayerAlt(RiemannianLayer):
    def __init__(self, in_features, out_features, manifold, over_param=False, weight_norm=False):
        super(GeodesicLayerAlt, self).__init__(in_features, out_features, manifold, over_param, weight_norm)

    def forward(self, input):
        input = input.unsqueeze(-2).expand(-1, self.out_features, self.in_features)
        res = self.manifold.normdist2plane(input, self.bias, self.weight_alt, signed=True, norm=self.weight_norm)
        return res

class GeodesicLayerSinhAlt(RiemannianLayer):
    def __init__(self, in_features, out_features, manifold, over_param=False, weight_norm=False):
        super(GeodesicLayerSinhAlt, self).__init__(in_features, out_features, manifold, over_param, weight_norm)

    def forward(self, input):
        input = input.unsqueeze(-2).expand(-1, self.out_features, self.in_features)
        res = self.manifold.normdist2plane(input, self.bias, self.weight_sinhalt, signed=True, norm=self.weight_norm)
        return res


class HyperbolicLayerWrapped(RiemannianLayer):
    """ hyperbolic layer: geodesic + sinh + direct map """
    def __init__(self, in_features, out_features, manifold, over_param=False, weight_norm=False):
        super(HyperbolicLayerWrapped, self).__init__(in_features, out_features, manifold, over_param, weight_norm)
        self.bn = nn.BatchNorm1d(out_features)
    
    def forward(self, input):
        input = input.unsqueeze(-2).expand(-1, self.out_features, self.in_features)  
        euclidean_features = self.manifold.normdist2plane(input, self.bias, self.weight_expmap0, signed=True, norm=self.weight_norm)
        euclidean_features_bn = self.bn(euclidean_features)
        res = self.manifold.expmap0(euclidean_features_bn)
        return res

class HyperbolicLayerWrappedAlt(RiemannianLayer):
    """ hyperbolic layer: geodesic + direct map """
    def __init__(self, in_features, out_features, manifold, over_param=False, weight_norm=False):
        super(HyperbolicLayerWrappedAlt, self).__init__(in_features, out_features, manifold, over_param, weight_norm)
        self.bn = nn.BatchNorm1d(out_features)
    
    def forward(self, input):
        input = input.unsqueeze(-2).expand(-1, self.out_features, self.in_features)  
        euclidean_features = self.manifold.normdist2plane(input, self.bias, self.weight_alt, signed=True, norm=self.weight_norm)
        euclidean_features_bn = self.bn(euclidean_features)
        res = self.manifold.direct_map(euclidean_features_bn)
        return res

class HyperbolicLayerWrappedSinhAlt(RiemannianLayer):
    """ hyperbolic layer: geodesic + sinh + direct map """
    def __init__(self, in_features, out_features, manifold, over_param=False, weight_norm=False):
        super(HyperbolicLayerWrappedSinhAlt, self).__init__(in_features, out_features, manifold, over_param, weight_norm)
        self.bn = nn.BatchNorm1d(out_features)
    
    def forward(self, input):
        input = input.unsqueeze(-2).expand(-1, self.out_features, self.in_features) 
        euclidean_features = self.manifold.normdist2plane(input, self.bias, self.weight_sinhalt, signed=True, norm=self.weight_norm)
        euclidean_features_bn = self.bn(euclidean_features)
        res = self.manifold.sinh_direct_map(euclidean_features_bn)
        return res


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, **kwargs):
        super(Linear, self).__init__(
            in_features,
            out_features,
        )


class MobiusLayer(RiemannianLayer):
    def __init__(self, in_features, out_features, manifold, over_param=False, weight_norm=False):
        super(MobiusLayer, self).__init__(in_features, out_features, manifold, over_param, weight_norm)

    def forward(self, input):
        res = self.manifold.mobius_matvec(self.weight, input)
        return res


class ExpZero(nn.Module):
    def __init__(self, manifold):
        super(ExpZero, self).__init__()
        self.manifold = manifold

    def forward(self, input):
        return self.manifold.expmap0(input)


class LogZero(nn.Module):
    def __init__(self, manifold):
        super(LogZero, self).__init__()
        self.manifold = manifold

    def forward(self, input):
        return self.manifold.logmap0(input)

