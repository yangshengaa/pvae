from numpy import prod

from functools import partial

import torch
import torch.distributions as dist
import torch.nn.functional as F

from pvae.utils import has_analytic_kl, log_mean_exp, Constants


def vae_objective(model, x, K=1, beta=1.0, components=False, analytical_kl=False, **kwargs):
    """Computes E_{p(x)}[ELBO] """
    qz_x, px_z, zs = model(x, K)
    _, B, D = zs.size()
    flat_rest = torch.Size([*px_z.batch_shape[:2], -1])
    lpx_z = px_z.log_prob(x.expand(px_z.batch_shape)).view(flat_rest).sum(-1)

    pz = model.pz(*model.pz_params)
    kld = dist.kl_divergence(qz_x, pz).unsqueeze(0).sum(-1) if \
        has_analytic_kl(type(qz_x), model.pz) and analytical_kl else \
        qz_x.log_prob(zs).sum(-1) - pz.log_prob(zs).sum(-1)

    obj = -lpx_z.mean(0).sum() + beta * kld.mean(0).sum()
    return (qz_x, px_z, lpx_z, kld, obj) if components else obj

def _iwae_objective_vec(model, x, K):
    """Helper for IWAE estimate for log p_\theta(x) -- full vectorisation."""
    qz_x, px_z, zs = model(x, K)
    flat_rest = torch.Size([*px_z.batch_shape[:2], -1])
    lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
    lpx_z = px_z.log_prob(x.expand(zs.size(0), *x.size())).view(flat_rest).sum(-1)
    lqz_x = qz_x.log_prob(zs).sum(-1)
    obj = lpz.squeeze(-1) + lpx_z.view(lpz.squeeze(-1).shape) - lqz_x.squeeze(-1)
    return -log_mean_exp(obj).sum()


def iwae_objective(model, x, K):
    """Computes an importance-weighted ELBO estimate for log p_\theta(x)
    Iterates over the batch as necessary.
    Appropriate negation (for minimisation) happens in the helper
    """
    split_size = int(x.size(0) / (K * prod(x.size()) / (3e7)))  # rough heuristic
    if split_size >= x.size(0):
        obj = _iwae_objective_vec(model, x, K)
    else:
        obj = 0
        for bx in x.split(split_size):
            obj = obj + _iwae_objective_vec(model, bx, K)
    return obj

# ============ for simulation only =============

# metics 
def _euclidean_pairwise_dist(data_mat):
    """ 
    compute the pairwise euclidean distance matrix 
    || x - y || ^ 2 = || x || ^ 2 - 2 < x, y > + || y || ^ 2
    
    :param data_mat: of N by D 
    :return dist_mat: of N by N 
    """
    data_norm_squared = torch.linalg.norm(data_mat, dim=1, keepdim=True) ** 2
    data_inner_prod = data_mat @ data_mat.T 
    template = torch.ones_like(data_inner_prod)  # keep dim 

    dist_mat_squared = data_norm_squared * template + data_norm_squared.T * template - 2 * data_inner_prod
    dist_mat = torch.sqrt(torch.clamp(dist_mat_squared, min=Constants.eta))  # for numerical stability
    return dist_mat

def _hyperbolic_pairwise_dist(data_mat, c=1, thr=0.9):
    """ compute pairwise hyperbolic distance """
    # hard threshold: https://doi.org/10.48550/arXiv.2107.11472
    data_mat_rescaled = data_mat * torch.clamp(thr / torch.linalg.norm(data_mat, dim=1, keepdim=True), max=1)
    data_mat_rescaled_norm = torch.linalg.norm(data_mat_rescaled, dim=1, keepdim=True)
    euclidean_dist_mat = _euclidean_pairwise_dist(data_mat_rescaled)
    denom = (1 - c * data_mat_rescaled_norm ** 2) @ (1 - c * data_mat_rescaled_norm ** 2).T

    dist_mat = 1 / torch.sqrt(c) * torch.arccosh(
        1 + 2 * c * euclidean_dist_mat ** 2 / denom
    )

    return dist_mat

# loss functions 
def _select_upper_triangular(emb_dists, real_dists):
    """ select the upper triangular portion of the distance matrix """
    mask = torch.triu(torch.ones_like(real_dists), diagonal=1) > 0
    emb_dists_selected = torch.masked_select(emb_dists, mask)
    real_dists_selected = torch.masked_select(real_dists, mask)
    return emb_dists_selected, real_dists_selected

def _pairwise_dist_loss(emb_dists_selected, real_dists_selected):
    """ equally weighted pairwise loss """
    loss = torch.mean((emb_dists_selected - real_dists_selected) ** 2)
    return loss 

def _relative_pairwise_dist_loss(emb_dists_selected, real_dists_selected):
    """ 
    relative distance pairwise loss, given by 
    ((d_e - d_r) / d_r) ** 2
    """ 
    loss = torch.mean((emb_dists_selected / real_dists_selected - 1) ** 2)
    return loss 

def _scaled_pairwise_dist_loss(emb_dists_selected, real_dists_selected):
    """ 
    scaled version that is more presumably more compatible with optimizing distortion 
    (d_e / mean(d_e) - d_r / mean(d_r)) ** 2
    """
    loss = torch.mean(
        ((emb_dists_selected / emb_dists_selected.mean()) - (real_dists_selected / real_dists_selected.mean())) ** 2
    )
    return loss

def _distortion_loss(emb_dists_selected, real_dists_selected):
    """ directly use average distortion as the loss """
    loss = torch.mean(emb_dists_selected / (real_dists_selected + Constants.eta)) * torch.mean(real_dists_selected / (emb_dists_selected + Constants.eta))
    return loss

# distortion evaluations 
def _max_distortion_rate(contractions, expansions):
    """ compute max distortion rate """ 
    with torch.no_grad():
        contraction = torch.max(contractions)       # max 
        expansion = torch.max(expansions)           # max 
        distortion = contraction * expansion
        return distortion


def _distortion_rate(contractions, expansions):
    """ compute 'average' distortion rate """
    with torch.no_grad():
        contraction = torch.mean(contractions)      # mean 
        expansion = torch.mean(expansions)          # mean 
        distortion = contraction * expansion
        return distortion


def ae_pairwise_dist_objective(model, data, shortest_path_mat, use_hyperbolic=False, loss_function_type='scaled', c=1):
    """
    minimize regression MSE (equally weighted) on the estimated pairwise distance. The output distance is 
    either measured in Euclidean or in hyperbolic sense

    assume that the data comes in the original sequence (shuffle = False)

    :param c: the curvature, if use_hyperbolic is true 
    :param loss_function_type: raw for the normal one, relative for relative dist, scaled for scaled
    """
    # reconstruct
    reconstructed_data = model(data).squeeze()

    # select distance 
    if use_hyperbolic: 
        emb_dists = _hyperbolic_pairwise_dist(reconstructed_data, c=c)
    else:
        emb_dists = _euclidean_pairwise_dist(reconstructed_data)
    
    # select upper triangular portion 
    emb_dists_selected, real_dists_selected = _select_upper_triangular(emb_dists, shortest_path_mat)

    # select loss function 
    if loss_function_type == 'raw':
        loss = _pairwise_dist_loss(emb_dists_selected, real_dists_selected)
    elif loss_function_type == 'relative':
        loss = _relative_pairwise_dist_loss(emb_dists_selected, real_dists_selected)
    elif loss_function_type == 'scaled':
        loss = _scaled_pairwise_dist_loss(emb_dists_selected, real_dists_selected)
    elif loss_function_type == 'distortion':
        loss = _distortion_loss(emb_dists_selected, real_dists_selected)
    else:
        raise NotImplementedError(f'loss function type {loss_function_type} not available')
    
    # compute distortion and variances 
    with torch.no_grad():
        contractions = real_dists_selected / (emb_dists_selected + Constants.eta)
        expansions = emb_dists_selected / (real_dists_selected + Constants.eta)
        contractions_std = torch.std(contractions)
        expansions_std = torch.std(expansions)
        
        distortion_rate = _distortion_rate(contractions, expansions)
        max_distortion_rate = _max_distortion_rate(contractions, expansions)

    return loss, distortion_rate, max_distortion_rate, contractions_std, expansions_std
