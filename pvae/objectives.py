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
    dist_mat = torch.cdist(data_mat, data_mat, p=2).clamp(5e-4)
    return dist_mat

def _hyperbolic_pairwise_dist(data_mat, c=1, thr=0.9):
    """ compute pairwise hyperbolic distance """
    # hard threshold: https://doi.org/10.48550/arXiv.2107.11472
    data_mat_rescaled = data_mat * torch.clamp(thr / torch.linalg.norm(data_mat, dim=1, keepdim=True), max=1)
    data_mat_rescaled_norm = torch.linalg.norm(data_mat_rescaled, dim=1, keepdim=True)
    euclidean_dist_mat = _euclidean_pairwise_dist(data_mat_rescaled)
    denom = (1 - c * data_mat_rescaled_norm ** 2) @ (1 - c * data_mat_rescaled_norm ** 2).T

    dist_mat = 1 / torch.sqrt(c) * torch.arccosh(
        1 + 
        (
            2 * c * euclidean_dist_mat ** 2 / denom
        ).clamp(min=1e-7)  # * note that the gradient of arcsinh could only be computed when input is larger than 1 + 1e-7
    )# .clamp(0.1)
    return dist_mat

def _diameter(emb_dists):
    """ compute the diameter of the embeddings """
    return torch.max(emb_dists)

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
    loss = torch.mean(emb_dists_selected / (real_dists_selected)) * torch.mean(real_dists_selected / (emb_dists_selected))
    return loss

def _individual_distortion_loss(emb_dists, real_dists):
    """ 
    compute the following loss 
    
    average of the average distortion 
    """
    # ! temporary !
    n = real_dists.shape[0]
    pairwise_contraction = real_dists / (emb_dists + Constants.eta)
    pairwise_expansion = emb_dists / (real_dists + Constants.eta)
    pairwise_contraction.fill_diagonal_(0)
    pairwise_expansion.fill_diagonal_(0)

    # print(torch.max(pairwise_contraction))
    # print(torch.max(pairwise_expansion))

    # compute individual
    individual_pairwise_contraction = pairwise_contraction.sum(axis=1) / (n - 1)
    individual_pairwise_expansion = pairwise_expansion.sum(axis=1) / (n - 1)
    individual_distortion = individual_pairwise_contraction * individual_pairwise_expansion
    
    loss =  individual_distortion.mean()
    return loss

def _modified_individual_distortion_loss(emb_dists, real_dists):
    """ 
    add epsilon to both the numerator and the denominator 
    """
    n = real_dists.shape[0]
    pairwise_contraction = (real_dists + Constants.eta) / (emb_dists + Constants.eta)
    pairwise_expansion = (emb_dists + Constants.eta) / (real_dists + Constants.eta)
    pairwise_contraction.fill_diagonal_(0)
    pairwise_expansion.fill_diagonal_(0)


    # compute individual
    individual_pairwise_contraction = pairwise_contraction.sum(axis=1) / (n - 1)
    individual_pairwise_expansion = pairwise_expansion.sum(axis=1) / (n - 1)
    individual_distortion = individual_pairwise_contraction * individual_pairwise_expansion
    
    loss =  individual_distortion.mean()
    return loss

def _robust_individual_distortion_loss(emb_dists, real_dists):
    """ convert to cosh before comparision """
    emb_dists_cosh = torch.cosh(emb_dists)
    real_dists_cosh = torch.cosh(real_dists)
    
    contraction = real_dists_cosh / emb_dists_cosh 
    expansion = emb_dists_cosh / real_dists_cosh

    loss = (contraction.mean(axis=1) * expansion.mean(axis=1)).mean()
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
    
def _individual_distortion_rate(emb_dists, real_dists):
    """ compute the average avarage distortion rate """
    with torch.no_grad():
        return _individual_distortion_loss(emb_dists, real_dists)


def ae_pairwise_dist_objective(model, data, shortest_path_mat, use_hyperbolic=False, loss_function_type='scaled', c=1, thr=0.9):
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
        emb_dists = _hyperbolic_pairwise_dist(reconstructed_data, c=c, thr=thr)
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
    elif loss_function_type == 'individual_distortion':
        loss = _individual_distortion_loss(emb_dists, shortest_path_mat)
    elif loss_function_type == 'modified_individual_distortion':
        loss = _modified_individual_distortion_loss(emb_dists, shortest_path_mat)
    elif loss_function_type == 'robust_individual_distortion':
        loss = _robust_individual_distortion_loss(emb_dists, shortest_path_mat)
    else:
        raise NotImplementedError(f'loss function type {loss_function_type} not available')
    
    # compute distortion and variances 
    with torch.no_grad():
        contractions = real_dists_selected / (emb_dists_selected)
        expansions = emb_dists_selected / (real_dists_selected)
        contractions_std = torch.std(contractions)
        expansions_std = torch.std(expansions)
        
        distortion_rate = _distortion_rate(contractions, expansions)
        max_distortion_rate = _max_distortion_rate(contractions, expansions)
        individual_distortion_rate = _individual_distortion_rate(emb_dists, shortest_path_mat)

        diameter = _diameter(emb_dists_selected)

    return reconstructed_data, loss, (emb_dists_selected, real_dists_selected, emb_dists)


def metric_report(
        emb_dists_selected, real_dists_selected,
        emb_dists, real_dists
    ):
    """ report metric along training """
    with torch.no_grad():
        contractions = real_dists_selected / (emb_dists_selected)
        expansions = emb_dists_selected / (real_dists_selected)
        contractions_std = torch.std(contractions)
        expansions_std = torch.std(expansions)
        
        # all candidate loss 
        distortion_rate = _distortion_rate(contractions, expansions)
        max_distortion_rate = _max_distortion_rate(contractions, expansions)
        individual_distortion_rate = _individual_distortion_rate(emb_dists, real_dists)
        relative_rate = _relative_pairwise_dist_loss(emb_dists_selected, real_dists_selected)
        scaled_rate = _scaled_pairwise_dist_loss(emb_dists_selected, real_dists_selected)

        diameter = _diameter(emb_dists_selected)

        return (
            distortion_rate, individual_distortion_rate, max_distortion_rate, 
            relative_rate, scaled_rate, 
            contractions_std, expansions_std, 
            diameter
        )