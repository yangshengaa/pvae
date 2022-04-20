from numpy import prod

from functools import partial

import torch
import torch.distributions as dist
import torch.nn.functional as F

from pvae.utils import has_analytic_kl, log_mean_exp, Constants

# cuda specification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
def _distortion_rate(emb_dists, real_dists):
    """ compute distortion rate """ 
    with torch.no_grad():
        N = emb_dists.shape[0]
        emb_dists_shift = emb_dists + torch.eye(N).to(device) 
        real_dists_shift = real_dists + torch.eye(N).to(device)

        contractions = real_dists_shift / (emb_dists_shift + Constants.eta)
        expansions = emb_dists_shift / (real_dists_shift + Constants.eta)
        contraction = torch.max(contractions)
        expansion = torch.max(expansions)
        distortion = contraction * expansion
        return distortion


def ae_pairwise_dist_objective(model, data, shortest_path_mat, use_hyperbolic=False, c=1):
    """
    minimize regression MSE (equally weighted) on the estimated pairwise distance. The output distance is 
    either measured in Euclidean or in hyperbolic sense

    assume that the data comes in the original sequence (shuffle = False)

    :param c: the curvature, if use_hyperbolic is true 
    """
    # reconstruct
    reconstructed_data = model(data).squeeze()

    # select loss function
    if use_hyperbolic: 
        emb_dists = _hyperbolic_pairwise_dist(reconstructed_data, c=c)
    else:
        emb_dists = _euclidean_pairwise_dist(reconstructed_data)

    loss = torch.mean((emb_dists - shortest_path_mat) ** 2)
    
    # compute distortion 
    distortion_rate = _distortion_rate(emb_dists, shortest_path_mat)

    return loss, distortion_rate
