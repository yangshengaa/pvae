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
def _hyperbolic_distance(z, y, c=1, thr=0.9):
    """ 
    helper function to compute the hyperbolic distance 

    :param thr: threshold for numerical stability
    """
    z_norm = torch.linalg.norm(z)
    y_norm = torch.linalg.norm(y)

    # hard threshold: https://doi.org/10.48550/arXiv.2107.11472
    z = z * thr / z_norm if z_norm > thr else z
    y = y * thr / y_norm if y_norm > thr else y

    # distance
    dist = 1 / torch.sqrt(c) * torch.arccosh(
        1 + 2 * c * torch.linalg.norm(z - y) ** 2 / ( 
            (1 - c * z_norm ** 2) * (1 - c * y_norm ** 2)  
        )
    )
    return dist

def _distortion_rate(emb_dists, real_dists):
    """ compute the distortion rate from embedding distances and real distances """
    with torch.no_grad():
        contractions = real_dists / emb_dists
        expansions = emb_dists / real_dists
        contraction = torch.max(contractions)
        expansion = torch.max(expansions)
        distortion = contraction * expansion
        return distortion

def ae_pairwise_dist_objective(model, data, labels, shortest_path_dict, use_hyperbolic=False, c=1):
    """
    minimize regression MSE (equally weighted) on the estimated pairwise distance. The output distance is 
    either measured in Euclidean or in hyperbolic sense

    :param c: the curvature, if use_hyperbolic is true 
    """
    reconstructed_data = model(data).squeeze()
    loss = 0

    # determine distance function 
    if use_hyperbolic:
        dist_f = partial(_hyperbolic_distance, c=c)
    else:
        dist_f = lambda x1, x2: torch.linalg.norm(x1 - x2)
    
    # compute pairwise dist MSE
    emb_dists, real_dists = [], []
    for i in range(len(reconstructed_data) - 1):
        reconstructed_data_1 = reconstructed_data[i]
        label_1 = labels[i].item()
        for j in range(i + 1, len(reconstructed_data)):
            reconstructed_data_2 = reconstructed_data[j]
            label_2 = labels[j].item()

            cur_dist = dist_f(reconstructed_data_1, reconstructed_data_2)
            real_dist = shortest_path_dict[(min(label_1, label_2), max(label_1, label_2))]

            # compute loss 
            loss += (cur_dist - real_dist) ** 2

            # record dist for distortion 
            emb_dists.append(cur_dist)
            real_dists.append(real_dist)

    # normalize loss 
    loss /= (len(data) * (len(data) - 1))

    # compute distortion 
    emb_dists = torch.Tensor(emb_dists)
    real_dists = torch.Tensor(real_dists)
    distortion_rate = _distortion_rate(emb_dists, real_dists)

    return loss, distortion_rate
