"""
load models and get intermediate layers
"""

# load packages 
import os
from re import L
import sys
import argparse
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn 
from torch import optim
from torch.nn.parameter import Parameter
from torch.utils.tensorboard import SummaryWriter

from geoopt import optim as geo_optim

# load files 
sys.path.append(".")
sys.path.append("..")
from utils import Logger, Timer, save_model, save_vars, probe_infnan, load_config
from objectives import (
    ae_pairwise_dist_objective, metric_report, 
    _euclidean_pairwise_dist, _hyperbolic_pairwise_dist,
    _distortion_loss, _select_upper_triangular
)
import models

torch.backends.cudnn.benchmark = True

# force 64 
dtype = torch.float64
torch.set_default_dtype(dtype)

# load path config 
path_config = load_config()

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

# * note: some arguments does not make sense in vanilla autoencoder, 
# * kept for future development 

### General
parser.add_argument('--model', type=str, metavar='M', help='model name')
parser.add_argument('--manifold', type=str, default='PoincareBall',
                    choices=['Euclidean', 'PoincareBall'])
parser.add_argument('--save-freq', type=int, default=100,
                    help='print objective values every value (if positive)')

### Dataset
parser.add_argument('--data-params', nargs='+', default=[],
                    help='parameters which are passed to the dataset loader')
parser.add_argument('--data-size', type=int, nargs='+',
                    default=[], help='size/shape of data observations')

### Optimisation
parser.add_argument('--opt', type=str, default='adam', choices=['adam', 'riemannian_adam'],
                    help="the choice of optimizer, now supporting adam and riemannian adam")
parser.add_argument('--obj', type=str, default='vae',
                    help='objective to minimise (default: vae)')
parser.add_argument('--epochs', type=int, default=50, metavar='E',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='first parameter of Adam (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='second parameter of Adam (default: 0.900)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learnign rate for optimser (default: 1e-4)')

## Objective
parser.add_argument('--use-euclidean',  action='store_true', default=False,
                    help='use hyperbolic or euclidean distance for outputs, default=False')
parser.add_argument('--loss-function', help='type of loss function', default='scaled', type=str)
                    # choices=['raw', 'relative', 'scaled', 'robust_scaled', 'distortion', 'individual_distortion', 'modified_individual_distortion', 'robust_individual_distortion', 'learning_relative'])

### Model
parser.add_argument('--latent-dim', type=int, default=10,
                    metavar='L', help='latent dimensionality (default: 10)')
parser.add_argument('--c', type=float, default=1., help='curvature')
parser.add_argument('--thr', type=float, default=0.99, help='relative hard boundary of Poincare Ball')
parser.add_argument('--posterior', type=str, default='WrappedNormal', help='posterior distribution',
                    choices=['WrappedNormal', 'RiemannianNormal', 'Normal'])

## Architecture
parser.add_argument('--num-hidden-layers', type=int, default=1, metavar='H',
                    help='number of hidden layers in enc and dec (default: 1)')
parser.add_argument('--hidden-dim', type=int, default=100,
                    help='number of hidden layers dimensions (default: 100)')
parser.add_argument('--output-dim', type=int, default=None,
                    help='output dimension, just for distortion simulation (if None, output = input)')
parser.add_argument('--nl', type=str, default='ELU', help='non linearity')
parser.add_argument('--hyp-nl', type=str, default='ELU', help='non linearity for hyperbolic layers')
parser.add_argument('--enc', type=str, default='Wrapped', help='allow to choose different implemented encoder',
                    choices=['Linear', 'Wrapped', 'WrappedNaive', 'WrappedAlt', 'WrappedSinhAlt', 'Mixture', 'MixturePP','WrappedNaive'])
parser.add_argument('--dec', type=str, default='Wrapped', help='allow to choose different implemented decoder',
                    choices=['Linear', 'Wrapped', 'Geo', 'Mob', 'LinearSim', 'WrappedSim', 'GeoSim', 'MobSim'])

## for EncMixture only 
parser.add_argument('--hidden-dims', type=int, nargs='+', 
                    help='specifying the dimensions of each hidden layers in order',
                    default=[50, 50, 50, 50])
parser.add_argument('--num-hyperbolic-layers', type=int, 
                    help='number of hyperbolic layers', default=1)
parser.add_argument('--no-final-lift', action='store_true', default=False,
                    help='used to indicate no final lifting in the mixture model')
parser.add_argument('--lift-type', default='expmap', choices=['expmap', 'direct', 'sinh_direct'], type=str,
                    help='method to lift euclidean features to hyperbolic space, now supporting expmap, direct, and sinh_direct')
parser.add_argument('--no-bn', action="store_true", default=False,
                    help='turn on to disable batch normalization')
parser.add_argument('--read-ancestral-mask', default=False, action='store_true', help='whether to read ancestral mask and compute corresponding metrics')

## Prior
parser.add_argument('--prior-iso', action='store_true',
                    default=False, help='isotropic prior')
parser.add_argument('--prior', type=str, default='WrappedNormal', help='prior distribution',
                    choices=['WrappedNormal', 'RiemannianNormal', 'Normal'])
parser.add_argument('--prior-std', type=float, default=1.,
                    help='scale stddev by this value (default:1.)')
parser.add_argument('--learn-prior-std', action='store_true', default=False)

### Technical
parser.add_argument('--no-cuda', action='store_true',
                    default=False, help='disables CUDA use')
parser.add_argument('--seed', type=int, default=0,
                    metavar='S', help='random seed (default: 1)')

### model save 
parser.add_argument('--log-model', default=False, action='store_true', help='serialize pytorch model')
parser.add_argument('--no-final-clip', action='store_true', default=False,
                    help='whether to save the clipped final embeddings')
parser.add_argument('--log-train', action='store_true', default=False,
                    help='whether to save model emb visualization')
parser.add_argument('--log-train-epochs', default=20, type=int, 
                    help='by this number of epochs we save a visualization')

### model metric report 
parser.add_argument('--no-model-report', action='store_true', default=False, 
                    help='whether to save model metrics')
parser.add_argument('--save-each-epoch', action='store_true', default=False,
                    help='whether to record statistics of each epoch')
parser.add_argument('--record-name', type=str, default='',
                    help='the name of the record file')
parser.add_argument('--model-save-dir', type=str, default='results', help='the path to store model pt and emb vis')
parser.add_argument('--use-translation', default=False, action='store_true', help='enable translation in visualization')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
args.prior_iso = args.prior_iso or args.posterior == 'RiemannianNormal'
args.batch_size = 1  # dummy variable, useless since all are trained using a full batch

# parameters for ae objectives
use_hyperbolic = False if args.use_euclidean else True
args.use_hyperbolic = use_hyperbolic
max_radius = np.sqrt(1 / args.c)
thr = args.thr * max_radius  # absolute scale hard boundary
# use_hyperbolic = False
curvature = torch.Tensor([args.c]).to(device).to(dtype)

c = torch.tensor(args.c, dtype=dtype).to(device)
thr = torch.tensor(thr, dtype=dtype).to(device)

# ===================== paths and data ===========================
if args.log_train:
    # encode model
    model_type = args.enc
    if 'Mixture' in model_type:
        model_save_dir_name = '{}_{}_hd_{}_lift_{}_numhyp_{}_nofinal_{}_ld_{}_loss_{}_epochs_{}'.format(
            model_type,
            args.data_params[0],
            args.hidden_dims,
            args.lift_type,
            args.num_hyperbolic_layers,
            args.no_final_lift,
            args.latent_dim,
            args.loss_function,
            args.epochs
        )
        model_save_dir_name += '_bn' if not args.no_bn else ''
    else:
        if model_type == 'Linear':
            model_type += '_(hyp)' if use_hyperbolic else '_(euc)'
        model_save_dir_name = '{}_{}_hd_{}_ld_{}_loss_{}_epochs_{}'.format(
            model_type,
            args.data_params[0],
            args.hidden_dim,
            args.latent_dim,
            args.loss_function,
            args.epochs
        )
        model_save_dir_name += '_bn' if not args.no_bn else ''
    model_save_dir = os.path.join(path_config['model_save_dir'], model_save_dir_name)

    # load edges and color encoding
    with open(os.path.join(path_config['dataset_root'], args.data_params[0], 'sim_tree_edges.npy'), 'rb') as f:
        edges = np.load(f)

    # TODO: make and load color encoding
    # Initialize tensorboard writer
    tb_writer = SummaryWriter(log_dir=model_save_dir)

# ancestral
if args.read_ancestral_mask:
    with open(os.path.join(path_config['dataset_root'], args.data_params[0], 'sim_tree_ancestral_mask.npy'), 'rb') as f:
        ancestral_mask = np.load(f)
    ancestral_mask = torch.as_tensor(ancestral_mask, device=device)

# ==============================================================
# Initialise model, optimizer, dataset loader and loss function
modelC = getattr(models, 'Enc_{}'.format(args.model))
model = modelC(args)

if 'learning' in args.loss_function:
    alpha = Parameter(torch.tensor(1.))  # initialize to 1 
    model.learning_alpha = alpha

model.load_state_dict(torch.load(os.path.join(model_save_dir, 'model.pt'), map_location=device))
model.eval()
model.to(device)

# =============================================================
# load data
overall_loader, shortest_path_mat = model.getDataLoaders(
    args.batch_size, True, device, dtype,
    *args.data_params, path_config['dataset_root']
)
loss_function = ae_pairwise_dist_objective
shortest_path_mat = shortest_path_mat.to(device).to(dtype)


# ==================== retrieve intermeidate layers =================
@torch.no_grad()
def get_intermediate_layer_mixturepp_output(model, data):
    """ return intermediate output distance matrix """
    enc_layer = model.enc
    intermediate_dist_mat = []
    output = data

    # euclidean 
    for layer in enc_layer.euclidean_layers:
        output = layer(output)
        if isinstance(layer, nn.Linear):
            dist_mat = _euclidean_pairwise_dist(output.detach().clone())
            intermediate_dist_mat.append(dist_mat)
    
    # bn 
    output = enc_layer.bn(output)

    # lifting / bridge
    output = enc_layer.bridge_map(output)
    if not isinstance(enc_layer.bridge_map, nn.Identity):
        dist_mat = _hyperbolic_pairwise_dist(output.detach().clone(), c=c, thr=thr)
        intermediate_dist_mat.append(dist_mat)
    
        # hyperbolic 
        if len(enc_layer.hyperbolic_layers) > 0:
            for hyp_layer in enc_layer.hyperbolic_layers:
                output = hyp_layer.manifold.normdist2planePP(output, hyp_layer.z, hyp_layer.r)
                dist_mat = _hyperbolic_pairwise_dist(output.detach().clone(), c=c, thr=thr)
                intermediate_dist_mat.append(dist_mat)

                output = hyp_layer.hyp_nl(output)
                output = hyp_layer.bn(output)
                if args.lift_type == 'expmap':
                    output = hyp_layer.manifold.expmap0(output)
                elif args.lift_type == 'direct':
                    output = hyp_layer.manifold.direct_map(output)
                elif args.lift_type == 'sinh_direct':
                    output = hyp_layer.manifold.sinh_direct_map(output)
            
            dist_mat = _hyperbolic_pairwise_dist(output.detach().clone(), c=c, thr=thr)
            intermediate_dist_mat.append(dist_mat)
    
    return intermediate_dist_mat

@torch.no_grad()
def get_intermediate_distortion(shortest_path_mat, intermediate_dist_mat):
    distortions = []
    for output in intermediate_dist_mat:
        emb_dists_selected, real_dists_selected = _select_upper_triangular(output, shortest_path_mat)
        distortion = _distortion_loss(emb_dists_selected, real_dists_selected)
        distortions.append(distortion.detach().cpu().item())
    return distortions

def main():
    for _, (data, _) in enumerate(overall_loader):  # no train test split needed, yet  
        intermediate_dist_mat = get_intermediate_layer_mixturepp_output(model, data)
        distortions = get_intermediate_distortion(shortest_path_mat, intermediate_dist_mat)
    
    # save 
    with open(os.path.join(model_save_dir, 'distortion_by_layer.txt'), 'w') as f:
        f.write(','.join([str(x) for x in distortions]))
        f.write('\n')


if __name__ == '__main__':
    main()
