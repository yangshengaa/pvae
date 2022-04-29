"""
for simulation to test distortion only, copy from main with minor modifications
"""

import sys
sys.path.append(".")
sys.path.append("..")
import os
import datetime
import json
import argparse
from tempfile import mkdtemp
from collections import defaultdict
import subprocess
import torch
from torch import optim
import numpy as np

from utils import Logger, Timer, save_model, save_vars, probe_infnan
from objectives import ae_pairwise_dist_objective
import models


runId = datetime.datetime.now().isoformat().replace(':', '_')
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

# * note: some arguments does not make sense in vanilla autoencoder, 
# * kept for future development 

### General
parser.add_argument('--save-dir', type=str, default='')
parser.add_argument('--model', type=str, metavar='M', help='model name')
parser.add_argument('--manifold', type=str, default='PoincareBall',
                    choices=['Euclidean', 'PoincareBall'])
parser.add_argument('--name', type=str, default='.',
                    help='experiment name (default: None)')
parser.add_argument('--save-freq', type=int, default=0,
                    help='print objective values every value (if positive)')
parser.add_argument('--skip-test', action='store_true',
                    default=False, help='skip test dataset computations')

### Dataset
parser.add_argument('--data-params', nargs='+', default=[],
                    help='parameters which are passed to the dataset loader')
parser.add_argument('--data-size', type=int, nargs='+',
                    default=[], help='size/shape of data observations')

### Metric & Plots
parser.add_argument('--iwae-samples', type=int, default=0,
                    help='number of samples to compute marginal log likelihood estimate')

### Optimisation
parser.add_argument('--obj', type=str, default='vae',
                    help='objective to minimise (default: vae)')
parser.add_argument('--epochs', type=int, default=50, metavar='E',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--batch-size', type=int, default=64,
                    metavar='N', help='batch size for data (default: 64)')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='first parameter of Adam (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='second parameter of Adam (default: 0.900)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learnign rate for optimser (default: 1e-4)')

## Objective
parser.add_argument('--K', type=int, default=1, metavar='K',
                    help='number of samples to estimate ELBO (default: 1)')
parser.add_argument('--beta', type=float, default=1.0,
                    metavar='B', help='coefficient of beta-VAE (default: 1.0)')
parser.add_argument('--analytical-kl', action='store_true',
                    default=False, help='analytical kl when possible')
parser.add_argument('--use-hyperbolic', help='whether to use hyperbolic distance for outputs, default=True', 
                    default=True, type=bool)
parser.add_argument('--loss-function', help='type of loss function', default='scaled', type=str, 
                    choices=['raw', 'relative', 'scaled', 'distortion'])

### Model
parser.add_argument('--latent-dim', type=int, default=10,
                    metavar='L', help='latent dimensionality (default: 10)')
parser.add_argument('--c', type=float, default=1., help='curvature')
parser.add_argument('--posterior', type=str, default='WrappedNormal', help='posterior distribution',
                    choices=['WrappedNormal', 'RiemannianNormal', 'Normal'])

## Architecture
parser.add_argument('--num-hidden-layers', type=int, default=1, metavar='H',
                    help='number of hidden layers in enc and dec (default: 1)')
parser.add_argument('--hidden-dim', type=int, default=100,
                    help='number of hidden layers dimensions (default: 100)')
parser.add_argument('--output-dim', type=int, default=None,
                    help='output dimension, just for distortion simulation (if None, output = input)')
parser.add_argument('--nl', type=str, default='ReLU', help='non linearity')
parser.add_argument('--enc', type=str, default='Wrapped', help='allow to choose different implemented encoder',
                    choices=['Linear', 'Wrapped', 'WrappedAlt', 'Mob'])
parser.add_argument('--dec', type=str, default='Wrapped', help='allow to choose different implemented decoder',
                    choices=['Linear', 'Wrapped', 'Geo', 'Mob', 'LinearSim', 'WrappedSim', 'GeoSim', 'MobSim'])

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
parser.add_argument('--save-model-emb', default=False, 
                    help='whether to save the trained embeddings', type=bool)

### model metric report 
parser.add_argument('--save-model-report', default=True, 
                    help='whether to save model metrics')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
args.prior_iso = args.prior_iso or args.posterior == 'RiemannianNormal'

# # Choosing and saving a random seed for reproducibility
# if args.seed == 0:
#     args.seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
# print('seed', args.seed)
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)
# torch.manual_seed(args.seed)
# torch.backends.cudnn.deterministic = True

# * disable model saving for now 

# Initialise model, optimizer, dataset loader and loss function
modelC = getattr(models, 'Enc_{}'.format(args.model))
model = modelC(args).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True, betas=(args.beta1, args.beta2))
# overall_loader, shortest_path_dict, shortest_path_mat = model.getDataLoaders(
#     args.batch_size, True, device, *args.data_params
# )
overall_loader, shortest_path_mat = model.getDataLoaders(
    args.batch_size, True, device, *args.data_params
)
loss_function = ae_pairwise_dist_objective # getattr(objectives, args.obj + '_objective')
shortest_path_mat = shortest_path_mat.to(device)

# parameters for ae objectives 
use_hyperbolic = args.use_hyperbolic
curvature = torch.Tensor([args.c]).to(device)
loss_function_type = args.loss_function


def train(epoch, agg):
    model.train()
    b_loss = 0.
    for _, (data, labels) in enumerate(overall_loader):  # no train test split needed, yet  
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss, train_distortion, train_max_distortion = loss_function(
            model, data, shortest_path_mat, 
            use_hyperbolic=use_hyperbolic, c=curvature,
            loss_function_type=loss_function_type
        )
        probe_infnan(loss, "Training loss:")
        loss.backward()
        optimizer.step()

        b_loss += loss.item()

    # agg['train_loss'].append(b_loss)
    agg['distortion'].append(train_distortion)
    agg['max_distortion'].append(train_max_distortion)
    agg['train_loss'].append(b_loss)
    if epoch % 100 == 0:
        print(f'====> Epoch: {epoch:03d} Loss: {b_loss:.4f}, Distortion: {train_distortion:.4f}, Max Distortion {train_max_distortion:.2f}')


def save_emb():
    """ save final embeddings """
    with torch.no_grad():
        for _, (data, _) in enumerate(overall_loader):
            data_emb = model(data).squeeze()
        data_emb_np = data_emb.numpy()
        
        # save 
        save_path = 'experiments'
        model_params = f'{args.data_params[0]},{args.data_size[0]},{args.latent_dim},{args.enc},{args.use_hyperbolic},{args.c},{args.loss_function}'
        with open(os.path.join(save_path, model_params + '_data_emb.npy'), 'wb') as f:
            np.save(f, data_emb_np)

def record_info(agg):
    """ record loss and distortion """
    basic_params = f'{args.data_params[0]},{args.data_size[0]},{args.latent_dim},{args.enc},{args.use_hyperbolic},{args.c},{args.loss_function},'
    main_report = basic_params + f'{agg["train_loss"][-1]:.4f},{agg["distortion"][-1]:.3f},{agg["max_distortion"][-1]:.3f}'
    loss_report = basic_params + ','.join([f"{agg['train_loss'][i]:.4f}" for i in range(len(agg['train_loss']))])
    distortion_report = basic_params + ','.join([f"{agg['distortion'][i]:.5f}" for i in range(len(agg['distortion']))])
    max_distortion_report = basic_params + ','.join([f"{agg['max_distortion'][i]:.5f}" for i in range(len(agg['max_distortion']))])

    # write to file 
    sim_record_path = 'experiments'
    with open(os.path.join(sim_record_path, 'sim_records.txt'), 'a') as f:
        f.write(main_report)
        f.write('\n')
    with open(os.path.join(sim_record_path, 'sim_loss.txt'), 'a') as f:
        f.write(loss_report)
        f.write('\n')
    with open(os.path.join(sim_record_path, 'sim_distortion.txt'), 'a') as f:
        f.write(distortion_report)
        f.write('\n')
    with open(os.path.join(sim_record_path, 'sim_max_distortion.txt'), 'a') as f:
        f.write(max_distortion_report)
        f.write('\n')
    
    # for testing only
    # with open(os.path.join(sim_record_path, 'temp_sim_records.txt'), 'a') as f:
    #     f.write(main_report)
    #     f.write('\n')
    # with open(os.path.join(sim_record_path, 'temp_sim_loss.txt'), 'a') as f:
    #     f.write(loss_report)
    #     f.write('\n')
    # with open(os.path.join(sim_record_path, 'temp_sim_distortion.txt'), 'a') as f:
    #     f.write(distortion_report)
    #     f.write('\n')

def main():
    """ main running """
    with Timer('ME-VAE') as t:
        # agg = defaultdict(list)
        agg = defaultdict(list)
        print('Starting training...')

        for epoch in range(1, args.epochs + 1):
            train(epoch, agg)

        # save embeddings 
        if args.save_model_emb:
            save_emb()
        
        # record simulation results
        if (args.save_model_report):
            record_info(agg)
 
if __name__ == '__main__':
    main()
