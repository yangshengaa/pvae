"""
for simulation to test distortion only, copy from main with minor modifications
"""

# load packages 
import os
import sys
import datetime
import argparse
from collections import defaultdict

import numpy as np

import torch
from torch import optim

# load files 
sys.path.append(".")
sys.path.append("..")
from utils import Logger, Timer, save_model, save_vars, probe_infnan
from objectives import ae_pairwise_dist_objective
import models

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

# * note: some arguments does not make sense in vanilla autoencoder, 
# * kept for future development 

### General
parser.add_argument('--save-dir', type=str, default='')
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
parser.add_argument('--use-euclidean',  action='store_false', default=False,
                    help='use hyperbolic or euclidean distance for outputs, default=False')
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
parser.add_argument('--save-model-emb', action='store_true', default=False, 
                    help='whether to save the trained embeddings')

### model metric report 
parser.add_argument('--no-model-report', action='store_true', default=False, 
                    help='whether to save model metrics')
parser.add_argument('--save-each-epoch', action='store_true', default=False,
                    help='whether to record statistics of each epoch')

## technical 
parser.add_argument('--cluster-code', default=0, type=int,
                    help='manually enable cluster computation. 0 means all in one instance, and 1, 2, 3 or 4 indicates current cluster')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
args.prior_iso = args.prior_iso or args.posterior == 'RiemannianNormal'
args.batch_size = 1  # dummy variable, useless since all are trained using a full batch


# Initialise model, optimizer, dataset loader and loss function
modelC = getattr(models, 'Enc_{}'.format(args.model))
model = modelC(args).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True, betas=(args.beta1, args.beta2))
overall_loader, shortest_path_mat = model.getDataLoaders(
    args.batch_size, True, device, *args.data_params
)
loss_function = ae_pairwise_dist_objective 
shortest_path_mat = shortest_path_mat.to(device)

# parameters for ae objectives 
use_hyperbolic = False if args.use_euclidean else True
args.use_hyperbolic = use_hyperbolic
# use_hyperbolic = False
curvature = torch.Tensor([args.c]).to(device)
loss_function_type = args.loss_function


def train(epoch, agg):
    model.train()
    b_loss = 0.
    for _, (data, _) in enumerate(overall_loader):  # no train test split needed, yet  
        data = data.to(device)
        optimizer.zero_grad()
        loss, train_distortion, train_max_distortion, contractions_std, expansions_std = loss_function(
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
    agg['contractions_std'].append(contractions_std)
    agg['expansions_std'].append(expansions_std)
    if epoch % args.save_freq == 0:
        print(f'====> Epoch: {epoch:03d} Loss: {b_loss:.4f}, Distortion: {train_distortion:.4f}, Max Distortion {train_max_distortion:.2f}' + 
        f', Contraction Std {contractions_std:.4f}, Expansion Std {expansions_std}')


def save_emb():
    """ save final embeddings """
    with torch.no_grad():
        for _, (data, _) in enumerate(overall_loader):
            data_emb = model(data).squeeze()

        # clip
        data_emb_clipped = data_emb * torch.clamp(0.9 / torch.linalg.norm(data_emb, dim=1, keepdim=True), max=1)
        data_emb_np = data_emb_clipped.numpy()
        
        # save 
        save_path = 'experiments'
        model_params = f'{args.data_params[0]},{args.data_size[0]},{args.latent_dim},{args.enc},{args.use_hyperbolic},{args.c},{args.loss_function}'
        with open(os.path.join(save_path, model_params + '_data_emb.npy'), 'wb') as f:
            np.save(f, data_emb_np)


def record_info(agg):
    """ record loss and distortion """
    basic_params = f'{args.data_params[0]},{args.data_size[0]},{args.latent_dim},{args.enc},{args.use_hyperbolic},{args.c},{args.loss_function},'
    main_report = basic_params + f'{agg["train_loss"][-1]:.4f},{agg["distortion"][-1]:.3f},{agg["max_distortion"][-1]:.3f},{agg["contractions_std"][-1]:.4f},{agg["expansions_std"][-1]}'

    if args.save_each_epoch:
        loss_report = basic_params + ','.join([f"{agg['train_loss'][i]:.4f}" for i in range(len(agg['train_loss']))])
        distortion_report = basic_params + ','.join([f"{agg['distortion'][i]:.5f}" for i in range(len(agg['distortion']))])
        max_distortion_report = basic_params + ','.join([f"{agg['max_distortion'][i]:.5f}" for i in range(len(agg['max_distortion']))])

    # write to file 
    sim_record_path = 'experiments'
    cluster = args.cluster_code
    with open(os.path.join(sim_record_path, 'sim_records.txt' if cluster == 0 else f'sim_records_{cluster}.txt'), 'a') as f:
        f.write(main_report)
        f.write('\n')
    
    if args.save_each_epoch:
        with open(os.path.join(sim_record_path, 'sim_loss.txt' if cluster == 0 else f'sim_loss_{cluster}.txt'), 'a') as f:
            f.write(loss_report)
            f.write('\n')
        with open(os.path.join(sim_record_path, 'sim_distortion.txt' if cluster == 0 else f'sim_distortion_{cluster}.txt'), 'a') as f:
            f.write(distortion_report)
            f.write('\n')
        with open(os.path.join(sim_record_path, 'sim_max_distortion.txt' if cluster == 0 else f'sim_max_distortion_{cluster}.txt'), 'a') as f:
            f.write(max_distortion_report)
            f.write('\n')
        
        # # for testing only
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
        if not args.no_model_report:
            record_info(agg)
 
if __name__ == '__main__':
    main()
