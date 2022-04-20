"""
for simulation to test distortion only, copy from main with minor modifications
"""

from email.policy import default
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
parser.add_argument('--use-hyperbolic', help='whether to use hyperbolic distance for outputs', 
                    default=False, type=bool)

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
                    choices=['Linear', 'Wrapped', 'Mob'])
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

# # Create directory for experiment if necessary
# directory_name = 'experiments/{}'.format(args.name)
# if args.name != '.':
#     if not os.path.exists(directory_name):
#         os.makedirs(directory_name)
#     runPath = mkdtemp(prefix=runId, dir=directory_name)
# else:
#     runPath = mkdtemp(prefix=runId, dir=directory_name)
# sys.stdout = Logger('{}/run.log'.format(runPath))
# print('RunID:', runId)

# # Save args to run
# with open('{}/args.json'.format(runPath), 'w') as fp:
#     json.dump(args.__dict__, fp)
# with open('{}/args.txt'.format(runPath), 'w') as fp:
#     git_hash = subprocess.check_output(
#         ['git', 'rev-parse', '--verify', 'HEAD'])
#     command = ' '.join(sys.argv[1:])
#     fp.write(git_hash.decode('utf-8') + command)
# torch.save(args, '{}/args.rar'.format(runPath))

# Initialise model, optimizer, dataset loader and loss function
modelC = getattr(models, 'Enc_{}'.format(args.model))
model = modelC(args).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True, betas=(args.beta1, args.beta2))
overall_loader, shortest_path_dict, shortest_path_mat = model.getDataLoaders(
    args.batch_size, True, device, *args.data_params
)
loss_function = ae_pairwise_dist_objective # getattr(objectives, args.obj + '_objective')
shortest_path_mat = shortest_path_mat.to(device)

# parameters for ae objectives 
use_hyperbolic = args.use_hyperbolic
curvature = torch.Tensor([args.c]).to(device)

def train(epoch, agg):
    model.train()
    b_loss = 0.
    for _, (data, labels) in enumerate(overall_loader):  # no train test split needed, yet  
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss, train_distortion = loss_function(
            model, data, shortest_path_mat, 
            use_hyperbolic=use_hyperbolic, c=curvature
        )
        probe_infnan(loss, "Training loss:")
        loss.backward()
        optimizer.step()

        b_loss += loss.item()

    # agg['train_loss'].append(b_loss)
    agg['distortion'].append(train_distortion)
    agg['train_loss'].append(b_loss)
    if epoch % 1 == 0:
        print(f'====> Epoch: {epoch:03d} Train Loss: {b_loss:.4f}, Train Distortion: {train_distortion:.2f}')


# def test(epoch, agg):
#     model.eval()
#     b_loss = 0. 
#     with torch.no_grad():
#         for _, (data, labels) in enumerate(test_loader):
#             data = data.to(device)
#             labels = labels.to(device)
#             loss, test_distortion = loss_function(
#                 model, data, labels, shortest_path_dict, 
#                 use_hyperbolic=use_hyperbolic, c=curvature
#             )

#             b_loss += loss.item()

#     agg['test_loss'].append(b_loss)
#     print('Test loss: {:.4f}, Test Distortion: {:.2f}'.format(agg['test_loss'][-1], test_distortion))


# def eval_overall(agg):
#     with torch.no_grad():
#         for _, (data, labels) in enumerate(overall_loader):  # gauranteed full batch, run once
#             data = data.to(device)
#             labels = labels.to(device)
#             overall_loss, overall_distortion = loss_function(
#                 model, data, labels, shortest_path_dict,
#                 use_hyperbolic=use_hyperbolic, c=curvature
#             )
#     agg['overall_loss'].append(overall_loss)
#     print(f'Overall loss: {overall_loss:2f}, Overall Distortion: {overall_distortion:.2f}')

def record_info(agg):
    """ record loss and distortion """
    basic_params = f'{args.data_params[0]},{args.data_size[0]},{args.latent_dim},{args.enc},{args.use_hyperbolic},{args.c},'
    main_report = basic_params + f'{agg["train_loss"][-1]:.4f},{agg["distortion"][-1]:.3f}'
    loss_report = basic_params + ','.join([f"{agg['train_loss'][i]:.4f}" for i in range(len(agg['train_loss']))])
    distortion_report = basic_params + ','.join([f"{agg['distortion'][i]:.3f}" for i in range(len(agg['distortion']))])

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
 
if __name__ == '__main__':
    with Timer('ME-VAE') as t:
        # agg = defaultdict(list)
        agg = defaultdict(list)
        print('Starting training...')

        # model.init_last_layer_bias(train_loader)
        for epoch in range(1, args.epochs + 1):
            train(epoch, agg)
            # print(epoch)
            # if args.save_freq == 0 or epoch % args.save_freq == 0:
            #     if not args.skip_test:
            #         test(epoch, agg)
            # save_model(model, runPath + '/model.rar')
            # save_vars(agg, runPath + '/losses.rar')
        # eval_overall(agg)

        # record simulation results
        record_info(agg)
