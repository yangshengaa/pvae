"""
for simulation to test distortion only, copy from main with minor modifications
"""

# load packages 
import os
import sys
import argparse
from collections import defaultdict

import numpy as np

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from geoopt import optim as geo_optim

# load files 
sys.path.append(".")
sys.path.append("..")
from utils import Logger, Timer, save_model, save_vars, probe_infnan
from objectives import ae_pairwise_dist_objective
import models
from vis import convert_fig_to_array, visualize_embeddings

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
parser.add_argument('--loss-function', help='type of loss function', default='scaled', type=str, 
                    choices=['raw', 'relative', 'scaled', 'distortion', 'individual_distortion'])

### Model
parser.add_argument('--latent-dim', type=int, default=10,
                    metavar='L', help='latent dimensionality (default: 10)')
parser.add_argument('--c', type=float, default=1., help='curvature')
parser.add_argument('--thr', type=float, default=0.99, help='hard boundary of Poincare Ball')
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
                    choices=['Linear', 'Wrapped', 'WrappedAlt', 'WrappedSinhAlt', 'Mixture', 'Mob'])
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


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
args.prior_iso = args.prior_iso or args.posterior == 'RiemannianNormal'
args.batch_size = 1  # dummy variable, useless since all are trained using a full batch


# ==============================================================
# Initialise model, optimizer, dataset loader and loss function
modelC = getattr(models, 'Enc_{}'.format(args.model))
model = modelC(args).to(device)

# select optimizer
if args.opt == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True, betas=(args.beta1, args.beta2))
elif args.opt == 'riemannian_adam':
    optimizer = geo_optim.RiemannianAdam(model.parameters(), lr=args.lr, amsgrad=True, betas=(args.beta1, args.beta2))
else:
    raise NotImplementedError(f'optimizer {args.optimizer} not supported')


# =============================================================
# load data
overall_loader, shortest_path_mat = model.getDataLoaders(
    args.batch_size, True, device, *args.data_params
)
loss_function = ae_pairwise_dist_objective 
shortest_path_mat = shortest_path_mat.to(device)

# parameters for ae objectives 
use_hyperbolic = False if args.use_euclidean else True
args.use_hyperbolic = use_hyperbolic
thr = args.thr
# use_hyperbolic = False
curvature = torch.Tensor([args.c]).to(device)
loss_function_type = args.loss_function


# ===========================================================
if args.log_train: 
    # encode model 
    model_type = args.enc 
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
    model_save_dir = os.path.join('results', model_save_dir_name)

    # load edges and color encoding 
    with open(os.path.join('data', args.data_params[0], 'sim_tree_edges.npy'), 'rb') as f:
        edges = np.load(f)

    # TODO: make and load color encoding 
    # Initialize tensorboard writer
    tb_writer = SummaryWriter(log_dir=model_save_dir)


# ==========================================================
def train(epoch, agg):
    model.train()
    b_loss = 0.
    for _, (data, _) in enumerate(overall_loader):  # no train test split needed, yet  
        data = data.to(device)
        optimizer.zero_grad()
        (
            reconstructed_data, 
            loss, 
            train_distortion, 
            train_max_distortion, 
            train_individual_distortion, 
            contractions_std, 
            expansions_std,
            diameter
        ) = loss_function(
            model, data, shortest_path_mat, 
            use_hyperbolic=use_hyperbolic, c=curvature,
            loss_function_type=loss_function_type, 
            thr=thr
        )
        probe_infnan(loss, "Training loss:")
        loss.backward()
        optimizer.step()

        b_loss += loss.item()

    # agg['train_loss'].append(b_loss)
    agg['distortion'].append(train_distortion)
    agg['max_distortion'].append(train_max_distortion)
    agg['individual_distortion'].append(train_individual_distortion)
    agg['train_loss'].append(b_loss)
    agg['contractions_std'].append(contractions_std)
    agg['expansions_std'].append(expansions_std)
    if epoch % args.save_freq == 0:
        print(f'====> Epoch: {epoch:04d} Loss: {b_loss:.4f}, Distortion: {train_distortion:.4f}, Idv Distortion: {train_individual_distortion:.4f}, Max Distortion {train_max_distortion:.2f}' + 
        f', Contraction Std {contractions_std:.4f}, Expansion Std {expansions_std:.6f}, Diameter {diameter:.2f}')
    
    # ! testing purpose 
    if args.log_train:
        with torch.no_grad():
            if epoch % args.log_train_epochs == 0:
                # visualize 
                trained_emb = reconstructed_data.cpu().numpy()  # feed back to cpu to plot 
                fig = visualize_embeddings(trained_emb, edges, model_save_dir_name, loss, diameter, thr)
                img_arr = convert_fig_to_array(fig)
                img_arr = torch.tensor(img_arr)

                # write to tensorboard 
                tb_writer.add_image('embedding/embedding_image', img_arr, epoch, dataformats='HWC')

                # add metrics 
                tb_writer.add_scalar('loss/loss', loss, epoch)
                tb_writer.add_scalar('distortion/distortion', train_distortion, epoch)
                tb_writer.add_scalar('individual_distortion/individual_distortion', train_individual_distortion, epoch)
                tb_writer.add_scalar('max_distortion/max_distortion', train_max_distortion, epoch)
                tb_writer.add_scalar('contraction_std/contraction_std', contractions_std, epoch)
                tb_writer.add_scalar('expansion_std/expansion_std', expansions_std, epoch)
                tb_writer.add_scalar('diameter/diameter', diameter, epoch)
                
                # flush to disk
                tb_writer.flush()


def save_emb():
    """ save final embeddings """
    with torch.no_grad():
        for _, (data, _) in enumerate(overall_loader):
            data_emb = model(data).squeeze()

        # clip
        if not args.no_final_clip:
            data_emb = data_emb * torch.clamp(thr / torch.linalg.norm(data_emb, dim=1, keepdim=True), max=1)
        data_emb_np = data_emb.numpy()
        
        # save 
        save_path = 'experiments'
        if 'Mixture' in args.enc: 
            model_params = f'{args.data_params[0]},{args.data_size[0]},{args.latent_dim},{args.enc},{args.use_hyperbolic},{args.c},{args.loss_function},{args.hidden_dims},{args.num_hyperbolic_layers},{args.no_final_lift},{args.lift_type}'
        else:
            model_params = f'{args.data_params[0]},{args.data_size[0]},{args.latent_dim},{args.enc},{args.use_hyperbolic},{args.c},{args.loss_function}'
        with open(os.path.join(save_path, model_params + '_data_emb.npy'), 'wb') as f:
            np.save(f, data_emb_np)


def record_info(agg):
    """ record loss and distortion """
    if 'Mixture' in args.enc:
        basic_params = f'{args.data_params[0]},{args.data_size[0]},{args.latent_dim},{args.enc},{args.use_hyperbolic},{args.c},{args.loss_function},\"{args.hidden_dims}\",{args.num_hyperbolic_layers},{args.no_final_lift},{args.lift_type},{args.opt},'
    else:
        basic_params = f'{args.data_params[0]},{args.data_size[0]},{args.latent_dim},{args.enc},{args.use_hyperbolic},{args.c},{args.loss_function},'
    main_report = basic_params + f'{agg["train_loss"][-1]:.4f},{agg["distortion"][-1]:.4f},{agg["individual_distortion"][-1]:.4f},{agg["max_distortion"][-1]:.3f},{agg["contractions_std"][-1]:.4f},{agg["expansions_std"][-1]}'

    if args.save_each_epoch:
        loss_report = basic_params + ','.join([f"{agg['train_loss'][i]:.4f}" for i in range(len(agg['train_loss']))])
        distortion_report = basic_params + ','.join([f"{agg['distortion'][i]:.5f}" for i in range(len(agg['distortion']))])
        max_distortion_report = basic_params + ','.join([f"{agg['max_distortion'][i]:.5f}" for i in range(len(agg['max_distortion']))])
        individual_distortion_report = basic_params + ','.join([f"{agg['individual_distortion'][i]:.5f}" for i in range(len(agg['individual_distortion']))])

    # write to file 
    sim_record_path = 'experiments'
    cluster = 0
    with open(os.path.join(sim_record_path, f'sim_records_{args.record_name}.txt' if cluster == 0 else f'sim_records_{args.record_name}_{cluster}.txt'), 'a') as f:
        f.write(main_report)
        f.write('\n')
    
    if args.save_each_epoch:
        with open(os.path.join(sim_record_path, f'sim_loss_{args.record_name}.txt' if cluster == 0 else f'sim_loss_{args.record_name}_{cluster}.txt'), 'a') as f:
            f.write(loss_report)
            f.write('\n')
        with open(os.path.join(sim_record_path, f'sim_distortion_{args.record_name}.txt' if cluster == 0 else f'sim_distortion_{args.record_name}_{cluster}.txt'), 'a') as f:
            f.write(distortion_report)
            f.write('\n')
        with open(os.path.join(sim_record_path, f'sim_max_distortion_{args.record_name}.txt' if cluster == 0 else f'sim_max_distortion_{args.record_name}_{cluster}.txt'), 'a') as f:
            f.write(max_distortion_report)
            f.write('\n')
        with open(os.path.join(sim_record_path, f'sim_individual_distortion_{args.record_name}.txt' if cluster == 0 else f'sim_individual_distortion_{args.record_name}_{cluster}.txt'), 'a') as f:
            f.write(individual_distortion_report)
            f.write('\n')
        

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
