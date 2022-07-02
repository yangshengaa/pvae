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
from torch.nn.parameter import Parameter
from torch.utils.tensorboard import SummaryWriter

from geoopt import optim as geo_optim

# load files 
sys.path.append(".")
sys.path.append("..")
from utils import Logger, Timer, save_model, save_vars, probe_infnan, load_config
from objectives import ae_pairwise_dist_objective, metric_report
import models
from vis import convert_fig_to_array, visualize_embeddings

torch.backends.cudnn.benchmark = True

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
parser.add_argument('--loss-function', help='type of loss function', default='scaled', type=str, 
                    choices=['raw', 'relative', 'scaled', 'robust_scaled', 'distortion', 'individual_distortion', 'modified_individual_distortion', 'robust_individual_distortion', 'learning_relative'])

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
parser.add_argument('--enc', type=str, default='Wrapped', help='allow to choose different implemented encoder',
                    choices=['Linear', 'Wrapped', 'WrappedNaive', 'WrappedAlt', 'WrappedSinhAlt', 'Mixture', 'WrappedNaive'])
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
parser.add_argument('--model-save-dir', type=str, default='results', help='the path to store model pt and emb vis')


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


# for specific loss functions, additional learning parameters need to be appended for update 
loss_function_type = args.loss_function
if loss_function_type == 'learning_relative':
    alpha = Parameter(torch.tensor(1.))  # initialize to 1 
    model.learning_alpha = alpha
    optimizer.add_param_group({'params': alpha})

# =============================================================
# load data
overall_loader, shortest_path_mat = model.getDataLoaders(
    args.batch_size, True, device, *args.data_params, path_config['dataset_root']
)
loss_function = ae_pairwise_dist_objective  
shortest_path_mat = shortest_path_mat.to(device)

# parameters for ae objectives 
use_hyperbolic = False if args.use_euclidean else True
args.use_hyperbolic = use_hyperbolic
max_radius = np.sqrt(1 / args.c)
thr = args.thr * max_radius  # absolute scale hard boundary
# use_hyperbolic = False
curvature = torch.Tensor([args.c]).to(device)

# ===========================================================
if args.log_train: 
    # encode model 
    model_type = args.enc 
    if model_type == 'Mixture':
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


# ==========================================================
def train(epoch, agg):
    model.train()
    b_loss = 0.
    for _, (data, _) in enumerate(overall_loader):  # no train test split needed, yet  
        # data = data.to(device)
        optimizer.zero_grad()
        (
            reconstructed_data, 
            loss, 
            (emb_dists_selected, real_dists_selected, emb_dists)
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

    # compute metric
    if epoch % args.save_freq == 0 or (epoch % args.log_train_epochs == 0 and args.log_train):
        ( 
            distortion, individual_distortion, max_distortion,
            relative_rate, scaled_rate,
            contractions_std, expansions_std,
            diameter 
        ) = metric_report(emb_dists_selected, real_dists_selected, emb_dists, shortest_path_mat)

        # log 
        agg['train_loss'].append(b_loss)
        agg['distortion'].append(distortion)
        agg['max_distortion'].append(max_distortion)
        agg['individual_distortion'].append(individual_distortion)
        agg['relative'].append(relative_rate)
        agg['scaled'].append(scaled_rate)
        agg['contractions_std'].append(contractions_std)
        agg['expansions_std'].append(expansions_std)
        agg['diameter'].append(diameter)

        # print 
        if epoch % args.save_freq == 0:
            print(f'====> Epoch: {epoch:04d}, ' + 
            f'Loss: {b_loss:.4f}, Distortion: {distortion:.4f}, Idv Distortion: {individual_distortion:.4f}, Max Distortion {max_distortion:.2f}, ' + 
            f'relative: {relative_rate:.2f}, scaled: {scaled_rate:.2f}, ' + 
            f'Contraction Std {contractions_std:.4f}, Expansion Std {expansions_std:.6f}, Diameter {diameter:.2f}')
    
    # tensorboard visualization 
        if args.log_train:
            with torch.no_grad():
                # visualize 
                trained_emb = reconstructed_data.cpu().numpy()  # feed back to cpu to plot 
                fig = visualize_embeddings(trained_emb, edges, model_save_dir_name, loss, diameter, thr, distortion)
                img_arr = convert_fig_to_array(fig)
                img_arr = torch.tensor(img_arr)

                # write to tensorboard 
                tb_writer.add_image('embedding/embedding_image', img_arr, epoch, dataformats='HWC')

                # add metrics 
                tb_writer.add_scalar('loss/loss', loss, epoch)
                tb_writer.add_scalar('distortion/distortion', distortion, epoch)
                tb_writer.add_scalar('individual_distortion/individual_distortion', individual_distortion, epoch)
                tb_writer.add_scalar('max_distortion/max_distortion', max_distortion, epoch)
                tb_writer.add_scalar('relative/relative', relative_rate, epoch)
                tb_writer.add_scalar('scaled/scaled', scaled_rate, epoch)
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
            model_params = f'{args.data_params[0]},{args.data_size[0]},{args.latent_dim},{args.enc},{args.use_hyperbolic},{args.c},{args.loss_function},{args.hidden_dims},{args.num_hyperbolic_layers},{args.no_final_lift},{args.lift_type},{args.no_bn},'
        else:
            model_params = f'{args.data_params[0]},{args.data_size[0]},{args.latent_dim},{args.enc},{args.use_hyperbolic},{args.c},{args.loss_function},{args.no_bn},'
        with open(os.path.join(save_path, model_params + '_data_emb.npy'), 'wb') as f:
            np.save(f, data_emb_np)


def record_info(agg):
    """ record loss and distortion """
    if 'Mixture' in args.enc:
        basic_params = f'{args.data_params[0]},{args.data_size[0]},{args.latent_dim},{args.enc},{args.use_hyperbolic},{args.c},{args.loss_function},\"{args.hidden_dims}\",{args.num_hyperbolic_layers},{args.no_final_lift},{args.lift_type},{args.opt},{args.no_bn},'
    else:
        basic_params = f'{args.data_params[0]},{args.data_size[0]},{args.latent_dim},{args.enc},{args.use_hyperbolic},{args.c},{args.loss_function},{args.no_bn},'
    main_report = (
        basic_params + 
        f'{agg["train_loss"][-1]:.4f},' + 
        f'{agg["distortion"][-1]:.4f},' + 
        f'{agg["individual_distortion"][-1]:.4f},' + 
        f'{agg["max_distortion"][-1]:.3f},' + 
        f'{agg["relative"][-1]:.4f},' + 
        f'{agg["scaled"][-1]:.4f},' + 
        f'{agg["contractions_std"][-1]:.4f},' + 
        f'{agg["expansions_std"][-1]:.6f},' + 
        f'{agg["diameter"][-1]:.2f}'
    )

    # if args.save_each_epoch:
    #     loss_report = basic_params + ','.join([f"{agg['train_loss'][i]:.4f}" for i in range(len(agg['train_loss']))])
    #     distortion_report = basic_params + ','.join([f"{agg['distortion'][i]:.5f}" for i in range(len(agg['distortion']))])
    #     max_distortion_report = basic_params + ','.join([f"{agg['max_distortion'][i]:.5f}" for i in range(len(agg['max_distortion']))])
    #     individual_distortion_report = basic_params + ','.join([f"{agg['individual_distortion'][i]:.5f}" for i in range(len(agg['individual_distortion']))])

    # write to file 
    sim_record_path = path_config['sim_record_path']
    cluster = 0
    with open(os.path.join(sim_record_path, f'sim_records_{args.record_name}.txt' if cluster == 0 else f'sim_records_{args.record_name}_{cluster}.txt'), 'a') as f:
        f.write(main_report)
        f.write('\n')
    
    # if args.save_each_epoch:
    #     with open(os.path.join(sim_record_path, f'sim_loss_{args.record_name}.txt' if cluster == 0 else f'sim_loss_{args.record_name}_{cluster}.txt'), 'a') as f:
    #         f.write(loss_report)
    #         f.write('\n')
    #     with open(os.path.join(sim_record_path, f'sim_distortion_{args.record_name}.txt' if cluster == 0 else f'sim_distortion_{args.record_name}_{cluster}.txt'), 'a') as f:
    #         f.write(distortion_report)
    #         f.write('\n')
    #     with open(os.path.join(sim_record_path, f'sim_max_distortion_{args.record_name}.txt' if cluster == 0 else f'sim_max_distortion_{args.record_name}_{cluster}.txt'), 'a') as f:
    #         f.write(max_distortion_report)
    #         f.write('\n')
    #     with open(os.path.join(sim_record_path, f'sim_individual_distortion_{args.record_name}.txt' if cluster == 0 else f'sim_individual_distortion_{args.record_name}_{cluster}.txt'), 'a') as f:
    #         f.write(individual_distortion_report)
    #         f.write('\n')
        

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
