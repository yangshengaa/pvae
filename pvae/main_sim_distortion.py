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

# force torch float64
dtype = torch.float64
torch.set_default_dtype(dtype) # force torch.64


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


# ==============================================================
# Initialise model, optimizer, dataset loader and loss function
modelC = getattr(models, 'Enc_{}'.format(args.model))
model = modelC(args).to(device).to(torch.float64)  # force 64

# select optimizer
if args.opt == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True, betas=(args.beta1, args.beta2))
elif args.opt == 'riemannian_adam':
    optimizer = geo_optim.RiemannianAdam(model.parameters(), lr=args.lr, amsgrad=True, betas=(args.beta1, args.beta2))
else:
    raise NotImplementedError(f'optimizer {args.optimizer} not supported')


# for specific loss functions, additional learning parameters need to be appended for update 
loss_function_type = args.loss_function
if 'learning' in loss_function_type:
    alpha = Parameter(torch.tensor(1.))  # initialize to 1 
    model.learning_alpha = alpha
    optimizer.add_param_group({'params': alpha})

# =============================================================
# load data
overall_loader, shortest_path_mat = model.getDataLoaders(
    args.batch_size, True, device, dtype, *args.data_params, path_config['dataset_root']
)
loss_function = ae_pairwise_dist_objective  
shortest_path_mat = shortest_path_mat.to(device).to(torch.float64)

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
    ancestral_mask = torch.as_tensor(ancestral_mask, device=device, dtype=torch.float64)


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
        if not args.read_ancestral_mask:
            ( 
                distortion, individual_distortion, max_distortion,
                relative_rate, scaled_rate,
                contractions_std, expansions_std,
                diameter 
            ) = metric_report(emb_dists_selected, real_dists_selected, emb_dists, shortest_path_mat)
        else:
            ( 
                distortion, individual_distortion, max_distortion,
                relative_rate, scaled_rate,
                contractions_std, expansions_std,
                diameter,
                ancestral_distortion, ancestral_individual_distortion, ancestral_max_distortion, 
                non_ancestral_distortion, non_ancestral_individual_distortion, non_ancestral_max_distortion, 
            ) = metric_report(emb_dists_selected, real_dists_selected, emb_dists, shortest_path_mat, ancestral_mask)

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

        if args.read_ancestral_mask: 
            agg['ancestral_distortion'].append(ancestral_distortion)
            agg['ancestral_individual_distortion'].append(ancestral_individual_distortion)
            agg['ancestral_max_distortion'].append(ancestral_max_distortion)
            agg['non_ancestral_distortion'].append(non_ancestral_distortion)
            agg['non_ancestral_individual_distortion'].append(non_ancestral_individual_distortion)
            agg['non_ancestral_max_distortion'].append(non_ancestral_max_distortion)


        # print 
        if epoch % args.save_freq == 0:
            metrics_msg = (
                f'Loss: {b_loss:.4f}, dist: {distortion:.4f}, Idv dist: {individual_distortion:.4f}, Max dist {max_distortion:.2f}, ' +
                f'relative: {relative_rate:.2f}, scaled: {scaled_rate:.2f}, ' +
                f'Contraction Std {contractions_std:.4f}, Expansion Std {expansions_std:.6f}, Diameter {diameter:.2f}'
            )
            if args.read_ancestral_mask:
                metrics_msg += ', '
                metrics_msg += (
                    f'anc dist: {ancestral_distortion:.4f}, anc Idv dist: {ancestral_individual_distortion:.4f}, anc Max dist: {ancestral_max_distortion:.2f}, ' + 
                    f'non_anc dist: {non_ancestral_distortion:.4f}, non_anc Idv dist: {non_ancestral_individual_distortion:.4f}, non_anc Max dist: {non_ancestral_max_distortion:.2f}' 
                )
            print(f'====> Epoch: {epoch:04d}, ', metrics_msg)
    
    # tensorboard visualization 
        if args.log_train:
            with torch.no_grad():
                # visualize 
                trained_emb = reconstructed_data.detach().cpu().numpy()  # feed back to cpu to plot 
                fig = visualize_embeddings(
                    trained_emb,
                    edges,
                    shortest_path_mat,
                    model_save_dir_name,
                    args.manifold,
                    thr,
                    use_translation=args.use_translation,
                    root_idx=0,
                    c=args.c,

                    loss=b_loss, 
                    distortion=distortion, 
                    diameter=diameter 
                )
                img_arr = convert_fig_to_array(fig)
                img_arr = torch.tensor(img_arr)

                # write to tensorboard 
                tb_writer.add_image('embedding/embedding_image', img_arr, epoch, dataformats='HWC')

                # add metrics 
                tb_writer.add_scalar('train/loss', loss, epoch)
                tb_writer.add_scalar('train/distortion', distortion, epoch)
                tb_writer.add_scalar('train/individual_distortion', individual_distortion, epoch)
                tb_writer.add_scalar('train/max_distortion', max_distortion, epoch)
                tb_writer.add_scalar('train/relative', relative_rate, epoch)
                tb_writer.add_scalar('train/scaled', scaled_rate, epoch)
                tb_writer.add_scalar('train/contraction_std', contractions_std, epoch)
                tb_writer.add_scalar('train/expansion_std', expansions_std, epoch)
                tb_writer.add_scalar('train/diameter', diameter, epoch)

                if args.read_ancestral_mask:
                    tb_writer.add_scalar('train/ancestral_distortion', ancestral_distortion, epoch)
                    tb_writer.add_scalar('train/ancestral_individual_distortion', ancestral_individual_distortion, epoch)
                    tb_writer.add_scalar('train/ancestral_max_distortion', ancestral_max_distortion, epoch)
                    tb_writer.add_scalar('train/non_ancestral_distortion', non_ancestral_distortion, epoch)
                    tb_writer.add_scalar('train/non_ancestral_individual_distortion', non_ancestral_individual_distortion, epoch)
                    tb_writer.add_scalar('train/non_ancestral_max_distortion', non_ancestral_max_distortion, epoch)
                
                # flush to disk
                tb_writer.flush()

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

    if args.read_ancestral_mask:
        main_report += ','
        main_report += (
            f'{agg["ancestral_distortion"][-1]:.4f},' + 
            f'{agg["ancestral_individual_distortion"][-1]:.4f},' + 
            f'{agg["ancestral_max_distortion"][-1]:.2f},' + 
            f'{agg["non_ancestral_distortion"][-1]:.4f},' + 
            f'{agg["non_ancestral_individual_distortion"][-1]:.4f},' + 
            f'{agg["non_ancestral_max_distortion"][-1]:.2f}'
        )

    # write to file 
    sim_record_path = path_config['sim_record_path']
    cluster = 0
    with open(os.path.join(sim_record_path, f'sim_records_{args.record_name}.txt' if cluster == 0 else f'sim_records_{args.record_name}_{cluster}.txt'), 'a') as f:
        f.write(main_report)
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
        if args.log_model:
            save_model(model, os.path.join(model_save_dir, 'model.pt'))

        # record simulation results
        if not args.no_model_report:
            record_info(agg)
 
if __name__ == '__main__':
    main()
