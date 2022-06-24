"""
for simulation to test distortion only, copy from main with minor modifications

train test driver 
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
from utils import Logger, Timer, save_model, save_vars, probe_infnan, load_config
from objectives import ae_pairwise_dist_objective, metric_report
import models
from vis import convert_fig_to_array, visualize_train_test_embeddings

torch.backends.cudnn.benchmark = True

# path config 
path_config = load_config(dataset_key='train_test')

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

# * note: some arguments does not make sense in vanilla autoencoder, 
# * kept for future development 

### General
parser.add_argument('--model',          type=str,   metavar='M',      help='model name')
parser.add_argument('--manifold',       type=str,   default='PoincareBall',   choices=['Euclidean', 'PoincareBall'])
parser.add_argument('--save-freq',      type=int,   default=100,      help='print objective values every value (if positive)')

### Dataset
parser.add_argument('--data-params',    nargs='+',  default=[],       help='parameters which are passed to the dataset loader')
parser.add_argument('--train-test-index', type=str, default=1,        help='the train test index of a dataset')
parser.add_argument('--data-size',      type=int,   nargs='+',        default=[], help='size/shape of data observations')

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
                    choices=['raw', 'relative', 'scaled', 'robust_scaled', 'distortion', 'individual_distortion', 'modified_individual_distortion', 'robust_individual_distortion'])

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
parser.add_argument('--nl', type=str, default='ReLU', help='non linearity')
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
train_loader, train_shortest_path_mat = model.getDataLoaders(
    args.batch_size, True, device, *args.data_params, path_config['dataset_root'], f'train_{args.train_test_index}'
)
test_loader, test_shortest_path_mat = model.getDataLoaders(
    args.batch_size, True, device, *args.data_params, path_config['dataset_root'], f'test_{args.train_test_index}'
)
loss_function = ae_pairwise_dist_objective  
train_shortest_path_mat = train_shortest_path_mat.to(device)
test_shortest_path_mat = test_shortest_path_mat.to(device)

# parameters for ae objectives 
use_hyperbolic = False if args.use_euclidean else True
args.use_hyperbolic = use_hyperbolic
max_radius = np.sqrt(1 / args.c)
thr = args.thr * max_radius  # absolute scale hard boundary
# use_hyperbolic = False
curvature = torch.Tensor([args.c]).to(device)
loss_function_type = args.loss_function


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
    with open(os.path.join(path_config['dataset_root'], args.data_params[0], f'sim_tree_edges_train_{args.train_test_index}.npy'), 'rb') as f:
        train_edges = np.load(f)
    with open(os.path.join(path_config['dataset_root'], args.data_params[0], f'sim_tree_edges_test_{args.train_test_index}.npy'), 'rb') as f:
        test_edges = np.load(f)

    # TODO: make and load color encoding 
    # Initialize tensorboard writer
    tb_writer = SummaryWriter(log_dir=model_save_dir)


# ==========================================================
def train(epoch, agg):
    model.train()
    b_loss = 0.
    for _, (data, _) in enumerate(train_loader):  # no train test split needed, yet  
        optimizer.zero_grad()
        (
            train_reconstructed_data, 
            train_loss, 
            (train_emb_dists_selected, train_real_dists_selected, train_emb_dists)
        ) = loss_function(
            model, data, train_shortest_path_mat, 
            use_hyperbolic=use_hyperbolic, c=curvature,
            loss_function_type=loss_function_type, 
            thr=thr
        )
        probe_infnan(train_loss, "Training loss:")
        train_loss.backward()
        optimizer.step()

        b_loss += train_loss.item()

    # compute metric
    if epoch % args.save_freq == 0 or (epoch % args.log_train_epochs == 0 and args.log_train):
        ( 
            tr_distortion, tr_individual_distortion, tr_max_distortion,
            tr_relative_rate, tr_scaled_rate,
            tr_contractions_std, tr_expansions_std,
            tr_diameter 
        ) = metric_report(train_emb_dists_selected, train_real_dists_selected, train_emb_dists, train_shortest_path_mat)

        # log 
        agg['train_loss'].append(b_loss)
        agg['train_distortion'].append(tr_distortion)
        agg['train_max_distortion'].append(tr_max_distortion)
        agg['train_individual_distortion'].append(tr_individual_distortion)
        agg['train_relative'].append(tr_relative_rate)
        agg['train_scaled'].append(tr_scaled_rate)
        agg['train_contractions_std'].append(tr_contractions_std)
        agg['train_expansions_std'].append(tr_expansions_std)
        agg['train_diameter'].append(tr_diameter)

        # print 
        if epoch % args.save_freq == 0:
            print(f'====> Epoch: {epoch:04d}, ' + 
            f'Loss: {b_loss:.4f}, Distortion: {tr_distortion:.4f}, Idv Distortion: {tr_individual_distortion:.4f}, Max Distortion: {tr_max_distortion:.2f}, ' + 
            f'relative: {tr_relative_rate:.2f}, scaled: {tr_scaled_rate:.2f}, ' + 
            f'Contraction Std: {tr_contractions_std:.4f}, Expansion Std: {tr_expansions_std:.6f}, Diameter: {tr_diameter:.2f}')
        
        # validation/test 
        model.eval()
        with torch.no_grad():
            test_b_loss = 0.
            for data, _ in test_loader:
                # feed 
                (
                    test_reconstructed_data, 
                    test_loss, 
                    (test_emb_dists_selected, test_real_dists_selected, test_emb_dists)
                ) = loss_function(
                    model, data, test_shortest_path_mat, 
                    use_hyperbolic=use_hyperbolic, c=curvature,
                    loss_function_type=loss_function_type, 
                    thr=thr
                )
                probe_infnan(test_loss, "Testing loss:")

                test_b_loss += test_loss

                # metric 
                ( 
                    ts_distortion, ts_individual_distortion, ts_max_distortion,
                    ts_relative_rate, ts_scaled_rate,
                    ts_contractions_std, ts_expansions_std,
                    ts_diameter 
                ) = metric_report(test_emb_dists_selected, test_real_dists_selected, test_emb_dists, test_shortest_path_mat)

                # log 
                agg['test_loss'].append(test_b_loss)
                agg['test_distortion'].append(ts_distortion)
                agg['test_max_distortion'].append(ts_max_distortion)
                agg['test_individual_distortion'].append(ts_individual_distortion)
                agg['test_relative'].append(ts_relative_rate)
                agg['test_scaled'].append(ts_scaled_rate)
                agg['test_contractions_std'].append(ts_contractions_std)
                agg['test_expansions_std'].append(ts_expansions_std)
                agg['test_diameter'].append(ts_diameter)

                print(f'====> Epoch: {epoch:04d}, ' + 
                    f'Loss: {test_b_loss:.4f}, Distortion: {ts_distortion:.4f}, Idv Distortion: {ts_individual_distortion:.4f}, Max Distortion: {ts_max_distortion:.2f}, ' + 
                    f'relative: {ts_relative_rate:.2f}, scaled: {ts_scaled_rate:.2f}, ' + 
                    f'Contraction Std: {ts_contractions_std:.4f}, Expansion Std: {ts_expansions_std:.6f}, Diameter: {ts_diameter:.2f}\n')


        # tensorboard visualization 
        if args.log_train:
            with torch.no_grad():
                # visualize 
                trained_emb = train_reconstructed_data.cpu().detach().numpy()  # feed back to cpu to plot 
                test_emb = test_reconstructed_data.cpu().detach().numpy()
                fig = visualize_train_test_embeddings(
                    model_save_dir_name,
                    trained_emb, train_edges, b_loss, tr_diameter,
                    test_emb, test_edges, test_b_loss, ts_diameter,
                    thr
                )
                img_arr = convert_fig_to_array(fig)
                img_arr = torch.tensor(img_arr)

                # write to tensorboard 
                tb_writer.add_image('embedding/embedding_image', img_arr, epoch, dataformats='HWC')

                # add metrics 
                tb_writer.add_scalar('train/loss', train_loss, epoch)
                tb_writer.add_scalar('train/distortion', tr_distortion, epoch)
                tb_writer.add_scalar('train/individual_distortion', tr_individual_distortion, epoch)
                tb_writer.add_scalar('train/max_distortion', tr_max_distortion, epoch)
                tb_writer.add_scalar('train/relative', tr_relative_rate, epoch)
                tb_writer.add_scalar('train/scaled', tr_scaled_rate, epoch)
                tb_writer.add_scalar('train/contraction_std', tr_contractions_std, epoch)
                tb_writer.add_scalar('train/expansion_std', tr_expansions_std, epoch)
                tb_writer.add_scalar('train/diameter', tr_diameter, epoch)

                tb_writer.add_scalar('test/loss', test_b_loss, epoch)
                tb_writer.add_scalar('test/distortion', ts_distortion, epoch)
                tb_writer.add_scalar('test/individual_distortion', ts_individual_distortion, epoch)
                tb_writer.add_scalar('test/max_distortion', ts_max_distortion, epoch)
                tb_writer.add_scalar('test/relative', ts_relative_rate, epoch)
                tb_writer.add_scalar('test/scaled', ts_scaled_rate, epoch)
                tb_writer.add_scalar('test/contraction_std', ts_contractions_std, epoch)
                tb_writer.add_scalar('test/expansion_std', ts_expansions_std, epoch)
                tb_writer.add_scalar('test/diameter', ts_diameter, epoch)
                
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
        f'{agg["train_distortion"][-1]:.4f},' + 
        f'{agg["train_individual_distortion"][-1]:.4f},' + 
        f'{agg["train_max_distortion"][-1]:.3f},' + 
        f'{agg["train_relative"][-1]:.4f},' + 
        f'{agg["train_scaled"][-1]:.4f},' + 
        f'{agg["train_contractions_std"][-1]:.4f},' + 
        f'{agg["train_expansions_std"][-1]:.6f},' + 
        f'{agg["train_diameter"][-1]:.2f}' + 

        f'{agg["train_loss"][-1]:.4f},' + 
        f'{agg["test_distortion"][-1]:.4f},' + 
        f'{agg["test_individual_distortion"][-1]:.4f},' + 
        f'{agg["test_max_distortion"][-1]:.3f},' + 
        f'{agg["test_relative"][-1]:.4f},' + 
        f'{agg["test_scaled"][-1]:.4f},' + 
        f'{agg["test_contractions_std"][-1]:.4f},' + 
        f'{agg["test_expansions_std"][-1]:.6f},' + 
        f'{agg["test_diameter"][-1]:.2f}'
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
        
        # record simulation results
        if not args.no_model_report:
            record_info(agg)
 
if __name__ == '__main__':
    main()
