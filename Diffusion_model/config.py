"""
Configuration and argument parsing for latent diffusion model training.

This module defines all CLI arguments for train.py, evaluate.py, and gridsearch_diffusion.py.

Key argument groups:
    - Dataset: root-dir, batch-size, augment, use-3d, num-slices
    - Training: learning-rate, num-epochs, cost-function, predictor-type
    - Model: in-channels, out-channels, features, attention
    - Physics losses: lambda-div, lambda-flow, lambda-smooth, lambda-laplacian
    - VAE paths: vae-path, vae-encoder-path, vae-decoder-path

Example usage:
    python train.py --root-dir data/dataset --in-channels 17 --out-channels 8 \
        --vae-encoder-path VAE_model/trained/stage2 \
        --vae-decoder-path VAE_model/trained/stage1 \
        --features 64 128 256 512 1024 --attention "3..2" --batch-size 3
"""

import argparse
import os
import os.path as osp
from datetime import datetime

import torch


def str_to_bool(value):
    """Convert string to boolean for argparse."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    if value.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")


parser = argparse.ArgumentParser()

parser.add_argument(
    '--name',
    type=str,
    default='unet',
    help='Arbitrary title describing the dataset used or model being trained.'
)
parser.add_argument(
    '--save-dir',
    type=str,
    default='Diffusion_model/trained/',
    help='Directory where to save results.'
)
parser.add_argument(
    '--mode',
    type=str,
    default='train',
    choices=['train', 'CV', 'optimize'],
    help='Flag indicating whether to train model, cross-validate model, or perform parameter optimization.'
)


group_dataset = parser.add_argument_group(
    'Dataset Parameters',
    'Parameters for loading dataset.'
)
group_train = parser.add_argument_group(
    'Training Parameters',
    'Parameters related to model training.'
)
group_optim = parser.add_argument_group(
    'Optimization Parameters',
    'Parameters related to model optimization.'
)

"""Dataset"""

group_dataset.add_argument(
    '--root-dir',
    type=str,
    required=True,
    help='Directory for dataset.'
)
group_dataset.add_argument(
    '--batch-size',
    type=int,
    default=10,
    help='Batch size.'
)
group_dataset.add_argument(
    '--augment',
    type=str_to_bool,
    default=False,
    help='Whether to augment dataset (e.g., by flipping images).'
)
group_dataset.add_argument(
    '--shuffle',
    type=str_to_bool,
    default=False,
    help='Whether to shuffle data during training.'
)
group_dataset.add_argument(
    '--k-folds',
    type=int,
    default=5,
    help='Number of folds when splitting dataset.'
)


"""Training"""

group_train.add_argument(
    '--device',
    type=str,
    default=None,
    help='Device (e.g., cpu, cuda) on which to train neural network.'
)
group_train.add_argument(
    '--learning-rate',
    type=float,
    default=1e-4,
    help='Learning rate.'
)
group_train.add_argument(
    '--weight-decay',
    type=float,
    default=0.0,
    help='Weight decay (L2 penalty).'
)
group_train.add_argument(
    '--scheduler-flag',
    type=str_to_bool,
    default=False,
    help='Whether to use learning rate scheduler.'
)
group_train.add_argument(
    '--scheduler-gamma',
    type=float,
    default=0.95499, # 0.977
    help='If `--scheduler-flag` is True, multiplicative factor of learning rate decay (in ExponentialLR).'
)
group_train.add_argument(
    '--num-epochs',
    type=int,
    default=100,
    help='Number of epochs.'
)
group_train.add_argument(
    '--cost-function',
    type=str,
    default='normalized_mse_loss_per_component',
    choices=['normalized_mae_loss', 'normalized_mse_loss', 'mae_loss', 'mse_loss', 'huber_loss', 
             'normalized_mae_loss_per_component', 'mae_loss_per_component', 
             'mse_loss_per_component', 'normalized_mse_loss_per_component'],
    help='Cost function for training. Default: normalized_mae_loss (scale-invariant). '
         'Use per_component variants for better w-component learning.'
)
group_train.add_argument(
    '--lambda-div',
    type=float,
    default=0.0,
    help='Weight for divergence/mass conservation loss (physics-based penalty). Recommended: 0.01'
)
group_train.add_argument(
    '--lambda-flow',
    type=float,
    default=0.0,
    help='Weight for flow-rate consistency loss (constant flux constraint). Recommended: 0.001'
)
group_train.add_argument(
    '--lambda-smooth',
    type=float,
    default=0.0,
    help='Weight for gradient smoothness regularization. Recommended: 0.001'
)
group_train.add_argument(
    '--lambda-laplacian',
    type=float,
    default=0.0,
    help='Weight for Laplacian smoothness loss (reduces high-freq noise). Recommended: 0.0001'
)
group_train.add_argument(
    '--physics-loss-freq',
    type=int,
    default=1,
    help='Compute physics loss every N batches (1=every batch, higher=less frequent for speed)'
)
group_train.add_argument(
    '--weight-u',
    type=float,
    default=1.0,
    help='Weight for u (vx) component in velocity loss. Default: 1.0'
)
group_train.add_argument(
    '--weight-v',
    type=float,
    default=1.0,
    help='Weight for v (vy) component in velocity loss. Default: 1.0'
)
group_train.add_argument(
    '--weight-w',
    type=float,
    default=1.0,
    help='Weight for w (vz) component in velocity loss. Increase to boost w learning (e.g., 3.0-10.0)'
)
group_train.add_argument(
    '--lambda-velocity',
    type=float,
    default=0.0,
    help='Weight for auxiliary velocity reconstruction loss (computed on decoded output). Recommended: 0.1-1.0'
)
group_train.add_argument(
    '--velocity-loss-primary',
    type=str_to_bool,
    default=False,
    help='If True, use per-channel velocity loss as PRIMARY loss instead of noise prediction. Slower but directly optimizes velocity channels.'
)

group_train.add_argument(
    '--predictor-type',
    type=str,
    default='latent-diffusion',
    choices=['latent-diffusion'],
    help='Type of ML predictor (for the velocity or pressure field)'
)
group_train.add_argument(
    '--model-name',
    type=str,
    default='UNet',
    help='Neural network model'
)
group_train.add_argument(
    '--in-channels',
    type=int,
    required=True,
    help='Number of channels in input data.'
)
group_train.add_argument(
    '--out-channels',
    type=int,
    required=True,
    help='Number of channels in output data.'
)
group_train.add_argument(
    '--features',
    type=int,
    nargs='+',
    default=[64, 128, 256, 512, 1024],
    help='Number of channels at each (depth) level in the U-Net architecture.'
)
group_train.add_argument(
    '--kernel-size',
    type=int,
    default=3,
    help='Kernel size for convolutional layers.'
)
group_train.add_argument(
    '--padding-mode',
    type=str,
    default='zeros',
    help='Type of padding for convolutional layers.'
)
group_train.add_argument(
    '--activation',
    type=str,
    default='silu',
    choices=['silu', 'relu', 'leakyrelu','softplus'],
    help='Activation functions inside neural network.'
)
group_train.add_argument(
    '--final-activation',
    type=str,
    default=None,
    choices=['silu', 'relu', 'leakyrelu','softplus'],
    help='Activation function before ouput.'
)
group_train.add_argument(
    '--attention',
    type=str,
    default='',
    help='Expression determining the use of attention in U-Net model (e.g., "4..1"). For details, see model documentation.'
)
group_train.add_argument(
    '--dropout',
    type=float,
    default=0.0,
    help='Dropout probability.'
)
group_train.add_argument(
    '--distance-transform',
    type=str_to_bool,
    default=True,
    help='Whether to use distance transform for input image.'
)
group_train.add_argument(
    '--vae-path',
    type=str,
    default=None,
    help='Path to pre-trained VAE model (required for latent-diffusion predictor).'
)
group_train.add_argument(
    '--vae-encoder-path',
    type=str,
    default=None,
    help='Path to VAE encoder weights (optional, for dual VAE: use Stage 2 E2D). If not provided, uses --vae-path.'
)
group_train.add_argument(
    '--vae-decoder-path',
    type=str,
    default=None,
    help='Path to VAE decoder weights (optional, for dual VAE: use Stage 1 D3D). If not provided, uses --vae-path.'
)
group_train.add_argument(
    '--num-slices',
    type=int,
    default=11,
    help='Number of 2D slices in 3D flow field (for latent-diffusion predictor).'
)
group_train.add_argument(
    '--use-3d',
    type=str_to_bool,
    default=True,
    help='Whether to use 3D velocity data from dataset.'
)
group_train.add_argument(
    '--num-timesteps',
    type=int,
    default=1000,
    help='Number of diffusion timesteps. Set to 1 for direct one-step prediction (no iterative denoising).'
)


"""Optimization Parameters"""

group_optim.add_argument(
    '--n-trials',
    type=int,
    default=100,
    help='Number of trials for optimization algorithm.'
)
group_optim.add_argument(
    '--range-batch-size',
    type=int,
    default=[10, 40],
    nargs=2,
    help='Range for batch size.'
)
group_optim.add_argument(
    '--range-kernel-size',
    type=int,
    default=[3, 7],
    nargs=2,
    help='Range for kernel size.'
)
group_optim.add_argument(
    '--range-level',
    type=int,
    default=[1, 7],
    nargs=2,
    help='Number of levels in U-Net.'
)
group_optim.add_argument(
    '--top-bottom',
    type=str_to_bool,
    default=True,
    nargs=2,
    help='If "True", define channel sizes from top-to-bottom. If "False", then proceed from bottom-to-top.'
)
group_optim.add_argument(
    '--top-feature-channels',
    type=int,
    default=32,
    help='Number of feature channels at top-level of U-Net.'
)
group_optim.add_argument(
    '--bottom-feature-channels',
    type=int,
    default=2048,
    help='Number of feature channels at bottom-level of U-Net.'
)
group_optim.add_argument(
    '--range-learning-rate',
    type=float,
    default=[1e-7, 1e-3],
    nargs=2,
    help='Range for learning rate.'
)



def process_args(args: argparse.Namespace):
    """
    Process command line arguments into a dictionary.

    `args`: command line arguments.
    """

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    param_dict = {
        'name': args.name,
        'mode': args.mode,
        'save_dir': args.save_dir,

        'dataset': {
            'root_dir': args.root_dir,
            'batch_size': args.batch_size,
            'augment': args.augment,
            'shuffle': args.shuffle,
            'k_folds': args.k_folds,
            'use_3d': args.use_3d
        },
        'training': {
            'device': args.device,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'scheduler': {
                'flag': args.scheduler_flag,
                'gamma': args.scheduler_gamma,
            },
            'num_epochs': args.num_epochs,
            'cost_function': args.cost_function,
            'lambda_div': args.lambda_div,
            'lambda_flow': args.lambda_flow,
            'lambda_smooth': args.lambda_smooth,
            'lambda_laplacian': args.lambda_laplacian,
            'physics_loss_freq': args.physics_loss_freq,
            'weight_u': args.weight_u,
            'weight_v': args.weight_v,
            'weight_w': args.weight_w,
            'lambda_velocity': args.lambda_velocity,
            'velocity_loss_primary': args.velocity_loss_primary,
            'predictor_type': args.predictor_type,
            'predictor': {
                'model_name':args.model_name,
                'model_kwargs': {
                    'in_channels': args.in_channels,
                    'out_channels': args.out_channels,
                    'features': args.features,
                    'kernel_size': args.kernel_size,
                    'padding_mode': args.padding_mode,
                    'activation': args.activation,
                    'final_activation': args.final_activation,
                    'attention': args.attention,
                    'dropout': args.dropout
                },
                'distance_transform': args.distance_transform,
                'vae_path': args.vae_path,
                #'vae_encoder_path': args.vae_encoder_path,
                #'vae_decoder_path': args.vae_decoder_path,
                'num_slices': args.num_slices,
                'num_timesteps': args.num_timesteps
            }
        },
        'optimization': {
            'n_trials': args.n_trials,
            'range_batch_size': args.range_batch_size,
            'range_kernel_size': args.range_kernel_size,
            'range_level': args.range_level,
            'range_learning_rate': args.range_learning_rate,
            'top_bottom': args.top_bottom,
            'top_feature_channels': args.top_feature_channels,
            'bottom_feature_channels': args.bottom_feature_channels
        }
    }
    return param_dict


def make_log_folder(param_dict: dict):
    """
    Create folder where to results.

    `param_dict`: dictionary with parameters.
    """

    name = param_dict['name']
    save_dir = param_dict['save_dir']

    dataset_kwargs = param_dict['dataset']
    train_kwargs = param_dict['training']

    batch_size = dataset_kwargs['batch_size']

    learning_rate = train_kwargs['learning_rate']
    num_epochs = train_kwargs['num_epochs']

    predictor_type = train_kwargs['predictor_type']
    predictor_kwargs = train_kwargs['predictor']
    in_channels = predictor_kwargs['model_kwargs']['in_channels']
    out_channels = predictor_kwargs['model_kwargs']['out_channels']
    features = predictor_kwargs['model_kwargs']['features']
    kernel_size = predictor_kwargs['model_kwargs']['kernel_size']
    padding_mode = predictor_kwargs['model_kwargs']['padding_mode']
    attention = predictor_kwargs['model_kwargs']['attention']
    dropout = predictor_kwargs['model_kwargs']['dropout']
    weight_decay = train_kwargs['weight_decay']


    # Create log folder
    time_stamp = datetime.now().strftime("%Y%m%d-%H%M")
    
    descr_str = f'in-{in_channels}-out-{out_channels}-' \
        f'f-{len(features)}-k-{kernel_size}-p-{padding_mode}-a-{attention}-' \
        f'dr-{dropout}-wd-{weight_decay:.2e}-' \
        f'b-{batch_size}-lr-{learning_rate:.2e}-ep-{num_epochs}'
    
    sub_dir = time_stamp + f'_{name}_{predictor_type}_' + descr_str
    log_folder = osp.join(save_dir, sub_dir)

    if not osp.exists(log_folder):
        os.makedirs(log_folder)

    return log_folder