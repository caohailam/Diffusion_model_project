"""
Training script for latent diffusion model.

This script trains a U-Net to denoise latents for 3D velocity prediction from 2D inputs.

Prerequisites:
    - Shared VAE trained on both 2D and 3D velocity fields
      (VAE_model/train_2d3d_vae.py)

Usage:
    python Diffusion_model/train.py \
        --root-dir "path/to/dataset_3d" \
        --vae-path "VAE-model/trained/vae_common" \
        --in-channels 17 --out-channels 8 \
        --features 64 128 256 512 1024 \
        --batch-size 3 --num-epochs 100
"""

import time
import json
import os.path as osp

import torch
import torch.optim as optim
import optuna
from optuna.trial import Trial
from sqlalchemy import create_engine

# Local
from utils.dataset import get_loader
from src.helper import set_model, run_epoch
from src.unet.metrics import cost_function
from src.predictor import LatentDiffusionPredictor

from config import parser, process_args, make_log_folder


args = parser.parse_args()


def train(
    train_loader,
    val_loader,
    test_loader=None,
    trial: Trial = None
):
    
    param_dict = process_args(args)
    log_dict = {
        'params': param_dict,
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'time': [],
        'learning_rate_history': [],
        # Physics metrics logging
        'physics_metrics': {
            'div_mean': [],
            'div_std': [],
            'flow_rate_cv': [],
            'vel_in_solid': [],
            'vel_mean_fluid': [],
            'gradient_smooth': [],
            'laplacian_smooth': [],
            'vel_u_mean': [],
            'vel_v_mean': [],
            'vel_w_mean': [],
            'vel_u_max': [],
            'vel_v_max': [],
            'vel_w_max': [],
            'loss_divergence': [],
            'loss_flow_rate': [],
            'loss_smoothness': [],
            'loss_laplacian': []
        }
    }
    log_folder = make_log_folder(param_dict)

    root_dir = param_dict['dataset']['root_dir']
    train_dict = param_dict['training']

    device = train_dict['device']
    learning_rate = train_dict['learning_rate']
    weight_decay = train_dict['weight_decay']
    scheduler_kwargs = train_dict['scheduler']
    num_epochs = train_dict['num_epochs']
    cost_function_name = train_dict['cost_function']
    
    # Physics loss parameters
    lambda_div = train_dict.get('lambda_div', 0.0)
    lambda_flow = train_dict.get('lambda_flow', 0.0)
    lambda_smooth = train_dict.get('lambda_smooth', 0.0)
    lambda_laplacian = train_dict.get('lambda_laplacian', 0.0)
    physics_loss_freq = train_dict.get('physics_loss_freq', 1)
    
    # Component weighting parameters
    lambda_velocity = train_dict.get('lambda_velocity', 0.0)
    weight_u = train_dict.get('weight_u', 1.0)
    weight_v = train_dict.get('weight_v', 1.0)
    weight_w = train_dict.get('weight_w', 1.0)
    velocity_loss_primary = train_dict.get('velocity_loss_primary', False)
    
    predictor_type = train_dict['predictor_type']
    predictor_kwargs = train_dict['predictor']
    
    # Print physics configuration
    physics_enabled = any([lambda_div > 0, lambda_flow > 0, lambda_smooth > 0, lambda_laplacian > 0])
    velocity_loss_enabled = lambda_velocity > 0 or velocity_loss_primary
    if physics_enabled:
        print("\n=== Physics-Informed Training Enabled ===")
        print(f"  lambda_div (mass conservation): {lambda_div}")
        print(f"  lambda_flow (flow-rate consistency): {lambda_flow}")
        print(f"  lambda_smooth (gradient smoothness): {lambda_smooth}")
        print(f"  lambda_laplacian (Laplacian smoothness): {lambda_laplacian}")
        print(f"  physics_loss_freq: every {physics_loss_freq} batches")
        print("==========================================\n")
    
    if velocity_loss_enabled:
        print("\n=== Component-Weighted Velocity Loss Enabled ===")
        if velocity_loss_primary:
            print(f"  MODE: PRIMARY LOSS (replaces noise prediction loss)")
        else:
            print(f"  MODE: AUXILIARY (lambda_velocity={lambda_velocity})")
        print(f"  weight_u (vx): {weight_u}")
        print(f"  weight_v (vy): {weight_v}")
        print(f"  weight_w (vz): {weight_w}")
        print("================================================\n")

    # Model
    predictor = set_model(
        type=predictor_type,
        kwargs=predictor_kwargs,
        norm_file=osp.join(root_dir, 'statistics.json')
    )
    predictor.to(device)

    optimizer = optim.Adam(
        predictor.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = None
    if scheduler_kwargs['flag']:
        gamma = scheduler_kwargs['gamma']
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    criterion = cost_function(cost_function_name)

    best_loss = float('inf')
    for epoch in range(num_epochs):

        current_lr = optimizer.param_groups[0]['lr']

        # run epoch with physics-informed training
        start_time = time.time()
        avg_train_loss, avg_val_loss, physics_metrics = run_epoch(
            loaders=(train_loader, val_loader),
            predictor=predictor,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            lambda_div=lambda_div,
            lambda_flow=lambda_flow,
            lambda_smooth=lambda_smooth,
            lambda_laplacian=lambda_laplacian,
            physics_loss_freq=physics_loss_freq,
            lambda_velocity=lambda_velocity,
            weight_u=weight_u,
            weight_v=weight_v,
            weight_w=weight_w,
            velocity_loss_primary=velocity_loss_primary
        )
        dtime = time.time() - start_time

        # log standard metrics
        log_dict['epoch'].append(epoch)
        log_dict['time'].append(dtime)
        log_dict['train_loss'].append(avg_train_loss)
        log_dict['val_loss'].append(avg_val_loss)
        log_dict['learning_rate_history'].append(current_lr)
        
        # log physics metrics
        for key in log_dict['physics_metrics']:
            if key in physics_metrics:
                log_dict['physics_metrics'][key].append(physics_metrics[key])
            elif key.replace('loss_', '') in physics_metrics:
                log_dict['physics_metrics'][key].append(physics_metrics[key.replace('loss_', '')])
            else:
                log_dict['physics_metrics'][key].append(0.0)

        # Save
        model_path = osp.join(log_folder, 'model.pt')
        best_model_path = osp.join(log_folder, 'best_model.pt')
        log_path = osp.join(log_folder, 'log.json')

        torch.save(predictor.state_dict(), model_path)
        if avg_val_loss < best_loss:
            torch.save(predictor.state_dict(), best_model_path)
            best_loss = avg_val_loss

        with open(log_path, 'w') as f:
            json.dump(log_dict, f, indent=4)

        # Print progress with physics metrics
        print(f"Epoch {epoch}: train_loss={avg_train_loss:.6f} | val_loss={avg_val_loss:.6f} | time={dtime:.2f} s")
        if physics_enabled and physics_metrics:
            print(f"  Physics: div_mean={physics_metrics.get('div_mean', 0):.6f} | "
                  f"flow_cv={physics_metrics.get('flow_rate_cv', 0):.6f}")
            print(f"  Smooth:  grad={physics_metrics.get('gradient_smooth', 0):.2e} | "
                  f"lapl={physics_metrics.get('laplacian_smooth', 0):.2e}")
            # Print velocity component stats for debugging
            print(f"  Velocity: u_mean={physics_metrics.get('vel_u_mean', 0):.2e} | "
                  f"v_mean={physics_metrics.get('vel_v_mean', 0):.2e} | "
                  f"w_mean={physics_metrics.get('vel_w_mean', 0):.2e}")
            print(f"           u_max={physics_metrics.get('vel_u_max', 0):.2e} | "
                  f"v_max={physics_metrics.get('vel_v_max', 0):.2e} | "
                  f"w_max={physics_metrics.get('vel_w_max', 0):.2e}")
        
        if scheduler is not None: scheduler.step()


        if trial is not None:
            trial.report(avg_val_loss, epoch)

            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    
    # Evaluate on test set after training completes
    if test_loader is not None:
        print("\nEvaluating on test set...")
        predictor.eval()
        
        # Load best model for test evaluation
        predictor.load_state_dict(torch.load(best_model_path, map_location=device))
        
        from src.helper import select_input_output
        
        option = 'latent-diffusion'

        test_loss = 0
        num_test_batch = len(test_loader)
        with torch.no_grad():
            for k, data in enumerate(test_loader):
                print(f"Test set: batch [{k+1}/{num_test_batch}]")
                
                input, targets = select_input_output(data, option, device)
                
                if option == 'latent-diffusion':
                    img = input[0]
                    velocity_2d = input[1]
                    target_latents = predictor.encode_target(targets, velocity_2d)
                    preds, target_noise = predictor(img, velocity_2d, x_start=target_latents)
                    loss = criterion(output=preds, target=target_noise)
                else:
                    preds = predictor(*input)
                    targets = predictor.normalizer['output'](targets)
                    loss = criterion(output=preds, target=targets)
                
                test_loss += loss.item()
        
        avg_test_loss = test_loss / num_test_batch
        log_dict['test_loss'] = avg_test_loss
        
        # Save updated log with test loss
        with open(log_path, 'w') as f:
            json.dump(log_dict, f, indent=4)
        
        print(f"\nTest Loss: {avg_test_loss}")
            
    return avg_train_loss, avg_val_loss


# def objective(trial: Trial):
#     """Objective function for hyper-parameter tuning."""

#     # sample hyper-parameters
#     args.batch_size = trial.suggest_int(
#         "batch_size",
#         args.range_batch_size[0],
#         args.range_batch_size[1]
#     )
#     args.kernel_size = trial.suggest_int(
#         "kernel_size",
#         args.range_kernel_size[0],
#         args.range_kernel_size[1],
#         step=2
#     )
#     levels = trial.suggest_int(
#         "levels",
#         args.range_level[0],
#         args.range_level[1]
#     )
#     factors = [2**val for val in range(levels)]
#     if args.top_bottom:
#         args.features = [args.top_feature_channels * val for val in factors]
#     else:
#         args.features = [int(args.bottom_feature_channels / val) for val in reversed(factors)]

#     args.learning_rate = trial.suggest_float(
#         "learning_rate",
#         args.range_learning_rate[0],
#         args.range_learning_rate[1],
#         log=True
#     )
    
#     # load data
#     train_loader, val_loader, test_loader = get_loader(
#         root_dir=args.root_dir,
#         batch_size=args.batch_size,
#         shuffle=args.shuffle,
#         augment=args.augment,
#         k_folds=None,
#         num_workers=0,
#         use_3d=args.use_3d
#     )[0]

#     # train
#     _, val_loss = train(train_loader, val_loader, test_loader, trial)

#     return val_loss


if __name__=='__main__':

    if args.mode == 'train':

        # load data
        train_loader, val_loader, test_loader = get_loader(
            root_dir=args.root_dir,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            augment=args.augment,
            k_folds=None,
            num_workers=0,
            use_3d=args.use_3d
        )[0]

        # train
        train(train_loader, val_loader, test_loader)

    elif args.mode == 'CV':
        # Cross-Validation

        # load data
        data_folds = get_loader(
            root_dir=args.root_dir,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            augment=args.augment,
            k_folds=args.k_folds,
            num_workers=0,
            use_3d=args.use_3d
        )

        # train
        for i, (train_loader, val_loader, test_loader) in enumerate(data_folds):
            print(f'Cross-Validation [{i+1}/{args.k_folds}]')

            args.name = f'kfold-{i+1}.{args.k_folds}'

            train(train_loader, val_loader, test_loader)


    # elif args.mode == 'optimize':

    #     # Create SQL engine
    #     db_path = osp.abspath(
    #         osp.join(args.save_dir, f'study.db')
    #     )
    #     url = f"sqlite:////{db_path}"
    #     engine = create_engine(url)

    #     # Set up study
    #     study = optuna.create_study(
    #         direction='minimize',
    #         study_name=args.name,
    #         storage=url
    #     )
    #     study.optimize(objective, n_trials=args.n_trials)

    #     pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    #     complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    #     print("Study statistics:")
    #     print("\t Number of finished trials: ", len(study.trials))
    #     print("\t Number of pruned trials: ", len(pruned_trials))
    #     print("\t Number of complete trials: ", len(complete_trials))

    #     print("Best trial:")
    #     trial = study.best_trial
    #     print("\t Value: ", trial.value)

    #     print("\t Params:")
    #     for key, value in trial.params.items():
    #         print(f"\t {key}: {value}")
