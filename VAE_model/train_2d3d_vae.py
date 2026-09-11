"""
Train a single shared VAE on both 2D and 3D velocity fields:

    2D velocity -> Encoder -> latent -> Decoder -> 2D velocity
    3D velocity -> Encoder -> latent -> Decoder -> 3D velocity

After this VAE is trained, the corresponding 2D/3D samples can be
encoded separately to obtain:

    z_2d[i] <-> z_3d[i]

These paired latent representations are then intended for training
the diffusion model.

Dataset format:
    U_all.pt = torch.cat([U_2d, U_3d], dim=0)

    U_2d shape = [100, 11, 3, 256, 256]
    U_3d shape = [100, 11, 3, 256, 256]

Therefore:
    U_all shape = [200, 11, 3, 256, 256]

The first 100 samples are 2D.
The last 100 samples are the corresponding 3D fields.

Internally, Conv3D expects:
    [B, C, D, H, W]

while the stored data is:
    [B, D, C, H, W]

Therefore we permute:
    [B, D, C, H, W] -> [B, C, D, H, W]
"""

import time
import os
import json
import os.path as osp
import argparse
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.vae.autoencoder import VariationalAutoencoder

from utils.metrics import (
    kl_divergence,
    mae_loss_per_channel,
    normalized_mae_loss_per_channel,
    normalized_mse_per_channel,
)


# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(
    sys.stdout, "reconfigure"
) else None


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(description="Train a single shared VAE on 2D and 3D velocity")

    parser.add_argument("--dataset-dir", type=str,
        default="/home/hlcao2/Diffusion_model_project/dataset_3d",
        help="Directory containing U_all.pt and statistics.json"
    )

    parser.add_argument("--data-file", type=str,
        default="x/U_all.pt",
        help="Concatenated velocity tensor"
    )

    parser.add_argument( "--save-dir", type=str,
        default="VAE_model/trained/vae_2d3d_reconstruction",
        help="Directory for saved model"
    )

    parser.add_argument("--in-channels", type=int,
        default=3,
        help="Number of velocity channels: u, v, w"
    )

    parser.add_argument("--latent-channels", type=int,
        default=8,
        help="Number of latent channels"
    )

    parser.add_argument("--batch-size", type=int,
        default=2,
        help="Batch size"
    )

    parser.add_argument("--num-epochs", type=int,
        default=50,
        help="Number of training epochs"
    )

    parser.add_argument("--learning-rate", type=float,
        default=1e-4,
        help="Learning rate"
    )

    parser.add_argument("--device", type=str,
        default=None,
        help="cuda or cpu"
    )

    parser.add_argument("--loss-function", type=str,
        default="mae_per_channel",
        choices=["mae_per_channel", "normalized_mae_per_channel", "normalized_mse_per_channel"],
        help="Reconstruction loss"
    )

    parser.add_argument("--debug-latent",
        action="store_true",
        help="Print latent statistics during first epoch"
    )

    parser.add_argument("--debug-batches", type=int,
        default=5,
        help="Number of batches for latent debugging"
    )

    parser.add_argument("--norm-mode",type=str,
        default="max", choices=["max", "mean"],
        help="Normalization mode"
    )

    parser.add_argument("--split-seed", type=int,
        default=2024,
        help="Random seed for train/val/test split"
    )

    return parser.parse_args()


# ============================================================
# TENSOR HEALTH
# ============================================================

def check_tensor_health(tensor: torch.Tensor, name: str, fail_on_bad: bool = True) -> bool:
    """
    Check tensor for NaN/Inf values and optionally fail with clear message.
    
    Args:
        tensor: Tensor to check
        name: Name for error messages
        fail_on_bad: If True, raise exception on NaN/Inf
        
    Returns:
        True if tensor is healthy, False otherwise
    """
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf:
        msg = f"FATAL: {name} contains {'NaN' if has_nan else ''}{'/' if has_nan and has_inf else ''}{'Inf' if has_inf else ''}"
        msg += f"\n  Shape: {tensor.shape}"
        msg += f"\n  Min: {tensor.min().item():.6e}, Max: {tensor.max().item():.6e}"
        msg += f"\n  NaN count: {torch.isnan(tensor).sum().item()}, Inf count: {torch.isinf(tensor).sum().item()}"
        
        if fail_on_bad:
            raise RuntimeError(msg)
        else:
            print(f"WARNING: {msg}")
            return False
    return True

# ============================================================
# LATENT DEBUGGING
# ============================================================

def log_latent_stats(mu: torch.Tensor, logvar: torch.Tensor, prefix: str = ""):
    """
    Log statistics of mu and logvar for debugging KL divergence issues.
    
    Args:
        mu: Mean tensor from encoder
        logvar: Log-variance tensor from encoder  
        prefix: Prefix for log messages (e.g., "Train" or "Val")
    """
    print(f"  {prefix} Latent Stats:")
    print(f"    mu    - min: {mu.min().item():+.4f}, max: {mu.max().item():+.4f}, "
          f"mean: {mu.mean().item():+.4f}, std: {mu.std().item():.4f}")
    print(f"    logvar - min: {logvar.min().item():+.4f}, max: {logvar.max().item():+.4f}, "
          f"mean: {logvar.mean().item():+.4f}, std: {logvar.std().item():.4f}")
    
    # Check for potential KL explosion indicators
    exp_logvar_max = torch.exp(logvar).max().item()
    if exp_logvar_max > 1e6:
        print(f"    WARNING: exp(logvar) max = {exp_logvar_max:.2e} - potential KL explosion!")
    
    # Compute raw KL components for diagnostics
    kl_mu_term = (mu.pow(2)).mean().item()
    kl_logvar_term = logvar.mean().item()
    kl_exp_term = logvar.exp().mean().item()
    print(f"    KL components: mu²={kl_mu_term:.4f}, logvar={kl_logvar_term:.4f}, exp(logvar)={kl_exp_term:.4f}")



# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    print("\n" + "=" * 70)
    print("TRAINING SINGLE SHARED VAE: 2D + 3D VELOCITY")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = args.device

    if device is None:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print(f"\nUsing device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    data_path = osp.join(args.dataset_dir, args.data_file)
    stats_path = osp.join(args.dataset_dir, "statistics.json")

    os.makedirs(args.save_dir, exist_ok=True)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found:\n{data_path}")

    # --------------------------------------------------------
    # Training stability parameters
    # --------------------------------------------------------

    kl_warmup_epochs = 10
    min_kl_coeff = 1e-5
    max_kl_coeff = 1e-3
    gradient_accumulation_steps = 10

    # --------------------------------------------------------
    # Load concatenated data
    # --------------------------------------------------------

    print("\nLoading concatenated dataset:")
    print(data_path)

    data = torch.load(data_path, map_location="cpu")

    print(
        f"Loaded tensor shape: "
        f"{tuple(data.shape)}"
    )

    if data.ndim != 5:

        raise ValueError(
            f"Expected 5D tensor [N,D,C,H,W], got {data.shape}"
        )

    total_samples = data.shape[0]

    # --------------------------------------------------------
    # Verify expected dimensions
    # --------------------------------------------------------

    expected_shape = (total_samples, 11, 3, 256, 256)

    if tuple(data.shape) != expected_shape:
        print(f"WARNING: Expected something like {expected_shape}, got {tuple(data.shape)}")

    # --------------------------------------------------------
    # Train / validation / test split
    # --------------------------------------------------------

    train_size = int(0.70 * total_samples)
    val_size = int(0.15 * total_samples)
    test_size = (total_samples - train_size - val_size)

    print("\nDataset split:")
    #print(f"  Total pairs: {num_pairs}")
    print(f"  Train: {train_size} (70%)")
    print(f"  Validation: {val_size} (15%)")
    print(f"  Test: {test_size} (15%)")

    generator = torch.Generator().manual_seed(args.split_seed)

    train_dataset, val_dataset, test_dataset = (
        random_split(
            data, #paired_dataset,
            [train_size,val_size,test_size],
            generator=generator
        )
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Normalization statistics
    # --------------------------------------------------------

    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"statistics.json not found: {stats_path}")

    with open(stats_path, "r") as f:
        statistics = json.load(f)

    use_per_component = True

    if use_per_component and "U_per_component" in statistics:

        pc = statistics["U_per_component"]
        pc_2d = statistics.get("U_2d_per_component", {})

        if args.norm_mode == "max":
            norm_u = max(
                pc["max_u"],
                pc_2d.get("max_u", 0)
            )

            norm_v = max(
                pc["max_v"],
                pc_2d.get("max_v", 0)
            )

            norm_w = max(
                pc["max_w"],
                pc_2d.get("max_w", 0)
            )

            stat_key = "max"

        else:
            norm_u = max(
                pc.get("mean_u", pc["max_u"]),
                pc_2d.get("mean_u", pc_2d.get("max_u", 0))
            )

            norm_v = max(
                pc.get("mean_v", pc["max_v"]),
                pc_2d.get("mean_v", pc_2d.get("max_v", 0))
            )

            norm_w = max(
                pc.get(
                    "mean_w",
                    pc["max_w"]
                ),
                pc_2d.get(
                    "mean_w",
                    pc_2d.get("max_w", 0)
                )
            )

            stat_key = "mean"

        norm_factors = torch.tensor([norm_u, norm_v, norm_w], dtype=torch.float32)

        print(f"\n=== Per-Component Normalization ({args.norm_mode.upper()}) ===")
        print(f"  {stat_key}_u: {norm_u:.6f}")
        print(f"  {stat_key}_v: {norm_v:.6f}")
        print(f"  {stat_key}_w: {norm_w:.6f}")
        print("=" * 50)

    else:
        # Fallback to global normalization

        max_U_2d = statistics.get("U_2d", statistics["U"])["max"]
        max_U_3d = statistics["U"]["max"]
        max_velocity = max(max_U_2d, max_U_3d)

        norm_factors = torch.tensor([max_velocity, max_velocity, max_velocity], dtype=torch.float32)

        print("\n=== Global Normalization ===")
        print(f"  max 2D velocity: {max_U_2d:.6f}")
        print(f"  max 3D velocity: {max_U_3d:.6f}")
        print(f"  normalization: {max_velocity:.6f}")
        print("=" * 50)

    # --------------------------------------------------------
    # Create VAE
    # --------------------------------------------------------

    print("\nCreating shared VAE:")
    print("  2D input  -> 2D reconstruction")
    print("  3D input  -> 3D reconstruction")
    print(f"  Input channels: {args.in_channels}")
    print(f"  Latent channels: {args.latent_channels}")

    # One shared VAE for both 2D and 3D
    vae = VariationalAutoencoder(in_channels=args.in_channels, latent_channels=args.latent_channels, conditional=False).to(device)

    # --------------------------------------------------------
    # Multi-GPU
    # --------------------------------------------------------

    # if device == "cuda" and torch.cuda.device_count() > 1:

    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     vae = torch.nn.DataParallel(vae)

    print(f"Model loaded on {device}")

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.Adam(
        vae.parameters(),
        lr=args.learning_rate
    )

    # --------------------------------------------------------
    # Loss function
    # --------------------------------------------------------

    loss_functions = {
        "mae_per_channel": mae_loss_per_channel,
        "normalized_mae_per_channel": normalized_mae_loss_per_channel,
        "normalized_mse_per_channel": normalized_mse_per_channel
    }

    reconstruction_loss_fn = loss_functions[args.loss_function]

    print(f"Using reconstruction loss: {args.loss_function}")

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    log_dict = {

        "loss": {
            "recons_train": [], "recons_val": [], 'recons_test': [],
            "kl_train": [], "kl_val": [], "kl_coeff": [], 'kl_test': []
        },

        "recons_test": None,
        "kl_test": None,

        "in_channels": args.in_channels,
        "latent_channels": args.latent_channels,
        "norm_factors": norm_factors.tolist(),
        "norm_mode": args.norm_mode,
        "loss_function": args.loss_function,
        "input": "2D and 3D velocity",
        "target": "same as input",
        "dataset_shape": list(data.shape),
        # "num_pairs": num_pairs
    }

    best_val_loss = float("inf")

    # ========================================================
    # TRAINING
    # ========================================================

    print("\n" + "=" * 70)
    print("START TRAINING")
    print("=" * 70)

    for epoch in range(args.num_epochs):

        start_time = time.time()

        # ----------------------------------------------------
        # KL annealing
        # ----------------------------------------------------

        if epoch < kl_warmup_epochs:
            kl_coeff = (min_kl_coeff + (max_kl_coeff - min_kl_coeff) * (epoch / kl_warmup_epochs))

        else:
            kl_coeff = max_kl_coeff

        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print(f"KL coefficient: {kl_coeff:.6f}")

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        vae.train()

        running_recons = 0.0
        running_kl = 0.0

        optimizer.zero_grad()

        num_batches = len(train_loader)

        for i, velocity in enumerate(train_loader):
            print(f"Training batch {i + 1}/{num_batches}")

            # ------------------------------------------------
            # Move to device
            # ------------------------------------------------

            velocity = velocity.to(device, non_blocking=True)

            # ------------------------------------------------
            # Convert stored format:
            #
            # [B, D, C, H, W]
            #
            # to:
            #
            # [B, C, D, H, W]
            # ------------------------------------------------

            velocity = velocity.permute(0, 2, 1, 3, 4)

            # ------------------------------------------------
            # Normalize
            # ------------------------------------------------

            nf = (
                norm_factors
                .to(device)
                .view(1, 3, 1, 1, 1)
            )

            velocity = velocity / nf

            # ------------------------------------------------
            # Health checks
            # ------------------------------------------------

            check_tensor_health(
                velocity,
                f"Input batch {i}"
            )

            # =================================================
            # [2D, 3D] -> VAE -> [2D, 3D] (Train)
            # =================================================

            # Use the actual VAE encoder so that logvar can be
            # clamped BEFORE sampling.
            mean, logvar = vae.encoder(velocity)
            logvar = torch.clamp(logvar, min=-10.0, max=10.0)
            z = vae.encoder.sample(mean, logvar)
            preds = vae.decoder(z)

            if not check_tensor_health(mean, f"2D mu batch {i}", fail_on_bad=False):
                print("Skipping reconstruction due to bad mu")
                continue

            if not check_tensor_health(logvar, f"2D logvar batch {i}", fail_on_bad=False):
                print("Skipping reconstruction due to bad logvar")
                continue

            if args.debug_latent and epoch == 0 and i < args.debug_batches:
                log_latent_stats(mean, logvar, prefix=f"2D Train batch {i}")

            targets = velocity
            reconstruction_loss = reconstruction_loss_fn(preds, targets)
            kl_loss = kl_divergence(mu=mean, logvar=logvar) 

            # ------------------------------------------------
            # Total VAE loss
            # ------------------------------------------------

            loss = reconstruction_loss + kl_coeff * kl_loss
            loss = (loss / gradient_accumulation_steps)

            # ------------------------------------------------
            # Backprop
            # ------------------------------------------------

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                vae.parameters(),
                max_norm=1.0
            )

            # ------------------------------------------------
            # Optimizer step
            # ------------------------------------------------

            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # ------------------------------------------------
            # Logging
            # ------------------------------------------------

            running_recons += (reconstruction_loss.item())
            running_kl += (kl_loss.item())

            if i % 10 == 0:
                print(f"  Mean Recons: {reconstruction_loss.item():.6f}")
                print(f"  Mean KL: {kl_loss.item():.6f}")

            # ------------------------------------------------
            # KL explosion protection
            # ------------------------------------------------

            if kl_loss.item() > 1000:

                raise RuntimeError(f"KL loss exploded to {kl_loss.item():.6f}")

        # ----------------------------------------------------
        # Apply remaining gradients
        # ----------------------------------------------------

        if (num_batches % gradient_accumulation_steps != 0):
            optimizer.step()
            optimizer.zero_grad()

        # ----------------------------------------------------
        # Training averages
        # ----------------------------------------------------

        avg_recons_train = running_recons / num_batches

        avg_kl_train = running_kl/ num_batches

        log_dict["loss"]["recons_train"].append(avg_recons_train)
        log_dict["loss"]["kl_train"].append(avg_kl_train)
        log_dict["loss"]["kl_coeff"].append(kl_coeff)

        # ====================================================
        # VALIDATION
        # ====================================================

        print("\nStarting validation...")

        vae.eval()

        running_recons = 0.0
        running_kl = 0.0

        with torch.no_grad():

            for j, velocity in enumerate(val_loader):
                velocity = velocity.to(device, non_blocking=True)

                # [B,D,C,H,W] -> [B,C,D,H,W]

                velocity = velocity.permute(0, 2, 1, 3, 4)

                nf = (
                    norm_factors
                    .to(device)
                    .view(1, 3, 1, 1, 1)
                )

                velocity = velocity / nf

                # =================================================
                # [2D, 3D] -> VAE -> [2D, 3D] (Validate)
                # =================================================

                mean, logvar = vae.encoder(velocity)
                logvar = torch.clamp(logvar, min=-10.0, max=10.0)

                z = vae.encoder.sample(mean, logvar)
                preds = vae.decoder(z)

                reconstruction_loss = reconstruction_loss_fn(preds, velocity) 
                kl_loss = kl_divergence(mu=mean, logvar=logvar) 

                running_recons += reconstruction_loss.item()
                running_kl += kl_loss.item()

                print(
                    f"Val batch {j}: "
                    f"Mean KL="
                    f"{kl_loss.item():.6f}"
                )

        avg_recons_val = running_recons / len(val_loader)
        avg_kl_val = running_kl/ len(val_loader)

        log_dict["loss"]["recons_val"].append(avg_recons_val)
        log_dict["loss"]["kl_val"].append(avg_kl_val)

        # ----------------------------------------------------
        # Epoch summary
        # ----------------------------------------------------

        epoch_time = time.time() - start_time

        print(
            "\n"
            + "=" * 70
        )

        print(f"Epoch {epoch + 1}/{args.num_epochs}")

        print(
            f"Train: "
            f"Recons="
            f"{avg_recons_train:.6f}, "
            f"KL="
            f"{avg_kl_train:.6f}"
        )

        print(
            f"Val:   "
            f"Recons="
            f"{avg_recons_val:.6f}, "
            f"KL="
            f"{avg_kl_val:.6f}"
        )

        print(f"KL coefficient: {kl_coeff:.6f}")
        print(f"Time: {epoch_time:.2f} s")
        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Save latest model
        # ----------------------------------------------------

        model_path = osp.join(args.save_dir, "vae.pt")

        torch.save(vae.state_dict(), model_path)
        log_path = osp.join(args.save_dir, "vae_log.json")

        with open(log_path, "w") as f:
            json.dump(log_dict, f, indent=2)

        print(f"Model saved to: {model_path}")

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        current_val_loss = avg_recons_val + kl_coeff * avg_kl_val

        if current_val_loss < best_val_loss:

            best_val_loss = current_val_loss
            best_model_path = osp.join(args.save_dir, "best_model.pt")

            torch.save(
                vae.state_dict(),
                best_model_path
            )

            print("✓ New best model!")
            print(f"  Val loss: {current_val_loss:.6f}")
            print(f"  Saved to: {best_model_path}")

    # ========================================================
    # FINAL TEST
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    vae.eval()

    running_recons = 0.0
    running_kl = 0.0

    with torch.no_grad():
        vae.load_state_dict(torch.load(osp.join(args.save_dir, "best_model.pt"), map_location=device))
        
        for k, velocity in enumerate(test_loader):
            velocity = velocity.to(device)

            # [B,D,C,H,W] -> [B,C,D,H,W]

            velocity = velocity.permute(0, 2, 1, 3, 4)

            nf = (norm_factors.to(device).view(1, 3, 1, 1, 1))
            velocity = velocity / nf

            # =================================================
            # [2D, 3D] -> VAE -> [2D, 3D] (Test)
            # =================================================

            mean, logvar = vae.encoder(velocity)

            logvar = torch.clamp(logvar, min=-10.0, max=10.0)
            z = vae.encoder.sample(mean, logvar)
            preds = vae.decoder(z)

            reconstruction_loss = reconstruction_loss_fn(preds, velocity) 
            kl_loss = kl_divergence(mu=mean, logvar=logvar)   

            running_recons += (reconstruction_loss.item())
            running_kl += kl_loss.item()

            print(
                f"Test batch {k}: "
                f"Mean Recons="
                f"{reconstruction_loss.item():.6f}, "
                f"Mean KL="
                f"{kl_loss.item():.6f}"
            )

    avg_recons_test = running_recons / len(test_loader)
    avg_kl_test = running_kl/ len(test_loader)

    log_dict['loss']["recons_test"] = (avg_recons_test)
    log_dict['loss']["kl_test"] = (avg_kl_test)

    print("\nFinal Test Results:")
    print(f"  Reconstruction: {avg_recons_test:.6f}")
    print(f"  KL: {avg_kl_test:.6f}")

    # --------------------------------------------------------
    # Final save
    # --------------------------------------------------------

    model_path = osp.join(args.save_dir, "vae.pt")
    torch.save(vae.state_dict(), model_path) 

    log_path = osp.join(args.save_dir, "vae_log.json")
    with open(log_path, "w") as f:
        json.dump(log_dict, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {args.save_dir}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        print("\n" + "=" * 70)
        print("FATAL ERROR")
        print("=" * 70)

        print(e)

        import traceback

        traceback.print_exc()

        sys.exit(1)