"""
VAE Inference Script: Check the current shared VAE.

The current VAE is a single standard VariationalAutoencoder trained on both
2D and 3D velocity fields:

    velocity -> shared Encoder -> latent -> shared Decoder -> reconstruction

Dataset format:
    U_all.pt = torch.cat([U_2d, U_3d], dim=0)
    shape = [200, 11, 3, 256, 256]

    first 100 samples: 2D velocity fields
    last 100 samples: 3D velocity fields

The same 70/15/15 split and split seed used during VAE training are reproduced
so that --index refers to a sample in the VAE test split.

Usage:
    python VAE_model/inference_vae.py --vae-path [path] [options]

By default, best_model.pt is loaded.
"""

import argparse
import json
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import random_split


# -----------------------------------------------------------------------------
# Paths / imports
# -----------------------------------------------------------------------------
current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.abspath(osp.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.vae.autoencoder import VariationalAutoencoder


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def check_tensor_health(tensor: torch.Tensor, name: str) -> None:
    """Raise an error if a tensor contains NaN or Inf values."""
    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"{name} contains NaN/Inf values. shape={tuple(tensor.shape)}"
        )


def compute_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    """Compute reconstruction metrics in physical velocity units."""
    error = torch.abs(original - reconstructed)

    mae = error.mean()
    mae_u = error[:, 0].mean()
    mae_v = error[:, 1].mean()
    mae_w = error[:, 2].mean()

    relative_error = mae / (torch.abs(original).mean() + 1e-8)

    return {
        "mae": mae.item(),
        "mae_u": mae_u.item(),
        "mae_v": mae_v.item(),
        "mae_w": mae_w.item(),
        "relative_error": relative_error.item(),
    }


def visualize_reconstruction_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    depth_slice: int = 0,
    figsize: tuple = (15, 12),
):
    """
    Compare original and reconstructed velocity at one depth slice.

    Args:
        original: (D, 3, H, W)
        reconstructed: (D, 3, H, W)
    """
    orig_slice = original[depth_slice]
    recon_slice = reconstructed[depth_slice]
    error = np.abs(orig_slice - recon_slice)

    component_names = ["u (vx)", "v (vy)", "w (vz)"]

    fig, axes = plt.subplots(3, 3, figsize=figsize)

    for i, name in enumerate(component_names):
        orig_data = orig_slice[i]
        recon_data = recon_slice[i]

        vabs = max(
            np.nanmax(np.abs(orig_data)),
            np.nanmax(np.abs(recon_data)),
            1e-12,
        )

        # Original
        im = axes[0, i].imshow(
            orig_data,
            cmap="coolwarm",
            origin="lower",
            vmin=-vabs,
            vmax=vabs,
        )
        axes[0, i].set_title(f"Original {name}")
        axes[0, i].axis("off")
        plt.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)

        # Reconstructed
        im = axes[1, i].imshow(
            recon_data,
            cmap="coolwarm",
            origin="lower",
            vmin=-vabs,
            vmax=vabs,
        )
        axes[1, i].set_title(f"Reconstructed {name}")
        axes[1, i].axis("off")
        plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

        # Absolute error
        im = axes[2, i].imshow(
            error[i],
            cmap="Reds",
            origin="lower",
        )
        axes[2, i].set_title(f"Error |Δ{name}|")
        axes[2, i].axis("off")
        plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Shared VAE Reconstruction (depth slice {depth_slice})"
    )
    plt.tight_layout()
    return fig


def visualize_w_across_depth(
    original: np.ndarray,
    reconstructed: np.ndarray,
):
    """Visualize input/reconstructed w over several depth slices."""
    n_depths = min(original.shape[0], 6)
    depth_indices = np.linspace(
        0, original.shape[0] - 1, n_depths, dtype=int
    )

    all_orig_w = original[:, 2]
    all_recon_w = reconstructed[:, 2]
    w_vabs = max(
        np.nanmax(np.abs(all_orig_w)),
        np.nanmax(np.abs(all_recon_w)),
        1e-12,
    )

    fig, axes = plt.subplots(
        2, n_depths,
        figsize=(3 * n_depths, 6),
        squeeze=False,
    )

    for i, depth in enumerate(depth_indices):
        axes[0, i].imshow(
            original[depth, 2],
            cmap="coolwarm",
            origin="lower",
            vmin=-w_vabs,
            vmax=w_vabs,
        )
        axes[0, i].set_title(f"Input w (d={depth})")
        axes[0, i].axis("off")

        im = axes[1, i].imshow(
            reconstructed[depth, 2],
            cmap="coolwarm",
            origin="lower",
            vmin=-w_vabs,
            vmax=w_vabs,
        )
        axes[1, i].set_title(f"Recon w (d={depth})")
        axes[1, i].axis("off")
        plt.colorbar(im, ax=axes[1, i], fraction=0.046)

    plt.suptitle("VAE: w Component Across Depth Slices")
    plt.tight_layout()
    return fig


def visualize_latent_space(
    latent: np.ndarray,
    depth_slice: int = 0,
):
    """Visualize up to eight latent channels at one depth slice."""
    latent_channels = latent.shape[0]
    n_show = min(latent_channels, 8)
    n_cols = min(4, n_show)
    n_rows = (n_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 4 * n_rows),
        squeeze=False,
    )

    for i in range(n_show):
        row = i // n_cols
        col = i % n_cols

        im = axes[row, col].imshow(
            latent[i, depth_slice],
            cmap="viridis",
            origin="lower",
        )
        axes[row, col].set_title(f"Latent Ch {i}")
        axes[row, col].axis("off")
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

    for i in range(n_show, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis("off")

    plt.suptitle(f"VAE Latent Representation (depth slice {depth_slice})")
    plt.tight_layout()
    return fig


def load_shared_vae(
    vae_path: str,
    device: str,
    checkpoint_name: str,
):
    """Load the current standard shared VAE and its training log."""
    log_path = osp.join(vae_path, "vae_log.json")
    checkpoint_path = osp.join(vae_path, checkpoint_name)

    if not osp.exists(log_path):
        raise FileNotFoundError(f"vae_log.json not found: {log_path}")
    if not osp.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(log_path, "r") as f:
        log = json.load(f)

    in_channels = log.get("in_channels", 3)
    latent_channels = log.get("latent_channels", 8)
    norm_factors = log.get("norm_factors", [1.0, 1.0, 1.0])

    print(f"Loading shared VAE from: {vae_path}")
    print(f"  Checkpoint: {checkpoint_name}")
    print(f"  in_channels: {in_channels}")
    print(f"  latent_channels: {latent_channels}")
    print(f"  conditional: False")
    print(f"  norm_factors: {norm_factors}")

    vae = VariationalAutoencoder(
        in_channels=in_channels,
        latent_channels=latent_channels,
        conditional=False,
    ).to(device)

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    vae.load_state_dict(state_dict, strict=True)
    vae.eval()

    return vae, log, torch.tensor(
        norm_factors,
        dtype=torch.float32,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Check reconstruction quality of the current shared VAE"
    )

    parser.add_argument(
        "--vae-path",
        type=str,
        default="VAE_model/trained/vae_2d3d_reconstruction",
        help="Directory containing vae_log.json and VAE checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pt",
        choices=["best_model.pt", "vae.pt"],
        help="Checkpoint to evaluate (default: best_model.pt)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/home/hlcao2/Diffusion_model_project/dataset_3d",
        help="Directory containing U_all.pt",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="x/U_all.pt",
        help="Concatenated VAE dataset file",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index within the VAE test split to visualize",
    )
    parser.add_argument(
        "--depth-slice",
        type=int,
        default=5,
        help="Depth slice to visualize (default: 5)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=2024,
        help="Seed used for the VAE train/val/test split",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default='VAE_model/inference_figures',
        help="Directory for output figures",
    )

    args = parser.parse_args()
    device = args.device

    print("\n" + "=" * 70)
    print("VAE CHECK")
    print("=" * 70)
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # Resolve VAE path
    # -------------------------------------------------------------------------
    vae_path = args.vae_path
    if not osp.isabs(vae_path):
        project_relative = osp.join(project_root, vae_path)
        script_relative = osp.join(current_dir, vae_path)

        if osp.exists(project_relative):
            vae_path = project_relative
        elif osp.exists(script_relative):
            vae_path = script_relative

    if not osp.exists(vae_path):
        raise FileNotFoundError(f"VAE path not found: {vae_path}")

    # -------------------------------------------------------------------------
    # Load VAE
    # -------------------------------------------------------------------------
    vae, log, norm_factors = load_shared_vae(
        vae_path=vae_path,
        device=device,
        checkpoint_name=args.checkpoint,
    )

    # -------------------------------------------------------------------------
    # Load U_all and reproduce the VAE test split
    # -------------------------------------------------------------------------
    data_path = osp.join(args.dataset_dir, args.data_file)

    if not osp.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    print(f"\nLoading dataset: {data_path}")
    data = torch.load(
        data_path,
        map_location="cpu",
        weights_only=True,
    )

    if data.ndim != 5:
        raise ValueError(
            f"Expected U_all shape [N,D,C,H,W], got {tuple(data.shape)}"
        )

    total_samples = data.shape[0]
    train_size = int(0.70 * total_samples)
    val_size = int(0.15 * total_samples)
    test_size = total_samples - train_size - val_size

    generator = torch.Generator().manual_seed(args.split_seed)
    _, _, test_dataset = random_split(
        data,
        [train_size, val_size, test_size],
        generator=generator,
    )

    if args.index < 0 or args.index >= len(test_dataset):
        raise IndexError(
            f"Test index {args.index} out of range. "
            f"Valid range: 0..{len(test_dataset) - 1}"
        )

    actual_idx = test_dataset.indices[args.index]
    velocity = test_dataset[args.index].unsqueeze(0).to(device)

    # First 100 samples are 2D, last 100 are 3D.
    sample_type = "2D" if actual_idx < 100 else "3D"

    print("\nDataset split:")
    print(f"  Total: {total_samples}")
    print(f"  Train: {train_size}")
    print(f"  Validation: {val_size}")
    print(f"  Test: {test_size}")
    print("\nSelected sample:")
    print(f"  Test-set index: {args.index}")
    print(f"  Original U_all index: {actual_idx}")
    print(f"  Sample type: {sample_type}")

    # Stored format: [B,D,C,H,W]
    # VAE format:     [B,C,D,H,W]
    velocity = velocity.permute(0, 2, 1, 3, 4).contiguous()
    check_tensor_health(velocity, "Input velocity")

    # -------------------------------------------------------------------------
    # Normalize exactly as during VAE training
    # -------------------------------------------------------------------------
    nf = norm_factors.view(1, 3, 1, 1, 1)
    velocity_normalized = velocity / nf
    check_tensor_health(velocity_normalized, "Normalized velocity")

    print(f"\nInput shape: {tuple(velocity.shape)}")
    print(
        f"Physical velocity range: "
        f"[{velocity.min().item():.6e}, {velocity.max().item():.6e}]"
    )

    w_component = velocity[:, 2]
    print(
        f"Input w: "
        f"max_abs={w_component.abs().max().item():.6e}, "
        f"mean_abs={w_component.abs().mean().item():.6e}"
    )

    # -------------------------------------------------------------------------
    # VAE encode / decode
    # Match the previous VAE evaluation behavior: sample z from (mu, logvar).
    # -------------------------------------------------------------------------
    print("\n--- VAE Encode / Decode ---")

    with torch.no_grad():
        mean, logvar = vae.encoder(velocity_normalized)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        latent = vae.encoder.sample(mean, logvar)
        reconstructed_normalized = vae.decoder(latent)

    check_tensor_health(mean, "Latent mean")
    check_tensor_health(logvar, "Latent logvar")
    check_tensor_health(latent, "Latent")
    check_tensor_health(reconstructed_normalized, "Normalized reconstruction")

    # Back to physical velocity units.
    reconstructed = reconstructed_normalized * nf
    check_tensor_health(reconstructed, "Reconstruction")

    print(f"Latent shape: {tuple(latent.shape)}")
    print(
        f"Latent mean range: "
        f"[{mean.min().item():.4f}, {mean.max().item():.4f}]"
    )
    print(f"Latent mean std: {mean.std().item():.4f}")
    print(
        f"Latent logvar range: "
        f"[{logvar.min().item():.4f}, {logvar.max().item():.4f}]"
    )
    print(f"Sampled latent std: {latent.std().item():.4f}")

    # -------------------------------------------------------------------------
    # Reconstruction metrics
    # -------------------------------------------------------------------------
    metrics = compute_metrics(velocity, reconstructed)

    print("\n--- Reconstruction Metrics ---")
    print(f"MAE overall:    {metrics['mae']:.6e}")
    print(f"MAE u (vx):     {metrics['mae_u']:.6e}")
    print(f"MAE v (vy):     {metrics['mae_v']:.6e}")
    print(f"MAE w (vz):     {metrics['mae_w']:.6e}")
    print(f"Relative error: {metrics['relative_error'] * 100:.2f}%")

    w_recon = reconstructed[:, 2]
    print(
        f"Reconstructed w: "
        f"max_abs={w_recon.abs().max().item():.6e}, "
        f"mean_abs={w_recon.abs().mean().item():.6e}"
    )

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    velocity_np = velocity[0].cpu().numpy()             # (3,D,H,W)
    reconstructed_np = reconstructed[0].cpu().numpy()   # (3,D,H,W)
    latent_np = latent[0].cpu().numpy()                 # (C,D',H',W')

    depth = velocity_np.shape[1]
    depth_slice = max(0, min(args.depth_slice, depth - 1))
    latent_depth_slice = min(depth_slice, latent_np.shape[1] - 1)

    fig_comparison = visualize_reconstruction_comparison(
        velocity_np.transpose(1, 0, 2, 3),
        reconstructed_np.transpose(1, 0, 2, 3),
        depth_slice=depth_slice,
    )

    fig_w = visualize_w_across_depth(
        velocity_np.transpose(1, 0, 2, 3),
        reconstructed_np.transpose(1, 0, 2, 3),
    )

    fig_latent = visualize_latent_space(
        latent_np,
        depth_slice=latent_depth_slice,
    )

    # -------------------------------------------------------------------------
    # Save / display
    # -------------------------------------------------------------------------
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

        prefix = (
            f"sample{args.index}_orig{actual_idx}_"
            f"{sample_type.lower()}_{args.checkpoint.replace('.pt', '')}"
        )

        comparison_path = osp.join(
            args.save_dir,
            f"reconstruction_{prefix}.png",
        )
        w_path = osp.join(
            args.save_dir,
            f"w_depth_{prefix}.png",
        )
        latent_path = osp.join(
            args.save_dir,
            f"latent_{prefix}.png",
        )

        fig_comparison.savefig(
            comparison_path,
            dpi=150,
            bbox_inches="tight",
        )
        fig_w.savefig(
            w_path,
            dpi=150,
            bbox_inches="tight",
        )
        fig_latent.savefig(
            latent_path,
            dpi=150,
            bbox_inches="tight",
        )

        print("\nFigures saved to:")
        print(f"  {comparison_path}")
        print(f"  {w_path}")
        print(f"  {latent_path}")
    else:
        plt.show()

    plt.close("all")
    print("\n--- Done ---")


if __name__ == "__main__":
    main()
