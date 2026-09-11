import argparse
import sys
import os
import os.path as osp
import json
import torch
import numpy as np

# Add project root and Diffusion_model to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

diffusion_model_path = os.path.join(project_root, 'Diffusion_model')
if diffusion_model_path not in sys.path:
    sys.path.append(diffusion_model_path)

from Diffusion_model.src.helper import set_model
from Diffusion_model.utils.dataset import get_loader

def main():
    parser = argparse.ArgumentParser(description="Inference for Microstructure Flow Prediction")
    parser.add_argument('--diffusion-model-path', type=str, required=True, help='Path to the trained diffusion model (directory or .pt file)')
    parser.add_argument('--sample-path', type=str, default=None, help='Path to the input sample (.pt file). If not provided, uses a sample from the test set.')
    parser.add_argument('--vae-path', type=str, default=None, help='Path to the trained VAE model directory. If not provided, uses VAE path from model config.')
    parser.add_argument('--vae-encoder-path', type=str, default=None, help='Path to VAE encoder weights (E2D, e.g. Stage 2). Optional.')
    parser.add_argument('--vae-decoder-path', type=str, default=None, help='Path to VAE decoder weights (D3D/E3D, e.g. Stage 1). Optional.')
    parser.add_argument('--dataset-dir', type=str, default=os.path.join(current_dir, '..', 'dataset_3d'), help='Path to the dataset directory. Used to locate statistics.json and test samples. Default: ../dataset_3d')
    parser.add_argument('--index', type=int, default=0, help='Index of the sample in the test set to use (only if sample_path is not provided). Default: 0')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    device = args.device

    print(f"--- Setting up Inference on {device} ---")

    # --- 1. & 2. Import already trained VAE and Diffusion Model ---
    # We use the helper function set_model which handles loading the VAE internally 
    # within LatentDiffusionPredictor logic based on config.
    
    if os.path.isfile(args.diffusion_model_path):
        model_dir = os.path.dirname(args.diffusion_model_path)
        weights_file = args.diffusion_diffusion_model_path
    else:
        model_dir = args.diffusion_model_path
        # Prefer best_model.pt over model.pt (model.pt may be corrupted or incomplete)
        best_model_path = os.path.join(model_dir, 'best_model.pt')
        model_path = os.path.join(model_dir, 'model.pt')
        if os.path.exists(best_model_path):
            weights_file = best_model_path
        elif os.path.exists(model_path):
            weights_file = model_path
        else:
            raise FileNotFoundError(f"No model weights found in {model_dir}. Expected best_model.pt or model.pt")
        
    print(f"Loading model configuration from {model_dir}")
    log_path = os.path.join(model_dir, 'log.json')
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"log.json not found in {model_dir}")
        
    with open(log_path, 'r') as f:
        log_try = json.load(f)
        if 'params' in log_try:
            config = log_try['params']
        else:
            config = log_try 

    predictor_kwargs = config['training']['predictor']
    
    # Dataset directory - defaults to ../dataset_3d relative to Inference folder
    dataset_root = os.path.abspath(args.dataset_dir)
    print(f"Dataset directory: {dataset_root}")
    
    # Load statistics.json for normalization from dataset directory
    norm_file = os.path.join(dataset_root, 'statistics.json')
    if not os.path.exists(norm_file):
        # Show what files exist in the directory for debugging
        if os.path.exists(dataset_root):
            files = os.listdir(dataset_root)
            raise FileNotFoundError(
                f"statistics.json not found in {dataset_root}\n"
                f"Files in directory: {files}"
            )
        else:
            raise FileNotFoundError(
                f"Dataset directory does not exist: {dataset_root}\n"
                f"Please provide --dataset-dir pointing to the dataset folder."
            )
    print(f"Using statistics from: {norm_file}")
            
    # Fix VAE path in config if it was absolute path from another machine

    # Handle VAE path logic (vae_path, vae_encoder_path, vae_decoder_path)
    def resolve_path(path):
        if not path:
            return None
        if os.path.exists(path):
            return path
        if not os.path.isabs(path):
            abs_path = os.path.join(project_root, path)
            if os.path.exists(abs_path):
                return abs_path
        if 'VAE_model' in path:
            idx = path.find('VAE_model')
            abs_path = os.path.join(project_root, path[idx:])
            if os.path.exists(abs_path):
                return abs_path
        return path  # fallback

    # Always set vae_path (required)
    vae_path = args.vae_path or predictor_kwargs.get('vae_path', None)
    if vae_path:
        vae_path = resolve_path(vae_path)
        predictor_kwargs['vae_path'] = vae_path
        print(f"Using VAE path: {vae_path}")

    # Optionally set vae_encoder_path and vae_decoder_path
    if args.vae_encoder_path:
        vae_encoder_path = resolve_path(args.vae_encoder_path)
        predictor_kwargs['vae_encoder_path'] = vae_encoder_path
        print(f"Using VAE encoder path: {vae_encoder_path}")
    if args.vae_decoder_path:
        vae_decoder_path = resolve_path(args.vae_decoder_path)
        predictor_kwargs['vae_decoder_path'] = vae_decoder_path
        print(f"Using VAE decoder path: {vae_decoder_path}")

    print("Initializing models...")
    # This initializes LatentDiffusionPredictor, which loads the VAE (Step 1)
    predictor = set_model(
        type='latent-diffusion',
        kwargs=predictor_kwargs,
        norm_file=norm_file
    )
    
    # Load Diffusion Model Weights (Step 2)
    # We use strict=False because:
    # 1. The checkpoint may have VAE weights from a different architecture (standard vs dual VAE)
    # 2. We've already loaded the VAE weights separately with proper handling
    print(f"Loading weights from {weights_file}")
    loaded_state = torch.load(weights_file, map_location=device)
    
    # Filter out VAE weights - we've already loaded them
    model_state = {k: v for k, v in loaded_state.items() if not k.startswith('vae.')}
    
    # Load model weights with strict=False to allow missing VAE keys
    missing, unexpected = predictor.load_state_dict(model_state, strict=False)
    
    # Only print warnings for non-VAE keys
    non_vae_missing = [k for k in missing if not k.startswith('vae.')]
    non_vae_unexpected = [k for k in unexpected if not k.startswith('vae.')]
    if non_vae_missing:
        print(f"Warning: Missing keys (non-VAE): {non_vae_missing}")
    if non_vae_unexpected:
        print(f"Warning: Unexpected keys (non-VAE): {non_vae_unexpected}")
    
    print(f"Loaded model weights. VAE weights were loaded separately with dual VAE handling.")
    predictor.to(device)
    predictor.eval()

    # --- Load Sample ---
    if args.sample_path:
        # Passing sample from file path 
        print(f"Loading sample from {args.sample_path}")
        sample_data = torch.load(args.sample_path, map_location=device)
        
        # Parse sample structure
        if isinstance(sample_data, dict):
            img = sample_data.get('microstructure')
            velocity_2d = sample_data.get('velocity_input')
            if velocity_2d is None:
                velocity_2d = sample_data.get('U_2d')
            
            target_velocity = sample_data.get('velocity')
            if target_velocity is None:
                target_velocity = sample_data.get('U')
        elif isinstance(sample_data, (list, tuple)):
            img = sample_data[0]
            velocity_2d = sample_data[1]
            target_velocity = sample_data[2] if len(sample_data) > 2 else None
        else:
            img = sample_data # Fallback if just tensor
            velocity_2d = None # Will fail if model needs it

        # Handle batch dimensions
        if img.dim() == 4: img = img.unsqueeze(0)
        if velocity_2d is not None and velocity_2d.dim() == 4: velocity_2d = velocity_2d.unsqueeze(0)
        if target_velocity is not None and target_velocity.dim() == 4: target_velocity = target_velocity.unsqueeze(0)
    
    else:
        # Use Test Set
        print(f"No sample path provided. Loading Test Set (seed=2024, index={args.index})...")
        
        # Use root_dir from config
        loaders = get_loader(
            root_dir=dataset_root,
            batch_size=1, # One sample per batch
            val_ratio=0.15,
            test_ratio=0.15,
            shuffle=False, # Important for consistent indexing inside the loader
            seed=2024,     # Enforce seed 2024
            augment=False,
            use_3d=True,   # Defaulting to true as most complex case, usually inferred from logic but let's assume 3D based on project
            num_workers=0  # Use 0 workers to avoid Windows multiprocessing issues
        )
        
        # get_loader with default k_folds=None returns [(train, val, test)]
        train_loader, val_loader, test_loader = loaders[0]
        
        if len(test_loader) <= args.index:
            raise IndexError(f"Test set size ({len(test_loader)}) is smaller than requested index {args.index}")
            
        print(f"Fetching sample at index {args.index} from test set...")
        # Efficiently skip to index
        for i, data in enumerate(test_loader):
            if i == args.index:
                # data is a dict because use_3d=True usually or depends on dataset implementation
                # Based on dataset.py MicroFlowDataset returns dict
                img = data['microstructure']
                velocity_2d = data.get('velocity_input')
                if velocity_2d is None:
                    velocity_2d = data.get('U_2d')

                target_velocity = data.get('velocity')
                if target_velocity is None:
                    target_velocity = data.get('U')
                
                dxyz = data.get('dxyz')
                break
    
    img = img.to(device)
    if velocity_2d is not None:
        velocity_2d = velocity_2d.to(device)

    print(f"Input Microstructure shape: {img.shape}")
    if velocity_2d is not None:
        print(f"Input Velocity shape: {velocity_2d.shape}")

    # --- 3, 4, 5. Run Prediction Pipeline ---
    # The predictor.predict() method internally performs:
    # 3. Pass sample (velocity_2d) to VAE encoder (to get conditioning latents)
    # 4. Pass output of 3 (+ microstructure features) to Diffusion Model (denoising loop)
    # 5. Pass output of 4 (denoised latents) to VAE decoder
    
    print("Running prediction (VAE Encode -> Diffusion -> VAE Decode)...")
    with torch.no_grad():
        prediction = predictor.predict(img, velocity_2d)
    print("Prediction complete.")
    print(f"Prediction output shape: {prediction.shape}")

    # --- 6a. Matplotlib 2D Slice Visualization with Colorbars ---
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        print("\nGenerating 2D slice visualization with velocity scale...")
        
        def to_numpy_plt(t):
            return t[0].detach().cpu().numpy()
        
        # Get prediction data: (slices, 3, H, W)
        pred_np = to_numpy_plt(prediction)
        num_slices = pred_np.shape[0]
        middle_slice = 3 #num_slices // 2
        
        # Calculate velocity magnitude
        pred_mag = np.sqrt(np.sum(pred_np**2, axis=1))  # (slices, H, W)
        
        # Get microstructure
        micro_np = to_numpy_plt(img)
        if micro_np.ndim == 4 and micro_np.shape[1] == 1:
            micro_np = micro_np[:, 0, :, :]  # (slices, H, W)
        
        # Get target if available
        target_np = None
        target_mag = None
        if target_velocity is not None:
            if target_velocity.dim() == 4:
                target_velocity = target_velocity.unsqueeze(0)
            target_np = to_numpy_plt(target_velocity)
            target_mag = np.sqrt(np.sum(target_np**2, axis=1))
        
        # Find global min/max for consistent colorbar across pred/target
        vmin_mag = pred_mag.min()
        vmax_mag = pred_mag.max()
        if target_mag is not None:
            vmin_mag = min(vmin_mag, target_mag.min())
            vmax_mag = max(vmax_mag, target_mag.max())
        
        # Create figure with colorbars
        if target_np is not None:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Velocity Field Comparison (Slice {middle_slice}/{num_slices})', fontsize=14)
            
            # Row 1: Prediction
            # Microstructure overlay
            ax = axes[0, 0]
            im = ax.imshow(pred_mag[middle_slice], cmap='coolwarm', vmin=vmin_mag, vmax=vmax_mag)
            ax.contour(micro_np[middle_slice], levels=[0.5], colors='black', linewidths=0.5)
            ax.set_title('Prediction: Velocity Magnitude')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('|V| (m/s)')
            
            # Vx component
            ax = axes[0, 1]
            vx_min, vx_max = pred_np[:, 0].min(), pred_np[:, 0].max()
            if target_np is not None:
                vx_min = min(vx_min, target_np[:, 0].min())
                vx_max = max(vx_max, target_np[:, 0].max())
            im = ax.imshow(pred_np[middle_slice, 0], cmap='coolwarm', vmin=vx_min, vmax=vx_max)
            ax.contour(micro_np[middle_slice], levels=[0.5], colors='black', linewidths=0.5)
            ax.set_title('Prediction: Vx')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('Vx (m/s)')
            
            # Vy component  
            ax = axes[0, 2]
            vy_min, vy_max = pred_np[:, 1].min(), pred_np[:, 1].max()
            if target_np is not None:
                vy_min = min(vy_min, target_np[:, 1].min())
                vy_max = max(vy_max, target_np[:, 1].max())
            im = ax.imshow(pred_np[middle_slice, 1], cmap='coolwarm', vmin=vy_min, vmax=vy_max)
            ax.contour(micro_np[middle_slice], levels=[0.5], colors='black', linewidths=0.5)
            ax.set_title('Prediction: Vy')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('Vy (m/s)')
            
            # Row 2: Target
            ax = axes[1, 0]
            im = ax.imshow(target_mag[middle_slice], cmap='coolwarm', vmin=vmin_mag, vmax=vmax_mag)
            ax.contour(micro_np[middle_slice], levels=[0.5], colors='black', linewidths=0.5)
            ax.set_title('Target: Velocity Magnitude')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('|V| (m/s)')
            
            ax = axes[1, 1]
            im = ax.imshow(target_np[middle_slice, 0], cmap='coolwarm', vmin=vx_min, vmax=vx_max)
            ax.contour(micro_np[middle_slice], levels=[0.5], colors='black', linewidths=0.5)
            ax.set_title('Target: Vx')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('Vx (m/s)')
            
            ax = axes[1, 2]
            im = ax.imshow(target_np[middle_slice, 1], cmap='coolwarm', vmin=vy_min, vmax=vy_max)
            ax.contour(micro_np[middle_slice], levels=[0.5], colors='black', linewidths=0.5)
            ax.set_title('Target: Vy')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('Vy (m/s)')
            
        else:
            # No target - just show prediction
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'Predicted Velocity Field (Slice {middle_slice}/{num_slices})', fontsize=14)
            
            ax = axes[0]
            im = ax.imshow(pred_mag[middle_slice], cmap='coolwarm', vmin=vmin_mag, vmax=vmax_mag)
            ax.contour(micro_np[middle_slice], levels=[0.5], colors='black', linewidths=0.5)
            ax.set_title('Velocity Magnitude')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('|V| (m/s)')
            
            ax = axes[1]
            im = ax.imshow(pred_np[middle_slice, 0], cmap='coolwarm')
            ax.contour(micro_np[middle_slice], levels=[0.5], colors='black', linewidths=0.5)
            ax.set_title('Vx')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('Vx (m/s)')
            
            ax = axes[2]
            im = ax.imshow(pred_np[middle_slice, 1], cmap='coolwarm')
            ax.contour(micro_np[middle_slice], levels=[0.5], colors='black', linewidths=0.5)
            ax.set_title('Vy')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('Vy (m/s)')
        
        plt.tight_layout()
        plt.savefig(f'Inference/velocity_field_comparison_slice_{middle_slice}.png', dpi=150, bbox_inches='tight')
        print(f"Saved 2D visualization to 'Inference/velocity_field_comparison_slice_{middle_slice}.png'")
        plt.show()
        
    except Exception as e:
        print(f"Error during matplotlib visualization: {e}")

    # --- 6b. Visualizer (Napari) ---
    try:
        import napari
        
        print("Starting Napari viewer...")
        viewer = napari.Viewer()
        
        # Calculate Scale based on user request (50x50x50 um volume, 5um z-spacing)
        # prediction shape is (Batch, Channels, Depth, Height, Width)
        depth = prediction.shape[2]
        height = prediction.shape[3]
        width = prediction.shape[4]
        
        # Z-spacing is explicitly 5.0 um
        # X and Y dimensions total 50.0 um
        z_scale = 5.0
        y_scale = 50.0 / height
        x_scale = 50.0 / width
        
        scale = [z_scale, y_scale, x_scale]
        print(f"Using physical scale: {scale} (z, y, x) for 50x50x50um volume")

        def to_numpy(t):
            return t[0].detach().cpu().numpy()

        # Add Microstructure (Slices, H, W)
        micro = to_numpy(img)
        # Check dims: (slices, 1, H, W) -> remove channel dim
        if micro.ndim == 4 and micro.shape[1] == 1:
            micro = micro[:, 0, :, :]
        
        viewer.add_image(micro, name='Microstructure', opacity=0.3, colormap='gray', rendering='translucent', depiction='volume', scale=scale)

        # Add Prediction (Slices, 3, H, W)
        pred = to_numpy(prediction)
        
        # Normalize each component using ABSOLUTE VALUE normalization
        # This is critical for velocity data symmetric around zero
        # Using (x - p1)/(p99 - p1) normalization causes issues when data is symmetric:
        # - Zero maps to ~0.5 which is the default iso threshold
        # - This causes most of the volume to appear solid!
        # Instead, use |x| / p99(|x|) to ensure zero stays at zero
        pred_normalized = np.zeros_like(pred)
        
        print("\nVelocity ranges (physical, m/s):")
        for i, component in enumerate(['V_x', 'V_y', 'V_z'][:pred.shape[1]]):
            comp_data = pred[:, i]
            abs_data = np.abs(comp_data)
            abs_p99 = np.percentile(abs_data, 99)
            print(f"  {component}: [{comp_data.min():.6f}, {comp_data.max():.6f}] (|x| p99: {abs_p99:.6f})")
            
            # Normalize using absolute value - ensures zero stays at zero
            if abs_p99 > 0:
                pred_normalized[:, i] = abs_data / abs_p99
            else:
                pred_normalized[:, i] = abs_data
        
        # Calculate Magnitude (use original physical values)
        pred_mag = np.sqrt(np.sum(pred**2, axis=1))
        # Normalize magnitude separately
        mag_min, mag_max = np.percentile(pred_mag, [1, 99])
        if mag_max > mag_min:
            pred_mag_normalized = (pred_mag - mag_min) / (mag_max - mag_min)
        else:
            pred_mag_normalized = pred_mag
        
        # Add components (each normalized to [0,1] independently)
        # Channels: 0=x, 1=y, 2=z
        viewer.add_image(pred_normalized[:, 0], name='Pred V_x', colormap='magma', rendering='iso', depiction='volume', scale=scale, blending='additive')
        viewer.add_image(pred_normalized[:, 1], name='Pred V_y', colormap='magma', visible=False, rendering='iso', depiction='volume', scale=scale, blending='additive')
        if pred.shape[1] > 2:
            viewer.add_image(pred_normalized[:, 2], name='Pred V_z', colormap='magma', visible=False, rendering='iso', depiction='volume', scale=scale, blending='additive')
            
        viewer.add_image(pred_mag_normalized, name='Pred Magnitude', colormap='turbo', visible=True, rendering='mip', depiction='volume', scale=scale, blending='additive')

        # Add Target if available
        if target_velocity is not None:
            if target_velocity.dim() == 4: target_velocity = target_velocity.unsqueeze(0)
            target = to_numpy(target_velocity)
            
            # Normalize using ABSOLUTE VALUE (same as prediction)
            # This ensures zero-centered data maps zero to zero, not ~0.5
            target_normalized = np.zeros_like(target)
            
            print("\nTarget velocity ranges (physical, m/s):")
            for i, component in enumerate(['V_x', 'V_y', 'V_z'][:target.shape[1]]):
                comp_data = target[:, i]
                abs_data = np.abs(comp_data)
                abs_p99 = np.percentile(abs_data, 99)
                print(f"  {component}: [{comp_data.min():.6f}, {comp_data.max():.6f}] (|x| p99: {abs_p99:.6f})")
                
                # Normalize using absolute value - ensures zero stays at zero
                if abs_p99 > 0:
                    target_normalized[:, i] = abs_data / abs_p99
                else:
                    target_normalized[:, i] = abs_data
            
            target_mag = np.sqrt(np.sum(target**2, axis=1))
            mag_min, mag_max = np.percentile(target_mag, [1, 99])
            if mag_max > mag_min:
                target_mag_normalized = (target_mag - mag_min) / (mag_max - mag_min)
            else:
                target_mag_normalized = target_mag
            
            viewer.add_image(target_normalized[:, 0], name='Target V_x', colormap='magma', visible=False, rendering='iso', depiction='volume', scale=scale)
            viewer.add_image(target_normalized[:, 1], name='Target V_y', colormap='magma', visible=False, rendering='iso', depiction='volume', scale=scale)
            if target.shape[1] > 2:
                viewer.add_image(target_normalized[:, 2], name='Target V_z', colormap='magma', visible=False, rendering='iso', depiction='volume', scale=scale)
                
            viewer.add_image(target_mag_normalized, name='Target Magnitude', colormap='turbo', visible=False, rendering='mip', depiction='volume', scale=scale)

        print("Opening napari window...")
        # Switch to 3D display mode
        viewer.dims.ndisplay = 3
        napari.run()
        
    except ImportError as e:
        print(f"Napari not found (ImportError: {e}). Skipping visualization. Try 'pip install napari[all]'")
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == "__main__":
    main()
