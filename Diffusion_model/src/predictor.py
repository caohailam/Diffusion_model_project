"""
Predictor classes for flow field prediction.

This module contains the `LatentDiffusionPredictor` - the main predictor class for
predicting 3D flow fields from 2D microstructures using diffusion in a shared VAE
latent space.

Architecture Overview:
    2D Input (microstructure + 2D flow)
        → Shared VAE Encoder
        → z2D
        → Diffusion U-Net
        → z3D
        → Shared VAE Decoder
    → 3D Flow Output

Training Pipeline:
    Stage 1: Train one shared VAE on both 2D and 3D velocity fields (VAE_model/train_2d3d_vae.py).
    Stage 2: Train diffusion U-Net to map z2D → z3D conditioned on microstructure (Diffusion_model/train.py).
"""

from abc import ABC, abstractmethod
from typing import Union, Any
import json
import os
import os.path as osp

import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage

from .normalizer import Normalizer, MaxNormalizer
from .unet.models import UNet
from utils.zenodo import download_data, unzip_data, is_url

import sys
sys.path.append(osp.join(osp.dirname(__file__), '..', '..'))
from VAE_model.src.vae.autoencoder import VariationalAutoencoder
#from VAE_model.src.dual_vae.model import DualBranchVAE


_model_type = Union[UNet, Any]

class Predictor(ABC, nn.Module):
    
    def __init__(
        self,
        model_name: str,
        model_kwargs: dict,
        distance_transform: bool,
    ):
        super().__init__()

        self.model_name = model_name
        self.model: _model_type = eval(model_name)(**model_kwargs)

        in_channels = model_kwargs['in_channels']
        out_channels = model_kwargs['out_channels']
        self.normalizer: dict[str, Normalizer] = self.init_normalizer(
            in_channels=in_channels,
            out_channels=out_channels
        )

        self.distance_transform = nn.Parameter(
            torch.Tensor([distance_transform]),
            requires_grad=False
        )
    
    @property
    @abstractmethod
    def type(self) -> str:
        pass

    @abstractmethod
    def forward(self, *args):
        pass

    @abstractmethod
    def predict(self, *args):
        pass

    @abstractmethod
    def pre_process(self, *args):
        pass

    @property
    def trainable_params(self):
        total = sum(
            [p.numel() for p in self.model.parameters() if p.requires_grad]
        )
        return total
    
    def init_normalizer(self, in_channels: int, out_channels: int):
        """Initialize normalizers for model input & output"""

        self.normalizer = nn.ModuleDict({
            'input': MaxNormalizer(scale_factors=[1 for _ in range(in_channels)]),
            'output': MaxNormalizer(scale_factors=[1 for _ in range(out_channels)])
        })
        return self.normalizer
    
    def set_normalizer(
        self,
        norm_dict: dict[str, Union[tuple, list, None]]
    ):
        """Set parameters for normalizers."""

        for key, val in norm_dict.items():
            if val is not None:
                self.normalizer[key] = MaxNormalizer(scale_factors=val)

        return self.normalizer

    def load_weights(self, model_path: str, device: str, strict: bool = True):
        """
        Load model parameters from `.pt` file.
        
        Args:
            model_path: Path to the model weights file.
            device: Device to load the model on.
            strict: If True, requires exact key match. If False, allows missing/unexpected keys.
        """
        state_dict = torch.load(model_path, map_location=torch.device(device))
        
        # Handle scheduler key mismatch (old models used simple attributes, new uses register_buffer)
        model_keys = set(self.state_dict().keys())
        loaded_keys = set(state_dict.keys())
        
        scheduler_keys_loaded = {k for k in loaded_keys if k.startswith('scheduler.')}
        scheduler_keys_model = {k for k in model_keys if k.startswith('scheduler.')}
        
        if scheduler_keys_loaded != scheduler_keys_model:
            print(f"Note: Scheduler format mismatch detected. Reinitializing scheduler.")
            # Remove scheduler keys from loaded state dict
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('scheduler.')}
            strict = False
        
        self.load_state_dict(state_dict, strict=strict)
        print(f'Loaded weights from "{model_path}".')

    @classmethod
    def from_directory(cls, folder: str, device: str) -> 'Predictor':
        """
        Load trained ML model from folder.
        
        Args:
            folder: Directory containing `model.pt` and `log.json` files created during training.
            device: Device (e.g. "cuda:0" or "cpu") to map the model to.
        """

        log_file = osp.join(folder, 'log.json')
        model_path = osp.join(folder, 'model.pt')

        with open(log_file) as fp:
            log_data = json.load(fp)
        
        param_dict = log_data['params']
        predictor_type = param_dict['training']['predictor_type']
        predictor_kwargs = param_dict['training']['predictor']

        if predictor_type == 'latent-diffusion':
            predictor_class = LatentDiffusionPredictor
        else:
            raise ValueError(f'Unknown or unsupported predictor type: {predictor_type}')

        predictor = predictor_class(**predictor_kwargs)
        predictor.to(device)
        predictor.load_weights(model_path, device=device)
        return predictor

    @classmethod
    def from_url(cls, url: str, device: str) -> 'Predictor':
        """
        Load trained ML model from URL. Pre-trained models are hosted here: https://doi.org/10.5281/zenodo.18341260.
        
        Args:
            url: URL pointing to a zipped folder containing `model.pt` and `log.json` files created during training.
            device: Device (e.g. "cuda:0" or "cpu") to map the model to.
        """

        _folder = 'pretrained'
        if not osp.exists(_folder): os.mkdir(_folder)

        # download pre-trained weights
        zip_path = download_data(url=url, save_dir=_folder)

        # unzip data
        folder_path = unzip_data(zip_path=zip_path, save_dir=_folder)

        predictor = cls.from_directory(folder_path, device=device)
        return predictor

    @classmethod
    def from_directory_or_url(
        cls,
        directory_or_url: str,
        device: str
    ) -> 'Predictor':
        """
        Load trained ML model from local directory or URL.
        
        Args:
            directory_or_url: either local directory or URL of the pre-trained model.
            device: Device (e.g. "cuda:0" or "cpu") to map the model to.
        """

        if is_url(directory_or_url):
            predictor = cls.from_url(url=directory_or_url, device=device)
        else:
            predictor = cls.from_directory(folder=directory_or_url, device=device)
        return predictor


class LatentDiffusionPredictor(Predictor):
    """
    Model for 3D velocity field prediction using multi-step diffusion in VAE latent space.
    """
    type: str = 'latent-diffusion'

    def __init__(
        self,
        model_name='UNet',
        model_kwargs: dict = {},
        distance_transform = True,
        vae_path: str = None,
        num_slices: int = 11,
        num_timesteps: int = 1000,
    ) -> None:
        
        # Helper imports
        from .diffusion import DiffusionScheduler

        # Add time embedding dimension to model kwargs if not present
        if 'time_embedding_dim' not in model_kwargs:
            model_kwargs['time_embedding_dim'] = 64
            
        super().__init__(
            model_name=model_name,
            model_kwargs=model_kwargs,
            distance_transform=distance_transform
        )
        
        self.num_slices = num_slices
        self.num_timesteps = num_timesteps
        
        # Init scheduler
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.scheduler = DiffusionScheduler(num_timesteps=self.num_timesteps, device=device)
        
        # Override normalizer for latent diffusion:
        # - Input: 1 channel (binary microstructure)
        # - Output: latent_channels (4 by default, set from model out_channels)
        latent_channels = model_kwargs.get('out_channels', 4)
        self.normalizer = nn.ModuleDict({
            'input': MaxNormalizer(scale_factors=[1]),  # 1 channel for microstructure
            'output': MaxNormalizer(scale_factors=[1 for _ in range(latent_channels)])
        })
        
        # Load pre-trained VAE
        if vae_path is None: #and (vae_encoder_path is None or vae_decoder_path is None)
            raise ValueError("VAE path must be provided for latent diffusion") # or both encoder and decoder paths must be specified")
        
        # Convert to absolute path if relative
        if vae_path is not None and not osp.isabs(vae_path):
            # Get the project root directory (2 levels up from this file)
            project_root = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
            vae_path = osp.join(project_root, vae_path)

        print(f"Loading shared VAE from {vae_path}...")

        # Load the single shared VAE used for both 2D and 3D fields
        self.vae = VariationalAutoencoder.from_directory(
            vae_path,
            device='cpu',
            in_channels=3,
            latent_channels=latent_channels,
            conditional=False
        )

        # Load the exact normalization factors used during VAE training
        vae_norm_factors = None
        vae_log_path = osp.join(vae_path, 'vae_log.json')

        if osp.exists(vae_log_path):
            with open(vae_log_path, 'r') as f:
                vae_log = json.load(f)

            if 'norm_factors' in vae_log:
                vae_norm_factors = vae_log['norm_factors']
                print(f"Loaded VAE norm_factors: {vae_norm_factors}")
        
        # Update output normalizer with VAE's norm_factors if available
        if vae_norm_factors is not None:
            # norm_factors is [max_u, max_v, max_w] for velocity channels
            self.normalizer['output'] = MaxNormalizer(scale_factors=vae_norm_factors)
            print(f"Set output normalizer to per-component: {vae_norm_factors}")
        else:
            raise ValueError(f"Could not find 'norm_factors' in {vae_log_path}")
            #print("WARNING: VAE norm_factors not found. Using default normalization.")
        
        # Store whether VAE norm_factors were loaded (to prevent overriding)
        self._vae_norm_loaded = True #vae_norm_factors is not None
        
        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()
        
        print(f'Initialized {self.type} predictor with {self.trainable_params} parameters.')
        print(f'Loaded VAE from {vae_path} (frozen).')
        
    def set_normalizer(self, norm_dict: dict):
        """
        Set normalizer parameters.
        
        For LatentDiffusionPredictor, if VAE norm_factors were loaded from vae_log.json,
        we skip overriding the output normalizer to ensure consistency with VAE training.
        """
        for key, val in norm_dict.items():
            if val is not None:
                # Skip output normalizer if VAE norm_factors were loaded
                if key == 'output' and hasattr(self, '_vae_norm_loaded') and self._vae_norm_loaded:
                    print(f"Keeping VAE norm_factors for '{key}' (not overriding with statistics.json)")
                    continue
                self.normalizer[key] = MaxNormalizer(scale_factors=val)
        return self.normalizer
        
    def to(self, device):
        super().to(device)
        self.scheduler.to(device)
        return self

    def forward(self, img: torch.Tensor, velocity_2d: torch.Tensor, x_start: torch.Tensor = None, noise: torch.Tensor = None):
        """
        Forward pass for multi-step diffusion model in latent space.
        
        Args:
            x_start: Target latents (clean data). Required for training step.
        """

        batch_size = img.shape[0]
        device = img.device
        num_slices = velocity_2d.shape[1]
        
        # img has shape (batch, num_slices, 1, H, W) - flatten to (batch*num_slices, 1, H, W)
        img_flat = img.view(batch_size * num_slices, 1, img.shape[3], img.shape[4])
        
        # Get latent dimensions from VAE encoder (uses 3D conv, expects 5D input)
        with torch.no_grad():
            # Create dummy input: (batch, channels, depth, height, width)
            dummy_5d = torch.zeros(1, 3, num_slices, img.shape[3], img.shape[4], device=device)
            latent_shape = self.vae.encoder(dummy_5d)[0].shape  # (1, latent_channels, depth, H/4, W/4)

            latent_channels = latent_shape[1]
            latent_depth = latent_shape[2]
            latent_h, latent_w = latent_shape[3], latent_shape[4]
        
        # Encode velocity_2d to latent space to use as conditioning
        # Permute to (batch, channels, num_slices, H, W) for 3D VAE
        velocity_2d_permuted = velocity_2d.permute(0, 2, 1, 3, 4)  # (batch, 3, num_slices, H, W)
        
        # CRITICAL FIX: Normalize velocity_2d before encoding (VAE was trained with normalized inputs)
        # Reshape for normalizer: (batch*depth, channels, H, W)
        v2d_shape = velocity_2d_permuted.shape  # (batch, 3, depth, H, W)
        velocity_2d_flat = velocity_2d_permuted.permute(0, 2, 1, 3, 4).contiguous().reshape(
            v2d_shape[0] * v2d_shape[2], v2d_shape[1], v2d_shape[3], v2d_shape[4]
        )  # (batch*depth, 3, H, W)
        velocity_2d_flat_norm = self.normalizer['output'](velocity_2d_flat)
        velocity_2d_norm_5d = velocity_2d_flat_norm.reshape(
            v2d_shape[0], v2d_shape[2], v2d_shape[1], v2d_shape[3], v2d_shape[4]
        ).permute(0, 2, 1, 3, 4)  # (batch, 3, depth, H, W)
        
        # Encode 2D velocity to latent space for conditioning
        # CRITICAL: Use deterministic encoding (mu only) for consistent conditioning
        with torch.no_grad():
            velocity_2d_latent_5d, _ = self.vae.encoder(velocity_2d_norm_5d)
        
        # Permute to (batch, depth, latent_channels, H, W)
        velocity_2d_latent = velocity_2d_latent_5d.permute(0, 2, 1, 3, 4)
        
        # Preprocess microstructure slices (apply distance transform if needed)
        feats_flat = self.pre_process(img_flat)  # Shape: (batch*num_slices, 1, H, W) at original resolution
        
        # Downsample microstructure features to latent resolution for conditioning
        feats_downsampled = torch.nn.functional.interpolate(
            feats_flat,
            size=(latent_h, latent_w),
            mode='bilinear',
            align_corners=False
        )  # (batch*num_slices, 1, latent_h, latent_w)
        
        # Flatten latents: (batch * latent_depth, latent_channels, latent_h, latent_w)
        # noise_flat = noise.reshape(batch_size * latent_depth, latent_channels, latent_h, latent_w) # We don't have noise here yet
        velocity_2d_latent_flat = velocity_2d_latent.reshape(batch_size * latent_depth, latent_channels, latent_h, latent_w)
        
        # Repeat microstructure features to match latent depth
        # We have num_slices microstructure slices but latent_depth latent slices
        # Interpolate feats to match latent_depth
        feats_3d = feats_downsampled.reshape(batch_size, num_slices, 1, latent_h, latent_w)
        
        # Interpolate along depth dimension to match latent_depth
        feats_3d_interp = torch.nn.functional.interpolate(
            feats_3d.permute(0, 2, 1, 3, 4),  # (batch, 1, num_slices, H, W)
            size=(latent_depth, latent_h, latent_w),
            mode='trilinear',
            align_corners=False
        ).permute(0, 2, 1, 3, 4)  # (batch, latent_depth, 1, H, W)
        
        feats_latent_flat = feats_3d_interp.reshape(batch_size * latent_depth, 1, latent_h, latent_w)
        
        # Training Logic
        if x_start is not None:
             # Flatten x_start to match latent_depth
             # x_start shape from encode_target: (batch, latent_depth, latent_channels, H, W)
             x_start_flat = x_start.reshape(batch_size * latent_depth, latent_channels, latent_h, latent_w)
             
             if noise is None:
                 noise = torch.randn_like(x_start_flat)
             else:
                 # Ensure noise is also flattened
                 noise = noise.reshape(batch_size * latent_depth, latent_channels, latent_h, latent_w)
                 
             # Sample timesteps
             t = torch.randint(0, self.num_timesteps, (batch_size * latent_depth,), device=device).long()
             
             # Add noise
             self.scheduler.to(device)
             x_t = self.scheduler.q_sample(x_start_flat, t, noise)
             
             # Network Input: Concatenate Noisy Latent + Conditioning
             unet_input = torch.cat([x_t, velocity_2d_latent_flat, feats_latent_flat], dim=1)
             
             # Predict noise
             noise_pred = self.model(unet_input, t)
             
             return noise_pred, noise
        
        else:
             raise ValueError("forward() requires x_start (target latents) for training. Use predict() for inference.")


    def predict(self, img: torch.Tensor, velocity_2d: torch.Tensor, noise: torch.Tensor = None):
        """
        Predict 3D velocity field using iterative denoising.
        """
        batch_size = img.shape[0]
        device = img.device
        num_slices = velocity_2d.shape[1]
        
        img_flat = img.view(batch_size * num_slices, 1, img.shape[3], img.shape[4])
        
        # Get dimensions
        with torch.no_grad():
            dummy_5d = torch.zeros(1, 3, num_slices, img.shape[3], img.shape[4], device=device)
            latent_shape = self.vae.encoder(dummy_5d)[0].shape
            latent_channels = latent_shape[1]
            latent_depth = latent_shape[2]
            latent_h, latent_w = latent_shape[3], latent_shape[4]
             
        # Prepare conditioning
        velocity_2d_permuted = velocity_2d.permute(0, 2, 1, 3, 4)
        
        # CRITICAL FIX: Normalize velocity_2d before encoding (VAE was trained with normalized inputs)
        # Reshape for normalizer: (batch*depth, channels, H, W)
        v2d_shape = velocity_2d_permuted.shape  # (batch, 3, depth, H, W)
        velocity_2d_flat = velocity_2d_permuted.permute(0, 2, 1, 3, 4).contiguous().reshape(
            v2d_shape[0] * v2d_shape[2], v2d_shape[1], v2d_shape[3], v2d_shape[4]
        )  # (batch*depth, 3, H, W)
        velocity_2d_flat_norm = self.normalizer['output'](velocity_2d_flat)
        velocity_2d_norm_5d = velocity_2d_flat_norm.reshape(
            v2d_shape[0], v2d_shape[2], v2d_shape[1], v2d_shape[3], v2d_shape[4]
        ).permute(0, 2, 1, 3, 4)  # (batch, 3, depth, H, W)
        
        with torch.no_grad():
            velocity_2d_latent_5d, _ = self.vae.encoder(velocity_2d_norm_5d)
    
        velocity_2d_latent = velocity_2d_latent_5d.permute(0, 2, 1, 3, 4)
        
        feats_flat = self.pre_process(img_flat)
        feats_downsampled = torch.nn.functional.interpolate(feats_flat, size=(latent_h, latent_w), mode='bilinear', align_corners=False)
        
        velocity_2d_latent_flat = velocity_2d_latent.reshape(batch_size * latent_depth, latent_channels, latent_h, latent_w)
        
        feats_3d = feats_downsampled.reshape(batch_size, num_slices, 1, latent_h, latent_w)
        feats_3d_interp = torch.nn.functional.interpolate(
            feats_3d.permute(0, 2, 1, 3, 4),
            size=(latent_depth, latent_h, latent_w),
            mode='trilinear',
            align_corners=False
        ).permute(0, 2, 1, 3, 4)
        feats_latent_flat = feats_3d_interp.reshape(batch_size * latent_depth, 1, latent_h, latent_w)
        
        # Sampling Loop
        if noise is None:
            noise = torch.randn(batch_size * latent_depth, latent_channels, latent_h, latent_w, device=device)
        else:
            noise = noise.reshape(batch_size * latent_depth, latent_channels, latent_h, latent_w)
            
        x = noise
        self.scheduler.to(device)
        
        if self.num_timesteps == 1:
            # One-step diffusion: predict noise at maximum timestep and denoise in one step
            t = self.num_timesteps - 1  # t=0 for num_timesteps=1
            t_batch = torch.full((batch_size * latent_depth,), t, device=device, dtype=torch.long)
            
            unet_input = torch.cat([x, velocity_2d_latent_flat, feats_latent_flat], dim=1)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.model(unet_input, t_batch)
            
            # One-step denoising: x_0 = (x_t - sqrt(1-alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)
            alpha_bar = self.scheduler.alphas_cumprod[t]
            x = (x - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
            # Clip to reasonable range in latent space
            x = torch.clamp(x, -30.0, 30.0)
        else:
            # Multi-step diffusion: iterative denoising
            for t in reversed(range(0, self.num_timesteps)):
                 t_batch = torch.full((batch_size * latent_depth,), t, device=device, dtype=torch.long)
                 
                 unet_input = torch.cat([x, velocity_2d_latent_flat, feats_latent_flat], dim=1)
                 
                 # Predict noise
                 with torch.no_grad():
                      noise_pred = self.model(unet_input, t_batch)
                 
                 # Step - use wider clip range for latent space
                 x = self.scheduler.p_sample(noise_pred, x, t, clip_denoised=True, clip_range=(-30.0, 30.0))
        
        predicted_latent_flat = x
        
        # Decode
        predicted_latents = predicted_latent_flat.reshape(batch_size, latent_depth, latent_channels, latent_h, latent_w)
        predicted_latents_5d = predicted_latents.permute(0, 2, 1, 3, 4)
        
        with torch.no_grad():
            velocity_5d = self.vae.decoder(predicted_latents_5d)
        
        # Permute back to (batch, num_slices, 3, H, W)
        velocity_3d = velocity_5d.permute(0, 2, 1, 3, 4)
        
        # Denormalize
        batch, depth, channels, height, width = velocity_3d.shape
        velocity_flat = velocity_3d.reshape(batch * depth, channels, height, width)
        velocity_flat = self.normalizer['output'].inverse(velocity_flat)
        velocity_3d = velocity_flat.reshape(batch, depth, channels, height, width)

        # Interpolate back to original number of slices if mismatch (e.g. VAE output 8 vs input 11)
        if depth != num_slices:
            velocity_3d = torch.nn.functional.interpolate(
                velocity_3d.permute(0, 2, 1, 3, 4), # (batch, channels, depth, height, width)
                size=(num_slices, height, width),
                mode='trilinear',
                align_corners=False
            ).permute(0, 2, 1, 3, 4)

        # Masking
        # img shape might be (batch, num_slices, 1, H, W) or (batch, 1, 1, H, W)
        # Ensure img is broadcastable or matches dimensions
        if img.dim() == 5 and img.shape[1] != velocity_3d.shape[1] and img.shape[1] == 1:
             # img is (batch, 1, 1, H, W), velocity is (batch, 11, ...)
             # Let it broadcast
             pass
        
        velocity_3d = velocity_3d * img
        
        return velocity_3d
    
    def predict_ddim(self, img: torch.Tensor, velocity_2d: torch.Tensor, num_steps: int = 50, eta: float = 0.0, noise: torch.Tensor = None):
        """
        Predict 3D velocity field using DDIM sampling (faster than DDPM).
        
        Args:
            img: Microstructure images. Shape: (batch, num_slices, 1, H, W)
            velocity_2d: 2D velocity input. Shape: (batch, num_slices, 3, H, W) 
            num_steps: Number of DDIM sampling steps (default 50, can be as low as 20)
            eta: DDIM stochasticity (0 = deterministic, 1 = DDPM-like)
            noise: Optional initial noise
        """
        batch_size = img.shape[0]
        device = img.device
        num_slices = velocity_2d.shape[1]
        
        img_flat = img.view(batch_size * num_slices, 1, img.shape[3], img.shape[4])
        
        # Get dimensions
        with torch.no_grad():
            dummy_5d = torch.zeros(1, 3, num_slices, img.shape[3], img.shape[4], device=device)
            latent_shape = self.vae.encoder(dummy_5d)[0].shape
            latent_channels = latent_shape[1]
            latent_depth = latent_shape[2]
            latent_h = latent_shape[3]
            latent_w = latent_shape[4]
             
        # Prepare conditioning
        velocity_2d_permuted = velocity_2d.permute(0, 2, 1, 3, 4)
        
        # CRITICAL FIX: Normalize velocity_2d before encoding (VAE was trained with normalized inputs)
        # Reshape for normalizer: (batch*depth, channels, H, W)
        v2d_shape = velocity_2d_permuted.shape  # (batch, 3, depth, H, W)
        velocity_2d_flat = velocity_2d_permuted.permute(0, 2, 1, 3, 4).contiguous().reshape(
            v2d_shape[0] * v2d_shape[2], v2d_shape[1], v2d_shape[3], v2d_shape[4]
        )  # (batch*depth, 3, H, W)
        velocity_2d_flat_norm = self.normalizer['output'](velocity_2d_flat)
        velocity_2d_norm_5d = velocity_2d_flat_norm.reshape(
            v2d_shape[0], v2d_shape[2], v2d_shape[1], v2d_shape[3], v2d_shape[4]
        ).permute(0, 2, 1, 3, 4)  # (batch, 3, depth, H, W)
        
        with torch.no_grad():
            velocity_2d_latent_5d, _ = self.vae.encoder(velocity_2d_norm_5d)

        velocity_2d_latent = velocity_2d_latent_5d.permute(0, 2, 1, 3, 4)
        
        feats_flat = self.pre_process(img_flat)
        feats_downsampled = torch.nn.functional.interpolate(feats_flat, size=(latent_h, latent_w), mode='bilinear', align_corners=False)
        
        velocity_2d_latent_flat = velocity_2d_latent.reshape(batch_size * latent_depth, latent_channels, latent_h, latent_w)
        
        feats_3d = feats_downsampled.reshape(batch_size, num_slices, 1, latent_h, latent_w)
        feats_3d_interp = torch.nn.functional.interpolate(
            feats_3d.permute(0, 2, 1, 3, 4),
            size=(latent_depth, latent_h, latent_w),
            mode='trilinear',
            align_corners=False
        ).permute(0, 2, 1, 3, 4)
        feats_latent_flat = feats_3d_interp.reshape(batch_size * latent_depth, 1, latent_h, latent_w)
        
        # Create DDIM timestep schedule
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, dtype=torch.long, device=device)
        
        # Sampling Loop with DDIM
        if noise is None:
            noise = torch.randn(batch_size * latent_depth, latent_channels, latent_h, latent_w, device=device)
        else:
            noise = noise.reshape(batch_size * latent_depth, latent_channels, latent_h, latent_w)
            
        x = noise
        self.scheduler.to(device)
        
        for i in range(len(timesteps)):
            t = timesteps[i].item()
            t_prev = timesteps[i + 1].item() if i + 1 < len(timesteps) else -1
            
            t_batch = torch.full((batch_size * latent_depth,), t, device=device, dtype=torch.long)
            
            unet_input = torch.cat([x, velocity_2d_latent_flat, feats_latent_flat], dim=1)
            
            with torch.no_grad():
                noise_pred = self.model(unet_input, t_batch)
            
            # DDIM step
            x = self.scheduler.ddim_sample(noise_pred, x, t, t_prev, eta=eta, clip_range=(-30.0, 30.0))
        
        predicted_latent_flat = x
        
        # Decode
        predicted_latents = predicted_latent_flat.reshape(batch_size, latent_depth, latent_channels, latent_h, latent_w)
        predicted_latents_5d = predicted_latents.permute(0, 2, 1, 3, 4)
        
        with torch.no_grad():
            velocity_5d = self.vae.decoder(predicted_latents_5d)
        
        velocity_3d = velocity_5d.permute(0, 2, 1, 3, 4)
        
        # Denormalize
        batch, depth, channels, height, width = velocity_3d.shape
        velocity_flat = velocity_3d.reshape(batch * depth, channels, height, width)
        velocity_flat = self.normalizer['output'].inverse(velocity_flat)
        velocity_3d = velocity_flat.reshape(batch, depth, channels, height, width)

        if depth != num_slices:
            velocity_3d = torch.nn.functional.interpolate(
                velocity_3d.permute(0, 2, 1, 3, 4),
                size=(num_slices, height, width),
                mode='trilinear',
                align_corners=False
            ).permute(0, 2, 1, 3, 4)

        velocity_3d = velocity_3d * img
        
        return velocity_3d

    def pre_process(self, img: torch.Tensor):
        """
        Pre-process microstructure inputs.

        Args:
            img: (binary) microstructure images. Shape: (batch, 1, height, width).
        """
        assert img.dim() == 4
        assert img.shape[1] == 1

        if self.distance_transform:
            img = apply_distance_transform(img)

        features = self.normalizer['input'](img)

        return features
    
    def encode_target(self, velocity_3d: torch.Tensor, velocity_2d: torch.Tensor = None):
        """
        Encode 3D velocity target into latent space using E3D encoder.
        
        Args:
            velocity_3d: Target 3D velocity field. Shape: (batch, num_slices, 3, H, W)
            velocity_2d: Input 2D velocity field (unused, kept for backward compatibility).
        
        Returns:
            latents: Encoded latent representations. Shape: (batch, num_slices, latent_channels, latent_h, latent_w)
        """
        # VAE expects 5D input: (batch, channels, depth, height, width)
        # Our data is: (batch, num_slices, channels, height, width)
        # Permute to match VAE input: (batch, channels, num_slices, height, width)
        velocity_permuted = velocity_3d.permute(0, 2, 1, 3, 4)  # (batch, 3, num_slices, H, W)
        
        # Normalize - VAE expects normalized input
        batch_size = velocity_permuted.shape[0]
        channels = velocity_permuted.shape[1]
        depth = velocity_permuted.shape[2]
        height = velocity_permuted.shape[3]
        width = velocity_permuted.shape[4]
        
        # Reshape to (batch*depth, channels, height, width) for normalization
        velocity_flat = velocity_permuted.permute(0, 2, 1, 3, 4).contiguous().reshape(batch_size * depth, channels, height, width)
        velocity_flat_norm = self.normalizer['output'](velocity_flat)
        # Reshape back to 5D: (batch, depth, channels, H, W) then permute to (batch, channels, depth, H, W)
        velocity_norm_5d = velocity_flat_norm.reshape(batch_size, depth, channels, height, width).permute(0, 2, 1, 3, 4)
        
        # Encode velocity target with E3D encoder (no gradient through VAE)
        # CRITICAL: Use deterministic encoding (mu only) to avoid random sampling noise
        with torch.no_grad():
            latent_5d, _ = self.vae.encoder(velocity_norm_5d)
        
        # Permute back to (batch, depth, latent_channels, H, W) to match expected output
        latents = latent_5d.permute(0, 2, 1, 3, 4)  # (batch, depth/4, latent_channels, H/4, W/4)
        
        return latents
    

def apply_distance_transform(imgs: torch.Tensor):
    """
    Perform distance transform of input images.

    Args:
        imgs: batch of images, with shape (n_img, 1, height, width)
    """
    device = imgs.device

    imgs = imgs.cpu().numpy()

    tmp_list = []
    for im in imgs:
        im = im[0]
        im_tr = ndimage.distance_transform_edt(im)
        tmp_list.append([[im_tr]])
    
    out = torch.from_numpy(
        np.concatenate(tmp_list)
    ).float()
    return out.to(device)
