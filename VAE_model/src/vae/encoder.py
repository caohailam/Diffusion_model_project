import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualBlock, ConditionalResidualBlock, FiLM, AttentionBlock
from ..common import get_padding, SelfAttention


class Encoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        conditional: bool = False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conditional = conditional
        padding = get_padding(self.kernel_size)
        
        # Use asymmetric stride to preserve depth dimension for z-flow accuracy
        # Only downsample in H and W, NOT in depth D
        
        # Initial conv
        self.conv_in = nn.Conv3d(self.in_channels, 64, kernel_size=self.kernel_size, padding=padding) #128
        
        # FiLM conditioning after initial conv (when conditional=True)
        if self.conditional:
            self.film_in = FiLM(condition_dim=1, feature_channels=64) #128
        
        # Stage 1: 128 channels
        if self.conditional:
            self.res1_1 = ConditionalResidualBlock(64, 64, conditional=True, condition_dim=1) #128,128
            self.res1_2 = ConditionalResidualBlock(64, 64, conditional=True, condition_dim=1) #128,128
        else:
            self.res1_1 = ResidualBlock(64, 64) #128,128
            self.res1_2 = ResidualBlock(64, 64) #128,128
        
        # Downsample 1: H/2, W/2
        self.down1 = nn.Conv3d(64, 64, kernel_size=self.kernel_size, stride=(1, 2, 2), padding=0) #128,128
        
        # Stage 2: 256 channels
        if self.conditional:
            self.res2_1 = ConditionalResidualBlock(64, 128, conditional=True, condition_dim=1) #128,256
            self.res2_2 = ConditionalResidualBlock(128, 128, conditional=True, condition_dim=1) #256,256
        else:
            self.res2_1 = ResidualBlock(64, 128) #128,256
            self.res2_2 = ResidualBlock(128, 128) #256,256
        
        # Downsample 2: H/4, W/4
        self.down2 = nn.Conv3d(128, 128, kernel_size=self.kernel_size, stride=(1, 2, 2), padding=0) #256,256
        
        # Stage 3: 512 channels
        if self.conditional:
            self.res3_1 = ConditionalResidualBlock(128, 256, conditional=True, condition_dim=1) #256,512
            self.res3_2 = ConditionalResidualBlock(256, 256, conditional=True, condition_dim=1) #512,512
        else:
            self.res3_1 = ResidualBlock(128, 256) #256,512
            self.res3_2 = ResidualBlock(256, 256) #512,512
        
        # Final layers
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=256) #512
        self.conv_out = nn.Conv3d(256, 2*self.out_channels, kernel_size=self.kernel_size, padding=padding)
        
        # Final FiLM on latent (optional, but can help differentiate latent space)
        if self.conditional:
            self.film_out = FiLM(condition_dim=1, feature_channels=2*self.out_channels)

        print(f'Trainable parameters: {self.trainable_params}.')

    def _pad_for_stride(self, x: torch.Tensor) -> torch.Tensor:
        """Apply asymmetric padding for stride-2 downsampling."""
        # Depth padding: 1 on each side to maintain depth dimension with kernel_size=3
        # H/W padding: asymmetric (0,1) for stride-2 downsampling
        left, right, top, bottom, front, back = 0, 1, 0, 1, 1, 1
        return F.pad(x, (left, right, top, bottom, front, back))

    def forward(
        self,
        x: torch.Tensor,  # (B, in_channels, D, H, W)
        condition: torch.Tensor = None  # (B,) boolean tensor: True=3D flow, False=2D flow
    ):
        """
        Forward pass through encoder with FiLM conditioning at multiple layers.
        
        Args:
            x: Input velocity field (B, in_channels, D, H, W)
            condition: Optional boolean tensor (B,) where True=3D flow (U), False=2D flow (U_2d)
                       Only used if self.conditional=True
        """
        # Initial conv
        x = self.conv_in(x)
        if self.conditional and condition is not None:
            x = self.film_in(x, condition)
        
        # Stage 1
        if self.conditional and condition is not None:
            x = self.res1_1(x, condition)
            x = self.res1_2(x, condition)
        else:
            x = self.res1_1(x)
            x = self.res1_2(x)
        
        # Downsample 1
        x = self._pad_for_stride(x)
        x = self.down1(x)
        
        # Stage 2
        if self.conditional and condition is not None:
            x = self.res2_1(x, condition)
            x = self.res2_2(x, condition)
        else:
            x = self.res2_1(x)
            x = self.res2_2(x)
        
        # Downsample 2
        x = self._pad_for_stride(x)
        x = self.down2(x)
        
        # Stage 3
        if self.conditional and condition is not None:
            x = self.res3_1(x, condition)
            x = self.res3_2(x, condition)
        else:
            x = self.res3_1(x)
            x = self.res3_2(x)
        
        # Final layers
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        
        # Apply final FiLM to latent space
        if self.conditional and condition is not None:
            x = self.film_out(x, condition)
        
        # from (B, C, d, h, w) -> 2 tensors of shape (B, C/2, d, h, w)
        mu, log_var = torch.chunk(x, 2, dim=1)

        return mu, log_var

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5*logvar)
        # set random seed for reproducibility
        # torch.manual_seed(0)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    @property
    def trainable_params(self):
        total = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )
        return total



