"""
ComfyUI Custom Node for Structured Noise Generation

This module provides ComfyUI nodes that wrap the structured_noise package
for phase-preserving diffusion.
"""

import sys
import os

# Try to import from installed package first (if user did pip install)
try:
    from structured_noise.structured_noise_pytorch import (
        generate_structured_noise_batch_vectorized,
    )
except ImportError:
    # Fallback: import from subdirectory (if repo is cloned into custom_nodes)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.join(current_dir, 'structured_noise')
    if os.path.exists(package_dir):
        sys.path.insert(0, current_dir)
        from structured_noise.structured_noise_pytorch import (
            generate_structured_noise_batch_vectorized,
        )
    else:
        raise ImportError(
            "Could not find structured_noise package. "
            "Please install it with: pip install structured-noise"
        )

import torch
import torch.nn.functional as F
from typing_extensions import override
from comfy.model_management import get_torch_device
from comfy_api.latest import io, ComfyExtension


class Noise_StructuredNoise:
    """
    Noise object that generates structured noise based on a latent.
    When generate_noise is called, it generates structured noise matching the target latent dimensions.
    """
    def __init__(self, latent_tensor, noise_std, cutoff_radius, transition_width, 
                 pad_factor, sampling_method, seed, input_noise_tensor=None):
        """
        Initialize structured noise generator.
        
        Args:
            latent_tensor: Latent tensor in (B, C, H, W) format
            noise_std: Standard deviation for Gaussian noise
            cutoff_radius: Frequency cutoff radius (None = full structure)
            transition_width: Width of smooth transition
            pad_factor: Padding factor
            sampling_method: Sampling method ('fft', 'cdf', 'two-gaussian')
            seed: Random seed (required by samplers, though structured noise is deterministic based on input)
            input_noise_tensor: Optional input noise tensor
        """
        self.latent_tensor = latent_tensor
        self.noise_std = noise_std
        self.cutoff_radius = cutoff_radius
        self.transition_width = transition_width
        self.pad_factor = pad_factor
        self.sampling_method = sampling_method
        self.seed = seed
        self.input_noise_tensor = input_noise_tensor
        self.device = latent_tensor.device
        
        # Generate structured noise from the reference latent
        self.structured_noise = generate_structured_noise_batch_vectorized(
            image_batch=latent_tensor,
            noise_std=noise_std,
            pad_factor=pad_factor,
            cutoff_radius=cutoff_radius,
            transition_width=transition_width,
            input_noise=input_noise_tensor,
            sampling_method=sampling_method
        )
    
    def generate_noise(self, input_latent):
        """
        Generate noise matching the input latent dimensions.
        
        Args:
            input_latent: Dict with 'samples' key containing latent tensor (B, C, H, W)
        
        Returns:
            Noise tensor matching latent dimensions
        """
        target_latent = input_latent["samples"]
        target_shape = target_latent.shape  # (B, C_target, H_target, W_target)
        
        # Structured noise was generated from reference latent (B, C_ref, H_ref, W_ref)
        batch_size_ref, channels_ref, height_ref, width_ref = self.structured_noise.shape
        batch_size_target, channels_target, height_target, width_target = target_shape
        
        # Handle batch size
        if batch_size_target <= batch_size_ref:
            noise_batch = self.structured_noise[:batch_size_target]
        else:
            # Repeat noise if needed
            noise_batch = self.structured_noise.repeat(
                (batch_size_target // batch_size_ref + 1, 1, 1, 1)
            )[:batch_size_target]
        
        # Resize to match target dimensions if needed
        if (height_target, width_target) != (height_ref, width_ref):
            noise_resized = F.interpolate(
                noise_batch,
                size=(height_target, width_target),
                mode='bilinear',
                align_corners=False
            )
        else:
            noise_resized = noise_batch
        
        # Handle channel mismatch
        if channels_target != channels_ref:
            if channels_target > channels_ref:
                # Replicate channels to match
                mult = (channels_target + channels_ref - 1) // channels_ref
                noise_output = noise_resized.repeat(1, mult, 1, 1)[:, :channels_target, :, :]
            else:
                # Take first N channels
                noise_output = noise_resized[:, :channels_target, :, :]
        else:
            noise_output = noise_resized
        
        # Ensure noise is on the same device and dtype as target latent
        noise_output = noise_output.to(target_latent.device).to(target_latent.dtype)
        
        return noise_output


class StructuredNoiseNode(io.ComfyNode):
    """
    ComfyUI node for generating structured noise.
    
    This node generates structured noise by combining Gaussian noise magnitude
    with image phase using frequency soft cutoff, as described in the paper.
    """
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StructuredNoise",
            category="sampling/custom_sampling/noise",
            inputs=[
                io.Latent.Input("latent"),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xffffffffffffffff,
                    control_after_generate=True
                ),
                io.Float.Input(
                    "noise_std",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.1,
                    display_mode=io.NumberDisplay.number
                ),
                io.Float.Input(
                    "cutoff_radius",
                    default=40.0,
                    min=0.0,
                    max=1000.0,
                    step=1.0,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Frequency cutoff radius (0 = full structure preservation)"
                ),
                io.Float.Input(
                    "transition_width",
                    default=2.0,
                    min=0.1,
                    max=20.0,
                    step=0.1,
                    display_mode=io.NumberDisplay.number
                ),
                io.Float.Input(
                    "pad_factor",
                    default=1.5,
                    min=1.0,
                    max=3.0,
                    step=0.1,
                    display_mode=io.NumberDisplay.number
                ),
                io.Combo.Input(
                    "sampling_method",
                    options=["fft", "cdf", "two-gaussian"],
                    default="fft"
                ),
                io.Latent.Input("input_noise", optional=True),
            ],
            outputs=[io.Noise.Output()],
        )
    
    @classmethod
    def execute(cls, latent, seed, noise_std, cutoff_radius, transition_width, pad_factor, 
                sampling_method, input_noise=None) -> io.NodeOutput:
        """
        Generate structured noise object.
        
        Args:
            latent: Latent dict with 'samples' key containing tensor (B, C, H, W)
            seed: Random seed (required by samplers)
            noise_std: Standard deviation for Gaussian noise
            cutoff_radius: Frequency cutoff radius (0 = full structure preservation)
            transition_width: Width of smooth transition for frequency cutoff
            pad_factor: Padding factor to reduce boundary artifacts
            sampling_method: Method to sample noise magnitude ('fft', 'cdf', 'two-gaussian')
            input_noise: Optional latent dict with 'samples' key containing noise tensor (B, C, H, W)
        
        Returns:
            Noise object that can be used with samplers
        """
        device = get_torch_device()
        
        # Extract latent tensor from dict (already in B, C, H, W format)
        latent_tensor = latent["samples"].to(device)
        
        # Handle optional input_noise
        input_noise_tensor = None
        if input_noise is not None:
            input_noise_tensor = input_noise["samples"].to(device)
        
        # Create noise object
        noise_obj = Noise_StructuredNoise(
            latent_tensor=latent_tensor,
            noise_std=noise_std,
            cutoff_radius=cutoff_radius,
            transition_width=transition_width,
            pad_factor=pad_factor,
            sampling_method=sampling_method,
            seed=seed,
            input_noise_tensor=input_noise_tensor
        )
        
        return io.NodeOutput(noise_obj)


# V3 Extension Definition (new API)
class StructuredNoiseExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [
            StructuredNoiseNode,
        ]


async def comfy_entrypoint() -> StructuredNoiseExtension:
    """ComfyUI entry point for loading the extension."""
    return StructuredNoiseExtension()

