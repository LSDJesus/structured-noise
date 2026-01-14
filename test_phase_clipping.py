#!/usr/bin/env python3
"""
Test to determine if phase clipping is helping or hurting structure preservation.

Theory: Phase encodes structure. If clipping phases damages structural info,
we should see LESS alignment between the structured noise and the input image.

Test methodology:
1. Generate structured noise WITH phase clipping (current behavior)
2. Generate structured noise WITHOUT phase clipping
3. Compare edge/structure preservation using edge correlation
4. Visual comparison with edge overlay
"""

import torch
import numpy as np
from PIL import Image
import os

# We'll modify the function inline to test both behaviors
from structured_noise.structured_noise_pytorch import (
    create_frequency_soft_cutoff_mask,
    clip_frequency_magnitude,
)


def generate_structured_noise_test(
    image_batch: torch.Tensor,
    clip_phase: bool = True,  # Toggle to test
    noise_std: float = 1.0,
    pad_factor: float = 1.5,
    cutoff_radius: float = None,
    transition_width: float = 2.0,
    seed: int = 42,
) -> torch.Tensor:
    """Modified version to test phase clipping effect."""
    
    torch.manual_seed(seed)  # Fixed seed for fair comparison
    
    batch_size, channels, height, width = image_batch.shape
    dtype = image_batch.dtype
    device = image_batch.device
    image_batch = image_batch.float()
    
    pad_h = int(height * (pad_factor - 1))
    pad_h = pad_h // 2 * 2
    pad_w = int(width * (pad_factor - 1))
    pad_w = pad_w // 2 * 2
    
    padded_images = torch.nn.functional.pad(
        image_batch, 
        (pad_w//2, pad_w//2, pad_h//2, pad_h//2), 
        mode='reflect'
    )
    
    padded_height = height + pad_h
    padded_width = width + pad_w
    
    if cutoff_radius is not None:
        cutoff_radius = min(min(padded_height/2, padded_width/2), cutoff_radius)
        freq_mask = create_frequency_soft_cutoff_mask(
            padded_height, padded_width, cutoff_radius, transition_width, device
        )
    else:
        freq_mask = torch.ones(padded_height, padded_width, device=device)
    
    fft = torch.fft.fft2(padded_images, dim=(-2, -1))
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    
    image_phases = torch.angle(fft_shifted)
    
    # THE KEY DIFFERENCE: To clip or not to clip?
    if clip_phase:
        image_phases = clip_frequency_magnitude(image_phases)  # Current behavior
    # else: leave phases untouched
    
    noise_batch = torch.randn_like(padded_images)
    noise_fft = torch.fft.fft2(noise_batch, dim=(-2, -1))
    noise_fft_shifted = torch.fft.fftshift(noise_fft, dim=(-2, -1))
    
    noise_magnitudes = torch.abs(noise_fft_shifted)
    noise_phases = torch.angle(noise_fft_shifted)
    
    noise_magnitudes = clip_frequency_magnitude(noise_magnitudes)
    noise_magnitudes = noise_magnitudes * noise_std
    
    mixed_phases = freq_mask.unsqueeze(0).unsqueeze(0) * image_phases + \
                   (1 - freq_mask.unsqueeze(0).unsqueeze(0)) * noise_phases
    
    fft_combined = noise_magnitudes * torch.exp(1j * mixed_phases)
    fft_unshifted = torch.fft.ifftshift(fft_combined, dim=(-2, -1))
    structured_noise_padded = torch.fft.ifft2(fft_unshifted, dim=(-2, -1))
    structured_noise_padded = torch.real(structured_noise_padded)
    
    clamp_mask = (structured_noise_padded < -5) + (structured_noise_padded > 5)
    clamp_mask = (clamp_mask > 0).float()
    structured_noise_padded = structured_noise_padded * (1 - clamp_mask) + noise_batch * clamp_mask
    
    structured_noise_batch = structured_noise_padded[
        :, :, 
        pad_h//2:pad_h//2 + height, 
        pad_w//2:pad_w//2 + width
    ]
    return structured_noise_batch.to(dtype)


def sobel_edges(tensor: torch.Tensor) -> torch.Tensor:
    """Extract edges using Sobel operator."""
    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=tensor.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=tensor.device)
    
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)
    
    # Convert to grayscale if needed
    if tensor.shape[1] == 3:
        gray = 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]
    else:
        gray = tensor
    
    # Apply Sobel
    edges_x = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
    edges_y = torch.nn.functional.conv2d(gray, sobel_y, padding=1)
    edges = torch.sqrt(edges_x**2 + edges_y**2)
    
    return edges


def edge_correlation(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Measure correlation between edge maps (higher = more similar structure)."""
    edges1 = sobel_edges(img1).flatten()
    edges2 = sobel_edges(img2).flatten()
    
    # Normalize
    edges1 = edges1 - edges1.mean()
    edges2 = edges2 - edges2.mean()
    
    # Pearson correlation
    corr = (edges1 * edges2).sum() / (edges1.norm() * edges2.norm() + 1e-8)
    return corr.item()


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image for visualization."""
    # Normalize to 0-1 range
    t = tensor[0].permute(1, 2, 0).cpu().numpy()
    t = (t - t.min()) / (t.max() - t.min() + 1e-8)
    t = (t * 255).astype(np.uint8)
    return Image.fromarray(t)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='dog.jpg', help='Input image')
    parser.add_argument('--cutoff', type=float, default=40.0, help='Frequency cutoff radius')
    parser.add_argument('--output_dir', type=str, default='phase_clip_test', help='Output directory')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load image
    image = Image.open(args.input).convert('RGB')
    image = np.array(image) / 255.0
    image = (image - 0.5) * 2.0  # Normalize to [-1, 1]
    image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).to(device)
    
    print(f"Image shape: {image.shape}")
    print(f"Cutoff radius: {args.cutoff}")
    
    # Generate with phase clipping (current behavior)
    print("\nGenerating WITH phase clipping...")
    noise_with_clip = generate_structured_noise_test(
        image, clip_phase=True, cutoff_radius=args.cutoff, seed=42
    )
    
    # Generate WITHOUT phase clipping
    print("Generating WITHOUT phase clipping...")
    noise_without_clip = generate_structured_noise_test(
        image, clip_phase=False, cutoff_radius=args.cutoff, seed=42
    )
    
    # Generate pure Gaussian noise for baseline
    torch.manual_seed(42)
    gaussian_noise = torch.randn_like(image)
    
    # Measure edge correlation with original image
    corr_with_clip = edge_correlation(image, noise_with_clip)
    corr_without_clip = edge_correlation(image, noise_without_clip)
    corr_gaussian = edge_correlation(image, gaussian_noise)
    
    print("\n" + "="*60)
    print("EDGE CORRELATION WITH ORIGINAL IMAGE (higher = better structure)")
    print("="*60)
    print(f"  Gaussian noise (baseline):     {corr_gaussian:.4f}")
    print(f"  WITH phase clipping:           {corr_with_clip:.4f}")
    print(f"  WITHOUT phase clipping:        {corr_without_clip:.4f}")
    print("="*60)
    
    diff = corr_without_clip - corr_with_clip
    if abs(diff) > 0.01:
        if diff > 0:
            print(f"\n✓ WITHOUT phase clipping preserves {diff:.4f} MORE structure!")
            print("  → Phase clipping appears to be HURTING structure preservation.")
        else:
            print(f"\n✓ WITH phase clipping preserves {-diff:.4f} MORE structure!")
            print("  → Phase clipping appears to be HELPING (unexpected but possible).")
    else:
        print(f"\n≈ Difference is negligible ({diff:.4f})")
        print("  → Phase clipping may not matter much in practice.")
    
    # Save visual comparison
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save images
    tensor_to_image(image).save(f"{args.output_dir}/1_original.png")
    tensor_to_image(noise_with_clip).save(f"{args.output_dir}/2_with_phase_clip.png")
    tensor_to_image(noise_without_clip).save(f"{args.output_dir}/3_without_phase_clip.png")
    tensor_to_image(gaussian_noise).save(f"{args.output_dir}/4_gaussian_baseline.png")
    
    # Save difference image (amplified)
    diff_tensor = (noise_without_clip - noise_with_clip) * 5  # Amplify difference
    tensor_to_image(diff_tensor).save(f"{args.output_dir}/5_difference_amplified.png")
    
    # Save edge maps
    edges_orig = sobel_edges(image)
    edges_with = sobel_edges(noise_with_clip)
    edges_without = sobel_edges(noise_without_clip)
    
    def edge_to_image(e):
        e = e[0, 0].cpu().numpy()
        e = (e - e.min()) / (e.max() - e.min() + 1e-8)
        return Image.fromarray((e * 255).astype(np.uint8))
    
    edge_to_image(edges_orig).save(f"{args.output_dir}/6_edges_original.png")
    edge_to_image(edges_with).save(f"{args.output_dir}/7_edges_with_clip.png")
    edge_to_image(edges_without).save(f"{args.output_dir}/8_edges_without_clip.png")
    
    print(f"\nImages saved to {args.output_dir}/")
    print("Compare edge maps to see if structure alignment differs!")


if __name__ == "__main__":
    main()
