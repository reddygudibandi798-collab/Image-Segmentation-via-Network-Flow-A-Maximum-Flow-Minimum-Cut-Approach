"""
Generate segmentation mask visualization for LaTeX paper
Creates a 3-panel figure showing: Input with seeds, Binary mask, Extracted foreground
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from image_segmentation import ImageSegmentation, create_synthetic_image
import os


def generate_mask_visualization(output_file='images/segmented_mask.png', size=80):
    """
    Generate a 3-panel visualization showing:
    - Left: Input image with foreground (blue) and background (red) seeds
    - Center: Binary segmentation mask (white=foreground, black=background)
    - Right: Extracted foreground by applying mask to original image
    
    Args:
        output_file: Path to save the output image
        size: Size of the square image (default 80x80)
    """
    
    # Ensure output directory exists
    os.makedirs('images', exist_ok=True)
    
    print("=" * 70)
    print("Generating Segmentation Mask Visualization")
    print("=" * 70)
    
    # Create synthetic image with gradient circle
    print(f"\n1. Creating {size}x{size} synthetic image with gradient circle...")
    image = create_synthetic_image(size, pattern='circle')
    
    # Define seeds
    center = size // 2
    
    # Foreground seeds: Conservative placement in the center region
    print("2. Placing foreground seeds in center region...")
    foreground_seeds = []
    seed_radius = int(size * 0.15)  # Conservative 15% radius
    for i in range(center - seed_radius, center + seed_radius + 1, 3):
        for j in range(center - seed_radius, center + seed_radius + 1, 3):
            if 0 <= i < size and 0 <= j < size:
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if dist <= seed_radius:
                    foreground_seeds.append((i, j))
    
    # Background seeds: Place around image edges
    print("3. Placing background seeds around image edges...")
    background_seeds = []
    margin = 8
    step = max(8, size // 10)
    
    # Top and bottom edges
    for x in range(0, size, step):
        if x < size:
            background_seeds.append((margin, x))
            background_seeds.append((size - margin - 1, x))
    
    # Left and right edges
    for y in range(0, size, step):
        if y < size:
            background_seeds.append((y, margin))
            background_seeds.append((y, size - margin - 1))
    
    print(f"   - Foreground seeds: {len(foreground_seeds)}")
    print(f"   - Background seeds: {len(background_seeds)}")
    
    # Perform segmentation
    print("4. Running graph cuts segmentation...")
    segmenter = ImageSegmentation(image, beta=20.0)
    mask, flow_value, comp_time = segmenter.segment(foreground_seeds, background_seeds)
    
    fg_pixels = np.sum(mask == 1)
    bg_pixels = np.sum(mask == 0)
    print(f"   - Computation time: {comp_time:.3f} seconds")
    print(f"   - Min-cut value: {flow_value:.2f}")
    print(f"   - Foreground pixels: {fg_pixels}")
    print(f"   - Background pixels: {bg_pixels}")
    
    # Create 3-panel figure
    print("5. Creating 3-panel visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # ===== Panel 1: Input image with seeds =====
    axes[0].imshow(image, cmap='gray', vmin=0, vmax=255)
    
    # Plot foreground seeds (blue)
    fg_coords = np.array(foreground_seeds)
    axes[0].scatter(fg_coords[:, 1], fg_coords[:, 0], 
                   c='blue', s=25, marker='o', 
                   label='Foreground Seeds', 
                   edgecolors='white', linewidths=0.5, alpha=0.8)
    
    # Plot background seeds (red)
    bg_coords = np.array(background_seeds)
    axes[0].scatter(bg_coords[:, 1], bg_coords[:, 0], 
                   c='red', s=25, marker='s', 
                   label='Background Seeds',
                   edgecolors='white', linewidths=0.5, alpha=0.8)
    
    axes[0].set_title('Input with Seeds', fontsize=14, fontweight='bold', pad=10)
    axes[0].legend(loc='upper right', fontsize=9, framealpha=0.9)
    axes[0].axis('off')
    
    # ===== Panel 2: Binary segmentation mask =====
    # White = foreground (M=1), Black = background (M=0)
    axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Binary Mask M\n(White=Foreground, Black=Background)', 
                     fontsize=14, fontweight='bold', pad=10)
    axes[1].axis('off')
    
    # ===== Panel 3: Extracted foreground =====
    # Apply mask to original image
    foreground_extracted = np.zeros_like(image)
    foreground_extracted[mask == 1] = image[mask == 1]
    
    axes[2].imshow(foreground_extracted, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('Extracted Foreground\n(Mask Applied to Image)', 
                     fontsize=14, fontweight='bold', pad=10)
    axes[2].axis('off')
    
    # Add overall title
    fig.suptitle('Image Segmentation via Graph Cuts: Mask Visualization', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Visualization saved to: {output_file}")
    plt.close()
    
    # Also create individual components for flexibility
    print("\n6. Generating individual component images...")
    
    # Save just the binary mask
    mask_file = 'images/binary_mask_only.png'
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray', vmin=0, vmax=1)
    plt.title('Binary Segmentation Mask M', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(mask_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   - Binary mask: {mask_file}")
    
    # Save just the extracted foreground
    fg_file = 'images/extracted_foreground_only.png'
    plt.figure(figsize=(6, 6))
    plt.imshow(foreground_extracted, cmap='gray', vmin=0, vmax=255)
    plt.title('Extracted Foreground', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(fg_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   - Extracted foreground: {fg_file}")
    
    # Save input with seeds
    input_file = 'images/input_with_seeds.png'
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.scatter(fg_coords[:, 1], fg_coords[:, 0], 
               c='blue', s=30, marker='o', 
               label='Foreground Seeds', 
               edgecolors='white', linewidths=0.8, alpha=0.9)
    plt.scatter(bg_coords[:, 1], bg_coords[:, 0], 
               c='red', s=30, marker='s', 
               label='Background Seeds',
               edgecolors='white', linewidths=0.8, alpha=0.9)
    plt.title('Input Image with User-Defined Seeds', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10, framealpha=0.95)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(input_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   - Input with seeds: {input_file}")
    
    print("\n" + "=" * 70)
    print("Visualization Generation Complete!")
    print("=" * 70)
    print("\nGenerated files for LaTeX:")
    print(f"  Main figure: {output_file}")
    print(f"  Components:  {mask_file}, {fg_file}, {input_file}")
    print("\nYou can now use these in your LaTeX document with:")
    print(f"  \\includegraphics[width=0.46\\textwidth]{{{output_file}}}")
    
    return {
        'main_figure': output_file,
        'mask_only': mask_file,
        'foreground_only': fg_file,
        'input_with_seeds': input_file,
        'computation_time': comp_time,
        'flow_value': flow_value,
        'foreground_pixels': fg_pixels,
        'background_pixels': bg_pixels
    }


def generate_comparison_sizes():
    """Generate mask visualizations at different image sizes for comparison."""
    print("\n" + "=" * 70)
    print("Generating Multi-Size Comparison")
    print("=" * 70)
    
    sizes = [50, 80, 100]
    results = []
    
    for size in sizes:
        print(f"\nProcessing {size}x{size} image...")
        result = generate_mask_visualization(
            output_file=f'images/segmented_mask_{size}x{size}.png',
            size=size
        )
        results.append((size, result))
    
    # Print summary table
    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print(f"\n{'Size':>6} {'Time(s)':>10} {'Min-Cut':>12} {'FG Pixels':>12} {'BG Pixels':>12}")
    print("-" * 70)
    for size, result in results:
        print(f"{size:>6} {result['computation_time']:>10.3f} "
              f"{result['flow_value']:>12.2f} "
              f"{result['foreground_pixels']:>12} "
              f"{result['background_pixels']:>12}")


def main():
    """Main function to generate all visualizations."""
    import sys
    
    print("\n" + "=" * 70)
    print("Segmentation Mask Visualization Generator")
    print("For LaTeX Paper: Complete Algorithm Section")
    print("=" * 70)
    
    # Check if size argument provided
    size = 80  # default
    if len(sys.argv) > 1:
        try:
            size = int(sys.argv[1])
            print(f"\nUsing custom image size: {size}x{size}")
        except ValueError:
            print(f"\nInvalid size argument, using default: {size}x{size}")
    
    # Generate main visualization
    result = generate_mask_visualization(size=size)
    
    # Ask if user wants multi-size comparison
    print("\n" + "-" * 70)
    generate_more = input("\nGenerate additional sizes (50, 80, 100)? [y/N]: ").strip().lower()
    if generate_more == 'y':
        generate_comparison_sizes()
    
    print("\n✓ All visualizations generated successfully!")
    print("\nNext steps:")
    print("  1. Check the 'images/' folder for generated PNG files")
    print("  2. Add to LaTeX with: \\includegraphics[width=0.46\\textwidth]{images/segmented_mask.png}")
    print("  3. Compile your LaTeX document to see the results")


if __name__ == "__main__":
    main()
