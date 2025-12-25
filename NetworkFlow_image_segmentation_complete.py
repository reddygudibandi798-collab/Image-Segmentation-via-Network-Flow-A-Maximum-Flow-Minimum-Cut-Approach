"""
Image Segmentation via Maximum Flow - Complete Implementation
Combines max flow network, image segmentation, and CLI interface in one file.

Usage:
    python image_segmentation_complete.py [image_path] [beta]
    
Examples:
    python image_segmentation_complete.py                    # Demo with synthetic image
    python image_segmentation_complete.py photo.jpg          # Segment photo.jpg
    python image_segmentation_complete.py photo.jpg 90       # With custom beta parameter
"""

import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL/Pillow not available. Install with: pip install Pillow")


# ============================================================================
# MAXIMUM FLOW IMPLEMENTATION (Edmonds-Karp Algorithm)
# ============================================================================

class FlowNetwork:
    def __init__(self):
        """Initialize an empty flow network."""
        self.graph = defaultdict(dict)
        self.vertices = set()
    
    def add_vertex(self, v):
        """Add a vertex to the network.
        
        Args:
            v: Vertex identifier
        """
        self.vertices.add(v)
    
    def add_edge(self, u, v, capacity):
        """Add a directed edge with given capacity.
        
        Args:
            u: Source vertex
            v: Destination vertex
            capacity: Edge capacity (non-negative)
        """
        self.vertices.add(u)
        self.vertices.add(v)
        
        # Add forward edge
        if v not in self.graph[u]:
            self.graph[u][v] = 0
        self.graph[u][v] += capacity
        
        # Ensure reverse edge exists (for residual graph)
        if u not in self.graph[v]:
            self.graph[v][u] = 0
    
    def get_neighbors(self, u):
        """Get all neighbors of vertex u.
        
        Args:
            u: Vertex identifier
            
        Returns:
            List of neighboring vertices
        """
        return list(self.graph[u].keys())
    
    def get_capacity(self, u, v):
        """Get capacity of edge (u,v).
        
        Args:
            u: Source vertex
            v: Destination vertex
            
        Returns:
            Capacity of edge, 0 if edge doesn't exist
        """
        return self.graph[u].get(v, 0)


class EdmondsKarp:
    """
    Edmonds-Karp algorithm for computing maximum flow.
    Uses BFS to find augmenting paths.
    Time Complexity: O(V * E^2)
    """
    
    def __init__(self, network: FlowNetwork):
        """Initialize with a flow network.
        
        Args:
            network: FlowNetwork instance
        """
        self.network = network
        self.flow = defaultdict(lambda: defaultdict(int))
    
    def bfs(self, source, sink) -> Tuple[bool, Dict]:
        """Find augmenting path using BFS.
        
        Args:
            source: Source vertex
            sink: Sink vertex
            
        Returns:
            Tuple of (path_exists, parent_dict)
        """
        visited = {source}
        queue = deque([source])
        parent = {source: None}
        
        while queue:
            u = queue.popleft()
            
            if u == sink:
                return True, parent
            
            for v in self.network.get_neighbors(u):
                # Check if there's available capacity in residual graph
                residual_capacity = (
                    self.network.get_capacity(u, v) - self.flow[u][v]
                )
                
                if v not in visited and residual_capacity > 0:
                    visited.add(v)
                    parent[v] = u
                    queue.append(v)
        
        return False, parent
    
    def find_min_capacity(self, source, sink, parent) -> float:
        """Find minimum residual capacity along the augmenting path.
        
        Args:
            source: Source vertex
            sink: Sink vertex
            parent: Parent dictionary from BFS
            
        Returns:
            Minimum capacity along the path
        """
        path_capacity = float('inf')
        v = sink
        
        while v != source:
            u = parent[v]
            residual_capacity = (
                self.network.get_capacity(u, v) - self.flow[u][v]
            )
            path_capacity = min(path_capacity, residual_capacity)
            v = u
        
        return path_capacity
    
    def augment_path(self, source, sink, parent, path_capacity):
        """Augment flow along the path found by BFS.
        
        Args:
            source: Source vertex
            sink: Sink vertex
            parent: Parent dictionary from BFS
            path_capacity: Amount to augment
        """
        v = sink
        
        while v != source:
            u = parent[v]
            self.flow[u][v] += path_capacity
            self.flow[v][u] -= path_capacity
            v = u
    
    def max_flow(self, source, sink) -> float:
        """Compute maximum flow from source to sink.
        
        Args:
            source: Source vertex
            sink: Sink vertex
            
        Returns:
            Maximum flow value
        """
        max_flow_value = 0
        
        # Find augmenting paths until none exist
        while True:
            path_exists, parent = self.bfs(source, sink)
            
            if not path_exists:
                break
            
            # Find minimum capacity along the path
            path_capacity = self.find_min_capacity(source, sink, parent)
            
            # Augment flow along the path
            self.augment_path(source, sink, parent, path_capacity)
            
            max_flow_value += path_capacity
        
        return max_flow_value
    
    def get_min_cut(self, source) -> Tuple[Set, Set]:
        """Get minimum cut (S, T) after computing max flow.
        
        Args:
            source: Source vertex
            
        Returns:
            Tuple of (source_side, sink_side) vertex sets
        """
        # Find all vertices reachable from source in residual graph
        visited = {source}
        queue = deque([source])
        
        while queue:
            u = queue.popleft()
            
            for v in self.network.get_neighbors(u):
                residual_capacity = (
                    self.network.get_capacity(u, v) - self.flow[u][v]
                )
                
                if v not in visited and residual_capacity > 0:
                    visited.add(v)
                    queue.append(v)
        
        source_side = visited
        sink_side = self.network.vertices - visited
        
        return source_side, sink_side


# ============================================================================
# IMAGE SEGMENTATION IMPLEMENTATION
# ============================================================================

class ImageSegmentation:
    """Image segmentation via graph cuts and maximum flow."""
    
    def __init__(self, image: np.ndarray, beta: float = 20.0):
        # Handle RGBA images by removing alpha channel
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        
        if len(image.shape) == 3:
            # Convert RGB to grayscale
            self.image = np.mean(image, axis=2)
        else:
            self.image = image.copy()
        
        self.height, self.width = self.image.shape
        self.beta = beta
        self.network = None
        self.source = 'source'
        self.sink = 'sink'
    
    def pixel_to_node(self, i: int, j: int) -> str:
        """Convert pixel coordinates to node identifier."""
        return f"p_{i}_{j}"
    
    def node_to_pixel(self, node: str) -> Tuple[int, int]:
        """Convert node identifier to pixel coordinates."""
        parts = node.split('_')
        return int(parts[1]), int(parts[2])
    
    def compute_data_term(
        self,
        i: int, 
        j: int, 
        foreground_seeds: Set[Tuple[int, int]],
        background_seeds: Set[Tuple[int, int]],
        foreground_mean: float,
        background_mean: float
    ) -> Tuple[float, float]:
        """Compute data term (unary cost) for pixel (i,j)."""
        pixel_intensity = self.image[i, j]
        
        # Hard constraints for seeds
        if (i, j) in foreground_seeds:
            return 0, float('inf')
        if (i, j) in background_seeds:
            return float('inf'), 0
        
        # Data term based on intensity difference
        diff_fg = abs(pixel_intensity - foreground_mean)
        diff_bg = abs(pixel_intensity - background_mean)
        
        # Compute minimum distance to seed regions
        min_dist_fg = float('inf')
        min_dist_bg = float('inf')
        
        # Sample seeds for distance computation
        fg_sample = list(foreground_seeds) if len(foreground_seeds) < 50 else list(foreground_seeds)[::max(1, len(foreground_seeds)//50)]
        bg_sample = list(background_seeds) if len(background_seeds) < 50 else list(background_seeds)[::max(1, len(background_seeds)//50)]
        
        for si, sj in fg_sample:
            dist = np.sqrt((i - si)**2 + (j - sj)**2)
            min_dist_fg = min(min_dist_fg, dist)
        
        for si, sj in bg_sample:
            dist = np.sqrt((i - si)**2 + (j - sj)**2)
            min_dist_bg = min(min_dist_bg, dist)
        
        # Combined cost: intensity difference + spatial distance
        intensity_weight = 10.0
        spatial_decay = 0.15
        spatial_penalty_fg = np.exp(spatial_decay * min_dist_fg)
        spatial_penalty_bg = np.exp(spatial_decay * min_dist_bg)
        
        cost_foreground = intensity_weight * (diff_fg ** 2) * spatial_penalty_fg
        cost_background = intensity_weight * (diff_bg ** 2) * spatial_penalty_bg
        
        return cost_foreground, cost_background
    
    def compute_boundary_penalty(self, i1: int, j1: int, i2: int, j2: int) -> float:
        """Compute boundary penalty (pairwise cost) between neighboring pixels."""
        intensity_diff = float(self.image[i1, j1]) - float(self.image[i2, j2])
        normalized_diff = intensity_diff / 255.0
        penalty = np.exp(-self.beta * (normalized_diff ** 2))
        return penalty
    
    def build_network(
        self,
        foreground_seeds: Set[Tuple[int, int]],
        background_seeds: Set[Tuple[int, int]]
    ) -> FlowNetwork:
        """Build flow network from image and seeds."""
        network = FlowNetwork()
        
        # Add source and sink
        network.add_vertex(self.source)
        network.add_vertex(self.sink)
        
        # Compute mean intensities for seeds
        if foreground_seeds:
            fg_intensities = [self.image[i, j] for i, j in foreground_seeds]
            foreground_mean = np.mean(fg_intensities)
        else:
            foreground_mean = np.max(self.image)
        
        if background_seeds:
            bg_intensities = [self.image[i, j] for i, j in background_seeds]
            background_mean = np.mean(bg_intensities)
        else:
            background_mean = np.min(self.image)
        
        print(f"Foreground mean intensity: {foreground_mean:.2f}")
        print(f"Background mean intensity: {background_mean:.2f}")
        
        # Add terminal edges (source/sink to pixels)
        for i in range(self.height):
            for j in range(self.width):
                node = self.pixel_to_node(i, j)
                
                cost_fg, cost_bg = self.compute_data_term(
                    i, j, foreground_seeds, background_seeds,
                    foreground_mean, background_mean
                )
                
                # Edge from source to pixel (capacity = cost of being background)
                network.add_edge(self.source, node, cost_bg)
                
                # Edge from pixel to sink (capacity = cost of being foreground)
                network.add_edge(node, self.sink, cost_fg)
        
        # Add neighbor edges (4-connectivity)
        for i in range(self.height):
            for j in range(self.width):
                node1 = self.pixel_to_node(i, j)
                
                # Right neighbor
                if j + 1 < self.width:
                    node2 = self.pixel_to_node(i, j + 1)
                    penalty = self.compute_boundary_penalty(i, j, i, j + 1)
                    network.add_edge(node1, node2, penalty)
                    network.add_edge(node2, node1, penalty)
                
                # Bottom neighbor
                if i + 1 < self.height:
                    node2 = self.pixel_to_node(i + 1, j)
                    penalty = self.compute_boundary_penalty(i, j, i + 1, j)
                    network.add_edge(node1, node2, penalty)
                    network.add_edge(node2, node1, penalty)
        
        return network
    
    def segment(
        self,
        foreground_seeds: List[Tuple[int, int]],
        background_seeds: List[Tuple[int, int]]
    ) -> Tuple[np.ndarray, float, float]:
        """Perform image segmentation.
        
        Returns:
            Tuple of (mask, max_flow_value, computation_time)
        """
        fg_seeds = set(foreground_seeds)
        bg_seeds = set(background_seeds)
        
        print(f"\nBuilding network for {self.height}x{self.width} image...")
        print(f"Foreground seeds: {len(fg_seeds)}")
        print(f"Background seeds: {len(bg_seeds)}")
        
        # Build the flow network
        build_start = time.time()
        self.network = self.build_network(fg_seeds, bg_seeds)
        build_time = time.time() - build_start
        print(f"Network construction time: {build_time:.3f} seconds")
        
        # Compute maximum flow
        print("\nComputing maximum flow...")
        ek = EdmondsKarp(self.network)
        
        flow_start = time.time()
        max_flow_value = ek.max_flow(self.source, self.sink)
        flow_time = time.time() - flow_start
        print(f"Max flow computation time: {flow_time:.3f} seconds")
        print(f"Maximum flow value: {max_flow_value:.2f}")
        
        total_time = build_time + flow_time
        
        # Get minimum cut
        source_side, sink_side = ek.get_min_cut(self.source)
        
        # Create segmentation mask
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        for node in source_side:
            if node != self.source:
                i, j = self.node_to_pixel(node)
                mask[i, j] = 1  # Foreground
        
        foreground_pixels = np.sum(mask)
        print(f"\nSegmentation complete:")
        print(f"  Foreground pixels: {foreground_pixels}")
        print(f"  Background pixels: {self.height * self.width - foreground_pixels}")
        
        return mask, max_flow_value, total_time
    
    def save_segmentation_results(
        self,
        mask: np.ndarray,
        original_image: np.ndarray,
        prefix: str = "output"
    ) -> Tuple[str, str]:
        """Save segmentation results to files."""
        # Handle RGBA images
        if len(original_image.shape) == 3 and original_image.shape[2] == 4:
            original_image = original_image[:, :, :3]
        
        # Ensure mask matches image dimensions
        if len(original_image.shape) == 2:
            # Grayscale image
            foreground = original_image.copy()
            background = original_image.copy()
            foreground[mask == 0] = 0
            background[mask == 1] = 0
        else:
            # RGB image
            foreground = original_image.copy()
            background = original_image.copy()
            foreground[mask == 0] = [0, 0, 0]
            background[mask == 1] = [0, 0, 0]
        
        # Save images
        fg_filename = f"{prefix}_foreground.png"
        bg_filename = f"{prefix}_background.png"
        
        plt.imsave(fg_filename, foreground, cmap='gray' if len(original_image.shape) == 2 else None)
        plt.imsave(bg_filename, background, cmap='gray' if len(original_image.shape) == 2 else None)
        
        print(f"\nSaved segmentation results:")
        print(f"  Foreground: {fg_filename}")
        print(f"  Background: {bg_filename}")
        
        return fg_filename, bg_filename


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_synthetic_image(size: int = 50, pattern: str = 'circle') -> np.ndarray:
    """Create synthetic test image."""
    image = np.zeros((size, size))
    center = size // 2
    
    if pattern == 'circle':
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if dist < size // 3:
                    image[i, j] = 200
                else:
                    image[i, j] = 50
    
    elif pattern == 'square':
        quarter = size // 4
        image[quarter:3*quarter, quarter:3*quarter] = 200
        image[:quarter, :] = 50
        image[3*quarter:, :] = 50
        image[:, :quarter] = 50
        image[:, 3*quarter:] = 50
    
    elif pattern == 'gradient':
        for i in range(size):
            image[i, :] = (i / size) * 255
    
    # Add some noise
    noise = np.random.normal(0, 10, (size, size))
    image = np.clip(image + noise, 0, 255)
    
    return image


def load_image(image_path: str) -> np.ndarray:
    """Load an image from file."""
    if not PIL_AVAILABLE:
        raise ImportError("PIL/Pillow is required to load images. Install with: pip install Pillow")
    
    img = Image.open(image_path)
    return np.array(img)


def create_sample_image(filename: str = "sample_image.png", size: int = 200):
    """Create a sample test image."""
    print(f"Creating sample test image: {filename}")
    
    # Create a more interesting synthetic image
    image = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Create a gradient background
    for i in range(size):
        for j in range(size):
            image[i, j] = [30 + i // 3, 50 + j // 4, 100]
    
    # Add a bright circle in the center (foreground object)
    center_y, center_x = size // 2, size // 2
    radius = size // 3
    
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            if dist < radius:
                # Bright foreground object
                intensity = int(255 * (1 - dist / radius) * 0.7 + 100)
                r = np.clip(intensity, 0, 255)
                g = np.clip(intensity - 20, 0, 255)
                b = np.clip(intensity - 40, 0, 255)
                image[i, j] = [r, g, b]
    
    # Add some noise
    noise = np.random.randint(-20, 20, (size, size, 3))
    image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
    
    plt.imsave(filename, image)
    print(f"Sample image created: {filename}")
    print(f"Image size: {size}x{size}")
    return filename


def segment_real_image(
    image_path: str,
    foreground_seeds: List[Tuple[int, int]],
    background_seeds: List[Tuple[int, int]],
    output_prefix: str = "segmented",
    beta: float = 25.0
) -> Tuple[np.ndarray, str, str]:
    """Segment a real image and save foreground/background separately."""
    print("\nReal Image Segmentation via Network Flow\n")
    
    # Load image
    print(f"\nLoading image: {image_path}")
    original_image = load_image(image_path)
    
    # Handle RGBA images
    if len(original_image.shape) == 3 and original_image.shape[2] == 4:
        print("Converting RGBA to RGB...")
        original_image = original_image[:, :, :3]
    
    print(f"Image size: {original_image.shape}")
    
    # Perform segmentation
    segmenter = ImageSegmentation(original_image, beta=beta)
    mask, flow_value, comp_time = segmenter.segment(
        foreground_seeds,
        background_seeds
    )
    
    print(f"\nTotal computation time: {comp_time:.3f} seconds")
    print(f"Energy (min cut value): {flow_value:.2f}")
    
    # Save separated images
    fg_file, bg_file = segmenter.save_segmentation_results(
        mask, original_image, output_prefix
    )
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    if len(original_image.shape) == 2:
        axes[0, 0].imshow(original_image, cmap='gray')
    else:
        axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Seeds overlay
    if len(original_image.shape) == 2:
        axes[0, 1].imshow(original_image, cmap='gray')
    else:
        axes[0, 1].imshow(original_image)
    fg_coords = np.array(foreground_seeds)
    bg_coords = np.array(background_seeds)
    axes[0, 1].scatter(fg_coords[:, 1], fg_coords[:, 0], c='red', s=50, 
                       label='Foreground Seeds', marker='x', linewidths=2)
    axes[0, 1].scatter(bg_coords[:, 1], bg_coords[:, 0], c='blue', s=50, 
                       label='Background Seeds', marker='+', linewidths=2)
    axes[0, 1].set_title('User Seeds', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].axis('off')
    
    # Segmentation mask
    axes[0, 2].imshow(mask, cmap='gray')
    axes[0, 2].set_title('Segmentation Mask\n(White=Foreground, Black=Background)', 
                         fontsize=11, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Foreground image
    foreground = original_image.copy()
    if len(original_image.shape) == 2:
        foreground[mask == 0] = 0
        axes[1, 0].imshow(foreground, cmap='gray')
    else:
        foreground[mask == 0] = [0, 0, 0]
        axes[1, 0].imshow(foreground)
    axes[1, 0].set_title('Foreground Only', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Background image
    background = original_image.copy()
    if len(original_image.shape) == 2:
        background[mask == 1] = 0
        axes[1, 1].imshow(background, cmap='gray')
    else:
        background[mask == 1] = [0, 0, 0]
        axes[1, 1].imshow(background)
    axes[1, 1].set_title('Background Only', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Overlay visualization
    overlay = original_image.copy()
    if len(original_image.shape) == 2:
        overlay_rgb = np.stack([overlay, overlay, overlay], axis=-1) / 255.0
        overlay_rgb[mask == 1, 0] = 1.0
        overlay_rgb[mask == 1, 1] *= 0.3
        overlay_rgb[mask == 1, 2] *= 0.3
        axes[1, 2].imshow(overlay_rgb)
    else:
        overlay_rgb = overlay.astype(float) / 255.0
        overlay_rgb[mask == 1, 0] = np.clip(overlay_rgb[mask == 1, 0] * 0.3 + 0.7, 0, 1)
        overlay_rgb[mask == 1, 1] *= 0.3
        overlay_rgb[mask == 1, 2] *= 0.3
        axes[1, 2].imshow(overlay_rgb)
    axes[1, 2].set_title('Overlay (Foreground in Red)', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Image Segmentation Results (Time: {comp_time:.3f}s, Energy: {flow_value:.2f})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    viz_filename = f"{output_prefix}_complete_visualization.png"
    plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
    print(f"  Visualization: {viz_filename}")
    plt.close()
    
    return mask, fg_file, bg_file


def interactive_seed_selection(image_path: str, beta: float = 90.0):
    """Perform segmentation with automatically generated seeds."""
    print("\nInteractive Image Segmentation\n")
    
    # Load and display image
    image = load_image(image_path)
    height, width = image.shape[:2]
    
    print(f"\nImage loaded: {width}x{height} pixels")
    print("\nFor this demo, we'll use predefined seeds.")
    print("In a full application, you could click on the image to select seeds.")
    
    # Define default seeds based on image size
    center_y, center_x = height // 2, width // 2
    
    # Foreground seeds (conservative placement in center)
    foreground_seeds = []
    radius_coverage = int(min(width, height) * 0.4)
    step = max(2, radius_coverage // 8)
    
    for dy in range(-radius_coverage, radius_coverage + 1, step):
        for dx in range(-radius_coverage, radius_coverage + 1, step):
            y, x = center_y + dy, center_x + dx
            dist = np.sqrt(dy**2 + dx**2)
            if dist <= radius_coverage and 0 <= y < height and 0 <= x < width:
                foreground_seeds.append((y, x))
    
    # Background seeds (edges)
    background_seeds = []
    margin = 10
    step = max(20, width // 10)
    
    # Top and bottom edges
    for x in range(0, width, step):
        if x < width:
            background_seeds.append((margin, x))
            if height - margin > 0:
                background_seeds.append((height - margin, x))
    
    # Left and right edges
    for y in range(0, height, step):
        if y < height:
            background_seeds.append((y, margin))
            if width - margin > 0:
                background_seeds.append((y, width - margin))
    
    print(f"\nForeground seeds: {len(foreground_seeds)}")
    print(f"Background seeds: {len(background_seeds)}")
    
    # Perform segmentation
    mask, fg_file, bg_file = segment_real_image(
        image_path,
        foreground_seeds,
        background_seeds,
        output_prefix="segmented",
        beta=beta
    )
    print("\nSegmentation complete!")
    
    return mask, fg_file, bg_file


def demo_segmentation():
    """Demonstrate image segmentation on synthetic images."""
    print("\nImage Segmentation via Network Flow - Demo\n")
    
    # Create synthetic image
    image_size = 30
    print(f"\nCreating {image_size}x{image_size} synthetic image...")
    image = create_synthetic_image(image_size, pattern='circle')
    
    # Define seeds
    center = image_size // 2
    foreground_seeds = [
        (center, center),
        (center-2, center),
        (center+2, center),
        (center, center-2),
        (center, center+2)
    ]
    
    background_seeds = [
        (2, 2),
        (2, image_size-3),
        (image_size-3, 2),
        (image_size-3, image_size-3),
        (image_size//2, 2),
        (image_size//2, image_size-3)
    ]
    
    # Perform segmentation
    segmenter = ImageSegmentation(image, beta=20.0)
    mask, flow_value, comp_time = segmenter.segment(
        foreground_seeds, 
        background_seeds
    )
    
    print(f"\nTotal computation time: {comp_time:.3f} seconds")
    print(f"Energy (min cut value): {flow_value:.2f}")
    
    # Save separated images
    fg_file, bg_file = segmenter.save_segmentation_results(
        mask, image, "demo"
    )
    
    print("\nSegmentation mask (1=foreground, 0=background):")
    print("First 10x10 region:")
    print(mask[:10, :10])
    
    print("\n" + "=" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point for the script."""
    print("\n" + "=" * 70)
    print("IMAGE SEGMENTATION VIA NETWORK FLOW")
    print("Maximum Flow / Minimum Cut Algorithm")
    print("=" * 70)
    
    if len(sys.argv) > 1:
        # Image path provided as command-line argument
        image_path = sys.argv[1]
        print(f"\nUsing provided image: {image_path}")
    else:
        # Create a sample image for demonstration
        image_path = create_sample_image("sample_image.png", size=80)
    
    # Get beta parameter if provided
    beta = 10.0
    if len(sys.argv) > 2:
        try:
            beta = float(sys.argv[2])
            print(f"Using beta parameter: {beta}")
        except ValueError:
            print(f"Invalid beta value. Using default: {beta}")
    
    # Perform segmentation with interactive seed selection
    try:
        mask, fg_file, bg_file = interactive_seed_selection(image_path, beta)
        print("\n" + "=" * 70)
        print("SUCCESS! Open these files to see the segmentation results:")
        print(f"  - {fg_file}")
        print(f"  - {bg_file}")
        print(f"  - segmented_complete_visualization.png")
        print("=" * 70 + "\n")
        
    except FileNotFoundError:
        print(f"\nError: Image file not found: {image_path}")
        print("\nUsage:")
        print("  python image_segmentation_complete.py [image_path] [beta]")
        print("\nExamples:")
        print("  python image_segmentation_complete.py photo.jpg 90")
        print("  python image_segmentation_complete.py              # Uses sample image")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
