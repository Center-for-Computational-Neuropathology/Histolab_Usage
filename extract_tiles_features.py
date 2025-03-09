"""
Extraction of patches and features from hippocampus regions using histolab and a pretrained model.
"""

import numpy as np
import os
import json
import h5py
import traceback
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms
from PIL import Image
import openslide
import cv2

# Import your existing histolab implementation
from histolab_implementation import (process_single_slide, CoordinatesRandomTiler, 
                                    CoordinatesGridTiler, PolygonMask)
from histolab.slide import Slide
from histolab.types import CoordinatePair

def extract_patch(slide_path, coord, patch_size=(256, 256)):
    """
    Extract a patch from the slide at the given coordinates using OpenSlide.
    
    Parameters:
    ----------
    slide_path : str
        Path to the slide file
    coord : CoordinatePair
        Coordinates for the patch
    patch_size : tuple
        Size of the patch
        
    Returns:
    -------
    np.ndarray
        Patch image as a numpy array with shape (patch_size[0], patch_size[1], 3)
    """
    # Open the slide using OpenSlide
    slide = openslide.OpenSlide(slide_path)
    
    # Extract the patch at the specified coordinates
    patch = slide.read_region(
        (coord.x_ul, coord.y_ul),
        0,  # level 0 for highest resolution
        patch_size
    )
    
    # Convert to RGB (remove alpha channel if it exists)
    patch = patch.convert('RGB')
    
    # Convert to numpy array
    patch_array = np.array(patch)
    
    # Ensure it's the right shape
    if patch_array.shape[:2] != patch_size:
        # Resize if needed
        patch = patch.resize(patch_size)
        patch_array = np.array(patch)
    
    return patch_array

def extract_features(patch, model, transform, device="cpu"):
    """
    Extract features from a patch using a pretrained model.
    
    Parameters:
    ----------
    patch : np.ndarray
        Patch image as a numpy array
    model : torch.nn.Module
        Pretrained model
    transform : torchvision.transforms
        Transformations to apply to the patch
    device : str
        Device to run the model on
        
    Returns:
    -------
    np.ndarray
        Feature vector
    """
    # Convert patch to PIL Image
    patch_img = Image.fromarray(patch.astype('uint8'))
    
    # Apply transformations
    patch_tensor = transform(patch_img).unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        features = model(patch_tensor)
        
        # If features is not a tensor but a tuple (some models return multiple values)
        if isinstance(features, tuple):
            features = features[0]
        
        # If the tensor has extra dimensions (like from AdaptiveAvgPool2d), flatten them
        if len(features.shape) > 2:
            features = torch.flatten(features, start_dim=1)
    
    # Convert to numpy array
    return features.cpu().numpy().flatten()

def create_attention_map(patch):
    """
    Create a simplified attention visualization.
    
    Parameters:
    ----------
    patch : np.ndarray
        Patch image as a numpy array
        
    Returns:
    -------
    np.ndarray
        Colorized patch to show as a placeholder for attention
    """
    # Convert to grayscale
    gray = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Apply a simple edge detection or other highlighting
    edges = cv2.Canny(gray, 100, 200)
    
    # Create a heat map
    heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
    
    # Ensure consistent data types
    overlay = patch.copy().astype(np.float32)
    heatmap = heatmap.astype(np.float32)  # Convert heatmap to float32
    
    # Overlay on original image
    alpha = 0.4  # Transparency factor
    overlay = cv2.addWeighted(overlay, 1 - alpha, heatmap, alpha, 0)
    
    return overlay.astype(np.uint8)

def process_and_extract_features(slide_path, json_path, output_dir, 
                               tiling_method='random', n_tiles=100, 
                               tile_size=(256, 256), model_name='resnet50'):
    """
    Process a slide, extract patches and features, and save to HDF5.
    
    Parameters:
    ----------
    slide_path : str
        Path to the slide file
    json_path : str
        Path to the JSON annotation file
    output_dir : str
        Directory to save results
    tiling_method : str
        'random' for RandomTiler or 'grid' for GridTiler
    n_tiles : int
        Number of tiles to extract per region (only used for RandomTiler)
    tile_size : tuple
        Size of each tile (width, height)
    model_name : str
        Name of the pretrained model to use
        
    Returns:
    -------
    str
        Path to the saved HDF5 file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Setting up environment variables to limit threads...")
    # Limit OpenBLAS threads
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    # Step 1: Get tile coordinates using your existing implementation
    print(f"Processing slide to get tile coordinates using {tiling_method} tiling...")
    start_time = time.time()
    
    try:
        # For grid tiler, don't pass n_tiles
        if tiling_method.lower() == 'grid':
            # Create a temporary output directory for process_single_slide
            temp_dir = os.path.join(output_dir, f"temp_{tiling_method}")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Call process_single_slide directly with modified parameters for GridTiler
            slide = Slide(slide_path, processed_path=temp_dir)
            
            # Load JSON with annotations
            with open(json_path, 'r') as f:
                annotations = json.load(f)
                
            # Extract regions using your existing implementation's helper functions
            from histolab_implementation import getZeroer, coordFixer, getPolygonOfAnnotationLabel
            
            regions = {}
            for annotation in annotations:
                if 'properties' in annotation and 'classification' in annotation['properties']:
                    name = annotation['properties']['classification']['name']
                    # Apply name mapping (from your implementation)
                    name_mapping = {
                        "DG": "Dentate Gyrus",
                        "dg": "Dentate Gyrus",
                        "Dg": "Dentate Gyrus",
                        "dG": "Dentate Gyrus",
                        "Subicular Complex": "Subiculum",
                        "Subiculum Complex": "Subiculum",
                        "subicular complex": "Subiculum",
                        "subiculum complex": "Subiculum",
                        "subicular Complex": "Subiculum",
                        "subiculum Complex": "Subiculum",
                        "Subicular complex": "Subiculum",
                        "Subiculum complex": "Subiculum"
                    }
                    name = name_mapping.get(name, name)
                    
                    # Skip Hippocampus annotations (parent region)
                    if "hippocampus" not in name.lower():
                        if 'geometry' in annotation and 'coordinates' in annotation['geometry']:
                            try:
                                coord_data = annotation['geometry']['coordinates'][0]
                                coord = np.array(coord_data, dtype=object)
                                coord = coordFixer(coord)
                                
                                # Store coordinates by region
                                if name not in regions:
                                    regions[name] = []
                                regions[name].append(coord)
                            except Exception as e:
                                print(f"Error processing coordinate: {e}")
            
            # Create GridTiler without n_tiles
            grid_tiler = CoordinatesGridTiler(
                tile_size=tile_size,
                level=0,
                tissue_percent=80.0
            )
            
            # Process each region
            results = {}
            vis_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            for region_name, region_coords_list in regions.items():
                print(f"Processing region: {region_name}")
                try:
                    # Convert coordinates to polygons
                    polygons = [getPolygonOfAnnotationLabel(coords) for coords in region_coords_list]
                    
                    # Combine multiple polygons if needed
                    if len(polygons) > 1:
                        from shapely.ops import unary_union
                        combined_polygon = unary_union(polygons)
                    else:
                        combined_polygon = polygons[0]
                    
                    # Get exterior coordinates
                    exterior_coords = list(combined_polygon.exterior.coords)
                    
                    # Create mask using PolygonMask
                    mask = PolygonMask(exterior_coords)
                    
                    # Extract coordinates using histolab tiler
                    tile_coords = grid_tiler.extract(slide, mask)
                    
                    # Apply n_tiles limit if specified
                    if n_tiles and len(tile_coords) > n_tiles:
                        # For grid tiler, select a random subset to respect n_tiles
                        np.random.shuffle(tile_coords)
                        tile_coords = tile_coords[:n_tiles]
                    
                    # Store results
                    results[region_name] = tile_coords
                    print(f"Extracted {len(tile_coords)} {tiling_method} tile coordinates for {region_name}")
                    
                    # Visualize if needed
                    from histolab_implementation import visualize_region_with_tiles
                    visualize_region_with_tiles(
                        slide=slide,
                        region_name=region_name,
                        region_coords=exterior_coords,
                        tile_coords=tile_coords,
                        output_dir=vis_dir,
                        method=tiling_method
                    )
                except Exception as e:
                    print(f"Error processing region {region_name}: {e}")
                    traceback.print_exc()
            
            # Also create Non-Hippocampus region if possible
            region_coords = results
        else:
            # For random tiler, use the regular process_single_slide
            region_coords = process_single_slide(
                slide_path=slide_path,
                json_path=json_path,
                output_dir=output_dir,
                tiling_method=tiling_method,
                n_tiles=n_tiles,
                tile_size=tile_size,
                visualize=True
            )
    except Exception as e:
        print(f"Error processing slide: {e}")
        traceback.print_exc()
        region_coords = {}
        
    print(f"Coordinates extraction completed in {time.time() - start_time:.2f} seconds")
    
    # Step 2: Set up model for feature extraction
    print(f"Setting up {model_name} for feature extraction...")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup model for feature extraction
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        # Remove the final classification layer
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        feature_dim = 2048
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        feature_dim = 512
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True).features
        model = torch.nn.Sequential(model, torch.nn.AdaptiveAvgPool2d((1, 1)))
        feature_dim = 512
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    model = model.to(device)
    model.eval()
    
    # Setup transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Step 3: Extract patches and features
    print("Extracting patches and features...")
    vis_dir = os.path.join(output_dir, f"{tiling_method}_attention_maps")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Count total patches to process
    total_patches = sum(len(coords) for coords in region_coords.values())
    
    # Prepare data structures
    all_coords = []
    all_patches = []
    all_features = []
    all_labels = []
    all_attentions = []
    
    # Create a progress bar
    pbar = tqdm(total=total_patches, desc="Processing patches")
    
    for region_name, coords_list in region_coords.items():
        print(f"\nProcessing {len(coords_list)} patches for region: {region_name}")
        
        # Create a figure for attention maps visualization (if there are any patches)
        if len(coords_list) > 0:
            n_cols = min(5, len(coords_list))
            n_rows = min(4, (len(coords_list) + n_cols - 1) // n_cols)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        for i, coord in enumerate(coords_list):
            try:
                # Extract patch using OpenSlide
                patch = extract_patch(slide_path, coord, patch_size=tile_size)
                
                # Extract features
                features = extract_features(patch, model, transform, device)
                
                # Create attention map
                attention_map = create_attention_map(patch)
                
                # Store data
                all_coords.append([float(coord.x_ul), float(coord.y_ul)])
                all_patches.append(patch)
                all_features.append(features)
                all_labels.append(region_name)
                all_attentions.append(attention_map)
                
                # Visualize some attention maps
                if i < len(axes):
                    axes[i].imshow(attention_map)
                    axes[i].set_title(f"{region_name}\n({coord.x_ul}, {coord.y_ul})")
                    axes[i].axis('off')
                
                # Update progress bar
                pbar.update(1)
                
            except Exception as e:
                print(f"Error processing patch at {coord}: {str(e)}")
                pbar.update(1)
        
        # Save the attention maps figure if we have any patches
        if len(coords_list) > 0:
            # Hide unused axes
            for j in range(min(i + 1, len(axes)), len(axes)):
                axes[j].axis('off')
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"{region_name}_attention_maps.png"), dpi=150)
            plt.close()
    
    pbar.close()
    
    # Step 4: Convert to numpy arrays
    print("Converting data to numpy arrays...")
    all_coords = np.array(all_coords, dtype=np.float32)
    all_patches = np.array(all_patches, dtype=np.float32)
    all_features = np.array(all_features, dtype=np.float32)
    all_labels = np.array([label.encode('utf-8') for label in all_labels])  # Convert to bytes for HDF5
    
    # Step 5: Save to HDF5 file
    print("Saving to HDF5 file...")
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    h5_file = os.path.join(output_dir, f"{slide_name}_{tiling_method}_features.h5")
    
    with h5py.File(h5_file, 'w') as f:
        f.create_dataset('PatchCoords', data=all_coords)
        f.create_dataset('Patches', data=all_patches)
        f.create_dataset('features', data=all_features)
        f.create_dataset('labels', data=all_labels)
    
    # Also save a visualization sample of attention maps
    if len(all_attentions) > 0:
        attention_sample_file = os.path.join(output_dir, f"{slide_name}_{tiling_method}_attention_sample.png")
        sample_size = min(5, len(all_attentions))
        fig, axes = plt.subplots(1, sample_size, figsize=(sample_size * 4, 4))
        if sample_size == 1:
            axes = [axes]
        for i in range(sample_size):
            axes[i].imshow(all_attentions[i])
            label = all_labels[i].decode('utf-8') if isinstance(all_labels[i], bytes) else all_labels[i]
            axes[i].set_title(f"{label}")
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(attention_sample_file, dpi=150)
        plt.close()
    
    # Step 6: Print summary information
    print(f"\nSuccessfully saved data to {h5_file}")
    print("HDF5 structure:")
    print(f"  - PatchCoords: shape={all_coords.shape}, dtype={all_coords.dtype}")
    print(f"  - Patches: shape={all_patches.shape}, dtype={all_patches.dtype}")
    print(f"  - features: shape={all_features.shape}, dtype={all_features.dtype}")
    print(f"  - labels: shape={all_labels.shape}, dtype={all_labels.dtype}")
    print(f"Attention maps visualizations saved to {vis_dir}")
    
    return h5_file

def inspect_h5_file(h5_file):
    """
    Print information about an HDF5 file.
    
    Parameters:
    ----------
    h5_file : str
        Path to the HDF5 file
    """
    with h5py.File(h5_file, 'r') as f:
        print(f"HDF5 File Structure:")
        for key in f.keys():
            dataset = f[key]
            shape = dataset.shape
            dtype = dataset.dtype
            print(f"Path: {key}")
            print(f"  - Type: Dataset")
            print(f"  - Shape: {shape}")
            print(f"  - Dtype: {dtype}")
            
            # Print a sample of the dataset
            if len(shape) == 1 and shape[0] > 0:
                sample = dataset[:min(5, shape[0])]
                print(f"  - First few values: {sample}")
            elif len(shape) == 2 and shape[0] > 0:
                sample = dataset[:min(5, shape[0]), :]
                print(f"  - First few values: {sample}")
            elif len(shape) > 2 and shape[0] > 0:
                sample = "Multidimensional data (truncated for display)"
                print(f"  - First few values: {sample}")
            else:
                print(f"  - First few values: []")
            print()

if __name__ == "__main__":
    import time
    
    # Process a single slide
    slide_path = "/sc/arion/projects/tauomics/PART_images/Hippocampus_LFB_HE/42054.svs"
    json_path = "/sc/arion/projects/tauomics/danielk/qupath_json_data_files/jsondatafiles/42054_1.json"
    output_dir = "/sc/arion/projects/tauomics/Shrishtee/HistoLab/GRID_TILER/feature_h5_output"
    
    # Extract patches and features using RandomTiler
    print("\n=== PROCESSING WITH RANDOM TILER ===")
    random_h5_file = process_and_extract_features(
        slide_path=slide_path,
        json_path=json_path,
        output_dir=output_dir,
        tiling_method='random',
        n_tiles=100,  # 100 tiles per region
        tile_size=(256, 256),
        model_name='resnet50'
    )
    
    # Inspect the HDF5 file
    inspect_h5_file(random_h5_file)
    
    # Extract patches and features using GridTiler
    print("\n=== PROCESSING WITH GRID TILER ===")
    grid_h5_file = process_and_extract_features(
        slide_path=slide_path,
        json_path=json_path,
        output_dir=output_dir,
        tiling_method='grid',
        n_tiles=100,  # max tiles to select from grid tiler output
        tile_size=(256, 256),
        model_name='resnet50'
    )
    
    # Inspect the HDF5 file
    inspect_h5_file(grid_h5_file)
