"""
Implementation of hippocampus region tiling using histolab.
"""

import numpy as np
import os
import json
import pandas as pd
from pathlib import Path
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL.Image
import PIL.ImageDraw

# Import histolab components
from histolab.slide import Slide
from histolab.masks import BinaryMask
from histolab.tiler import GridTiler, RandomTiler
# Import specific utils that should be available in most histolab versions
from histolab.util import rectangle_to_mask, scale_coordinates
from histolab.types import CoordinatePair

# Import shapely for polygon handling
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import traceback

from histolab.filters.image_filters import RgbToGrayscale, OtsuThreshold
from skimage.morphology import binary_dilation as BinaryDilation
from histolab.filters.morphological_filters import RemoveSmallObjects

from skimage.morphology import disk

# Define named tuples for vertices
ScaledPolygonVertices = namedtuple("ScaledPolygonVertices", 'vertices_list')
SP = ScaledPolygonVertices

# Custom polygon_to_mask implementation (since it might not be in histolab.util)
def polygon_to_mask(dims, vertices):
    """
    Return a binary mask with True inside of polygon ``vertices`` and False outside.
    
    Parameters
    ----------
    dims : Tuple[int, int]
        (width, height) of the binary mask
    vertices : ScaledPolygonVertices
        ScaledPolygonVertices representing the polygon vertices
        
    Returns
    -------
    np.ndarray
        Binary mask with True inside of the polygon, False outside.
    """
    if not isinstance(vertices, ScaledPolygonVertices):
        raise ValueError(f"vertices must be of type ScaledPolygonVertices, got {type(vertices)}")
    
    polygon_vertices = vertices.vertices_list
    
    img = PIL.Image.new("L", dims, 0)
    PIL.ImageDraw.Draw(img).polygon(polygon_vertices, outline=1, fill=1)
    return np.array(img).astype(bool)

# Properly implement PolygonMask by extending BinaryMask
class PolygonMask(BinaryMask):
    """
    Create a binary mask from a polygon.
    """
    def __init__(self, vertices_list):
        super().__init__()
        self.vertices_list = vertices_list
        self.scaled_vertices = SP(vertices_list)
    
    def _mask(self, slide):
        """
        Generate binary mask from polygon vertices.
        
        This properly implements the abstract method in BinaryMask.
        """
        # Use our custom polygon_to_mask function
        mask = polygon_to_mask(slide.dimensions, self.scaled_vertices)
        return mask

class CustomBooleanMask(BinaryMask):
    """
    Create a binary mask from a boolean array.
    """
    def __init__(self, boolean_mask):
        super().__init__()
        self.boolean_mask = boolean_mask
    
    def _mask(self, slide):
        """
        Return the boolean mask.
        
        This properly implements the abstract method in BinaryMask.
        """
        return self.boolean_mask

# Custom RandomTiler that only extracts coordinates
class CoordinatesRandomTiler(RandomTiler):
    """
    Modified RandomTiler that returns coordinates instead of extracting tiles.
    """
    def extract(self, slide, extraction_mask, log_level="INFO"):
        """
        Extract random tile coordinates without saving tiles.
        """
        self._validate_level(slide)
        self.tile_size = self._tile_size(slide)
        self._validate_tile_size(slide)
        
        coords_list = []
        for _ in range(self.n_tiles):
            try:
                coords = self._random_tile_coordinates(slide, extraction_mask)
                coords_list.append(coords)
            except Exception as e:
                print(f"Error generating random tile: {e}")
                # Continue trying to extract remaining tiles
                continue
            
            # Stop if we've reached the desired number of tiles
            if len(coords_list) >= self.n_tiles:
                break
                
        return coords_list

# Custom GridTiler that only extracts coordinates
class CoordinatesGridTiler(GridTiler):
    """
    Modified GridTiler that returns coordinates instead of extracting tiles.
    """
    def extract(self, slide, extraction_mask, log_level="INFO"):
        """
        Extract grid tile coordinates without saving tiles.
        """
        self._validate_level(slide)
        self.tile_size = self._tile_size(slide)
        self._validate_tile_size(slide)
        
        # Get binary mask from extraction_mask
        binary_mask = extraction_mask(slide)
        
        # Get the coordinates in the reference frame of the binary mask
        reference_coords = list(self._grid_coordinates_from_bbox_coordinates(
            self._compute_tile_coordinates(binary_mask)
        ))
        
        # Convert the reference coordinates to slide coordinates
        tile_wsi_coords = [
            self._ref_to_wsi_coordinates(bbox_coords, binary_mask, slide)
            for bbox_coords in reference_coords
        ]
        
        # Check if we have a limit on the number of tiles
        if self.n_tiles and len(tile_wsi_coords) > self.n_tiles:
            tile_wsi_coords = tile_wsi_coords[:self.n_tiles]
            
        return tile_wsi_coords
    
    def _compute_tile_coordinates(self, binary_mask):
        """
        Compute coordinates for tiles in a grid pattern.
        
        Parameters:
        ----------
        binary_mask : np.ndarray
            Binary mask of the region
        
        Returns:
        -------
        list
            List of coordinates for tiles
        """
        height, width = binary_mask.shape
        
        # Calculate number of tiles in each dimension
        n_tiles_h = height // self.tile_size[0]
        n_tiles_w = width // self.tile_size[1]
        
        # Create list to store coordinates
        coordinates = []
        
        for h in range(n_tiles_h):
            for w in range(n_tiles_w):
                # Calculate tile coordinates
                top_left = (w * self.tile_size[1], h * self.tile_size[0])
                bottom_right = ((w + 1) * self.tile_size[1], (h + 1) * self.tile_size[0])
                
                # Create a mask for this tile
                tile_mask = np.zeros_like(binary_mask)
                tile_mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 1
                
                # Check if tile overlaps with tissue (binary_mask)
                tissue_percentage = np.sum(tile_mask & binary_mask) / np.sum(tile_mask) * 100
                
                # Add coordinates if tissue percentage meets threshold
                if tissue_percentage >= self.tissue_percent:
                    coordinates.append((top_left, bottom_right))
        
        return coordinates

# Helper functions
def getZeroer(array):
    """Extract first element if array is 3D, otherwise return as is."""
    array = np.array(array)
    if len(array.shape) != 2:
        return array[0]
    else:
        return array

def coordFixer(coords):
    """Fix coordinate array format."""
    return getZeroer(coords)

def getPolygonOfAnnotationLabel(coord):
    """Convert coordinates to a valid polygon."""
    polygon = Polygon(coord)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon

def getAnnotationNames():
    """Get list of hippocampus region annotation names."""
    return np.array([
        'Dentate Gyrus', 
        'CA1', 
        'CA2', 
        'CA3', 
        'Subiculum'
    ])

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder to handle NumPy types when serializing to JSON."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def process_single_slide(slide_path, json_path, output_dir, tiling_method='random', n_tiles=100, tile_size=(256, 256), visualize=True):
    """
    Process a single slide and extract tile coordinates.
    
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
        Number of tiles to extract per region
    tile_size : tuple
        Size of each tile (width, height)
    visualize : bool
        Whether to create visualizations
        
    Returns:
    -------
    dict
        Dictionary with region names as keys and lists of coordinates as values
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create Slide object using histolab
        slide = Slide(slide_path, processed_path=output_dir)
        print(f"Slide dimensions: {slide.dimensions}")
        
        # Load JSON with annotations
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        
        # Extract region polygons
        regions = {}
        for annotation in annotations:
            if 'properties' in annotation and 'classification' in annotation['properties']:
                name = annotation['properties']['classification']['name']
                
                # Clean up annotation names
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
        
        # Process each region
        results = {}
        
        # Choose the appropriate tiler
        if tiling_method.lower() == 'random':
            tiler = CoordinatesRandomTiler(
                tile_size=tile_size,
                n_tiles=n_tiles,
                level=0,
                tissue_percent=95.0  # Increased threshold for tissue percentage
            )
        else:  # grid method
            tiler = CoordinatesGridTiler(
                tile_size=tile_size,
                n_tiles=n_tiles,
                level=0,
                tissue_percent=95.0  # Increased threshold for tissue percentage
            )
        
        # Create visualization directory if needed
        if visualize:
            vis_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
        
        for region_name, region_coords_list in regions.items():
            print(f"Processing region: {region_name}")
            
            try:
                # Convert coordinates to polygons
                polygons = [getPolygonOfAnnotationLabel(coords) for coords in region_coords_list]
                
                # Combine multiple polygons if needed
                if len(polygons) > 1:
                    combined_polygon = unary_union(polygons)
                else:
                    combined_polygon = polygons[0]
                
                # Get exterior coordinates
                exterior_coords = list(combined_polygon.exterior.coords)
                
                # Create mask using PolygonMask
                mask = PolygonMask(exterior_coords)
                
                # Extract coordinates using histolab tiler
                tile_coords = tiler.extract(slide, mask)
                
                # Store results
                results[region_name] = tile_coords
                print(f"Extracted {len(tile_coords)} {tiling_method} tile coordinates for {region_name}")
                
                # Save coordinates to JSON
                serializable_coords = [
                    {'x_ul': int(coord.x_ul), 'y_ul': int(coord.y_ul), 'x_br': int(coord.x_br), 'y_br': int(coord.y_br)}
                    for coord in tile_coords
                ]
                
                coord_file = os.path.join(output_dir, f"{region_name.replace(' ', '_')}_coords.json")
                with open(coord_file, 'w') as f:
                    json.dump(serializable_coords, f, indent=2, cls=NumpyEncoder)
                
                # Create visualization if requested
                if visualize:
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
        
        # Create Non-Hippocampus region
        try:
            # Check if we have all necessary regions
            necessary_regions = {'CA1', 'CA2', 'CA3', 'Subiculum', 'Dentate Gyrus'}
            present_regions = set(regions.keys())
            
            if necessary_regions.issubset(present_regions):
                print("Creating Non-Hippocampus region")
                
                # Create combined mask for all hippocampus regions
                hc_poly_list = []
                
                for region_name in necessary_regions:
                    region_coords_list = regions[region_name]
                    
                    for coords in region_coords_list:
                        polygon = getPolygonOfAnnotationLabel(coords)
                        hc_poly_list.append(polygon)
                
                # Create union of all polygons
                hc_union = unary_union(hc_poly_list)
                hc_exterior = list(hc_union.exterior.coords)
                
                # Create improved tissue mask
                tissue_mask = enhanced_tissue_detection(slide)
                
                # Resize tissue mask to slide dimensions
                slide_dimensions = slide.dimensions
                tissue_mask_resized = np.zeros((slide_dimensions[1], slide_dimensions[0]), dtype=bool)
                mask_image = PIL.Image.fromarray(tissue_mask.astype(np.uint8) * 255)
                mask_resized = mask_image.resize((slide_dimensions[0], slide_dimensions[1]))
                tissue_mask_resized = np.array(mask_resized) > 0
                
                # Create hippocampus mask at slide dimensions
                hc_mask_slide_dims = np.zeros_like(tissue_mask_resized)
                hc_img = PIL.Image.new('L', (slide_dimensions[0], slide_dimensions[1]), 0)
                PIL.ImageDraw.Draw(hc_img).polygon(hc_exterior, outline=1, fill=1)
                hc_mask_slide_dims = np.array(hc_img).astype(bool)
                
                # Combine masks: tissue AND NOT hippocampus
                tissue_not_hc_mask = tissue_mask_resized & ~hc_mask_slide_dims
                
                # Create custom mask for tiler
                not_hc_tissue_mask = CustomBooleanMask(tissue_not_hc_mask)
                
                # Use the new extract_valid_tissue_tiles function
                tile_coords = extract_valid_tissue_tiles(
                    slide=slide,
                    mask=not_hc_tissue_mask,
                    tiler=tiler,
                    n_tiles=n_tiles,
                    tissue_mask=tissue_not_hc_mask
                )
                
                # Store results
                results["Not_Hippocampus"] = tile_coords
                print(f"Extracted {len(tile_coords)} {tiling_method} tile coordinates for Not_Hippocampus (Tissue Only)")
                
                # Save coordinates to JSON
                serializable_coords = [
                    {'x_ul': int(coord.x_ul), 'y_ul': int(coord.y_ul), 'x_br': int(coord.x_br), 'y_br': int(coord.y_br)}
                    for coord in tile_coords
                ]
                
                coord_file = os.path.join(output_dir, "Not_Hippocampus_coords.json")
                with open(coord_file, 'w') as f:
                    json.dump(serializable_coords, f, indent=2, cls=NumpyEncoder)
                
                # Create visualization if requested
                if visualize:
                    visualize_not_hippocampus_improved(
                        slide=slide,
                        hc_coords=hc_exterior,
                        tile_coords=tile_coords,
                        output_dir=vis_dir,
                        method=tiling_method,
                        tissue_mask=tissue_mask_resized
                    )
            else:
                print("Not all necessary regions present, skipping Non-Hippocampus region")
                
        except Exception as e:
            print(f"Error creating Non-Hippocampus region: {e}")
            traceback.print_exc()
            
        return results
    
    except Exception as e:
        print(f"Error processing slide: {e}")
        traceback.print_exc()
        return {}
    
def process_single_slide_with_improved_non_hc(slide_path, json_path, output_dir, 
                                            tiling_method='random', n_tiles=100, 
                                            tile_size=(256, 256), visualize=True):
    """
    Modified version of process_single_slide with improved Not_Hippocampus extraction.
    
    This is a template for how to integrate the improved methods into the existing code.
    """
    from histolab_implementation_2 import process_single_slide
    import os
    import json
    import numpy as np
    
    # Process the slide using the original function
    results = process_single_slide(
        slide_path=slide_path,
        json_path=json_path,
        output_dir=output_dir,
        tiling_method=tiling_method,
        n_tiles=n_tiles,
        tile_size=tile_size,
        visualize=visualize
    )
    
    # Create Slide object for improved non-hippocampus processing
    from histolab.slide import Slide
    slide = Slide(slide_path, processed_path=output_dir)
    
    # Create visualization directory if needed
    if visualize:
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
    
    # Extract the Not_Hippocampus region using the improved method
    try:
        # Load JSON with annotations to get regions
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        
        from histolab_implementation_2 import getZeroer, coordFixer, getPolygonOfAnnotationLabel
        
        # Extract region polygons
        regions = {}
        for annotation in annotations:
            if 'properties' in annotation and 'classification' in annotation['properties']:
                name = annotation['properties']['classification']['name']
                
                # Clean up annotation names
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
        
        # Replace Not_Hippocampus with the improved version
        print("\nGenerating improved Not_Hippocampus tiles...")
        tile_coords, tissue_mask = enhanced_tissue_detection(
            slide=slide,
            regions=regions,
            tiling_method=tiling_method,
            n_tiles=n_tiles,
            tile_size=tile_size
        )
        
        # Store results
        results["Not_Hippocampus"] = tile_coords
        print(f"Extracted {len(tile_coords)} {tiling_method} tile coordinates for Not_Hippocampus (Improved)")
        
        # Save coordinates to JSON
        from histolab.types import CoordinatePair
        serializable_coords = [
            {'x_ul': int(coord.x_ul), 'y_ul': int(coord.y_ul), 'x_br': int(coord.x_br), 'y_br': int(coord.y_br)}
            for coord in tile_coords
        ]
        
        coord_file = os.path.join(output_dir, "Not_Hippocampus_coords.json")
        
        class NumpyEncoder(json.JSONEncoder):
            """Custom encoder to handle NumPy types when serializing to JSON."""
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        with open(coord_file, 'w') as f:
            json.dump(serializable_coords, f, indent=2, cls=NumpyEncoder)
        
        # Create visualization if requested
        if visualize:
            # Create hippocampus boundary for visualization
            from shapely.ops import unary_union
            
            necessary_regions = {'CA1', 'CA2', 'CA3', 'Subiculum', 'Dentate Gyrus'}
            hc_poly_list = []
            
            for region_name in necessary_regions:
                if region_name in regions:
                    region_coords_list = regions[region_name]
                    for coords in region_coords_list:
                        polygon = getPolygonOfAnnotationLabel(coords)
                        hc_poly_list.append(polygon)
            
            hc_union = unary_union(hc_poly_list)
            hc_exterior = list(hc_union.exterior.coords)
            
            # Generate improved visualization
            visualize_not_hippocampus_improved(
                slide=slide,
                hc_coords=hc_exterior,
                tile_coords=tile_coords,
                output_dir=vis_dir,
                method=tiling_method,
                tissue_mask=tissue_mask
            )
            
            print("Improved visualization generated.")
    
    except Exception as e:
        import traceback
        print(f"Error creating improved Non-Hippocampus region: {e}")
        traceback.print_exc()
    
    return results

def visualize_region_with_tiles(slide, region_name, region_coords, tile_coords, output_dir, method):
    """
    Create a visualization of a region with tile coordinates.
    
    Parameters:
    ----------
    slide : histolab.slide.Slide
        Slide object
    region_name : str
        Name of the region
    region_coords : list
        List of (x, y) coordinates defining the region boundary
    tile_coords : list
        List of CoordinatePair objects for the tiles
    output_dir : str
        Directory to save the visualization
    method : str
        Tiling method ('random' or 'grid')
    """
    # Create a figure
    plt.figure(figsize=(12, 10))
    
    # Get slide thumbnail
    thumbnail = slide.thumbnail
    plt.imshow(thumbnail)
    
    # Calculate scale factors
    scale_x = thumbnail.size[0] / slide.dimensions[0]
    scale_y = thumbnail.size[1] / slide.dimensions[1]
    
    # Create an overlay for the region
    scaled_coords = [(int(x * scale_x), int(y * scale_y)) for x, y in region_coords]
    region_img = PIL.Image.new('L', thumbnail.size, 0)
    PIL.ImageDraw.Draw(region_img).polygon(scaled_coords, outline=1, fill=1)
    region_mask = np.array(region_img).astype(bool)
    
    # Overlay the region mask
    plt.imshow(region_mask, alpha=0.3, cmap='cool')
    
    # Add tile rectangles
    for coord in tile_coords:
        # Scale coordinates to thumbnail size
        x_ul_scaled = int(coord.x_ul * scale_x)
        y_ul_scaled = int(coord.y_ul * scale_y)
        width_scaled = int((coord.x_br - coord.x_ul) * scale_x)
        height_scaled = int((coord.y_br - coord.y_ul) * scale_y)
        
        # Add rectangle
        rect = patches.Rectangle(
            (x_ul_scaled, y_ul_scaled), width_scaled, height_scaled,
            linewidth=1, edgecolor='r', facecolor='none'
        )
        plt.gca().add_patch(rect)
    
    plt.title(f"{region_name} with {method} tiling ({len(tile_coords)} tiles)")
    plt.axis('off')
    
    # Save the visualization
    output_file = os.path.join(output_dir, f"{region_name.replace(' ', '_')}_{method}_tiling.png")
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_file}")

def visualize_not_hippocampus_improved(slide, hc_coords, tile_coords, output_dir, method, tissue_mask):
    """
    Create an improved visualization that better shows the tissue mask and tile selection.
    
    Parameters:
    ----------
    slide : histolab.slide.Slide
        Slide object
    hc_coords : list
        List of (x, y) coordinates defining the hippocampus boundary
    tile_coords : list
        List of CoordinatePair objects for the tiles
    output_dir : str
        Directory to save the visualization
    method : str
        Tiling method ('random' or 'grid')
    tissue_mask : np.ndarray
        Binary mask of tissue areas
    """
    import os
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    import PIL.Image
    import PIL.ImageDraw
    
    # Create a figure with two subplots - one for overview, one zoomed in
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Get slide thumbnail
    thumbnail = slide.thumbnail
    
    # Calculate scale factors
    scale_x = thumbnail.size[0] / slide.dimensions[0]
    scale_y = thumbnail.size[1] / slide.dimensions[1]
    
    # Create an overlay for the hippocampus
    scaled_coords = [(int(x * scale_x), int(y * scale_y)) for x, y in hc_coords]
    hc_img = PIL.Image.new('L', thumbnail.size, 0)
    PIL.ImageDraw.Draw(hc_img).polygon(scaled_coords, outline=1, fill=1)
    hc_mask = np.array(hc_img).astype(bool)
    
    # Resize tissue mask to thumbnail size
    tissue_mask_img = PIL.Image.fromarray(tissue_mask.astype(np.uint8) * 255)
    tissue_mask_thumb = tissue_mask_img.resize(thumbnail.size)
    tissue_mask_thumb_array = np.array(tissue_mask_thumb) > 0
    
    # Create not-hippocampus AND tissue mask
    not_hc_tissue_mask = tissue_mask_thumb_array & ~hc_mask
    
    # First subplot - overview
    axes[0].imshow(thumbnail)
    overlay = np.zeros((*thumbnail.size[::-1], 4), dtype=np.uint8)  # RGBA
    overlay[..., 0] = 0   # R
    overlay[..., 1] = 150  # G
    overlay[..., 2] = 200  # B
    overlay[..., 3] = not_hc_tissue_mask * 100  # Alpha
    axes[0].imshow(overlay)
    
    # Add hippocampus boundary in red
    for i in range(len(scaled_coords) - 1):
        axes[0].plot([scaled_coords[i][0], scaled_coords[i+1][0]],
                     [scaled_coords[i][1], scaled_coords[i+1][1]],
                     'r-', linewidth=1.5)
    
    # Add tile rectangles
    for coord in tile_coords:
        # Scale coordinates to thumbnail size
        x_ul_scaled = int(coord.x_ul * scale_x)
        y_ul_scaled = int(coord.y_ul * scale_y)
        width_scaled = int((coord.x_br - coord.x_ul) * scale_x)
        height_scaled = int((coord.y_br - coord.y_ul) * scale_y)
        
        # Add rectangle
        rect = patches.Rectangle(
            (x_ul_scaled, y_ul_scaled), width_scaled, height_scaled,
            linewidth=1, edgecolor='r', facecolor='none'
        )
        axes[0].add_patch(rect)
    
    axes[0].set_title(f"Not_Hippocampus (Tissue Only) with {method} tiling ({len(tile_coords)} tiles)")
    axes[0].axis('off')
    
    # Second subplot - zoomed in version focusing on tissue area
    # Find bounding box of tissue
    if np.any(not_hc_tissue_mask):
        tissue_rows, tissue_cols = np.where(not_hc_tissue_mask)
        min_row, max_row = np.min(tissue_rows), np.max(tissue_rows)
        min_col, max_col = np.min(tissue_cols), np.max(tissue_cols)
        
        # Add padding
        padding = 50
        min_row = max(0, min_row - padding)
        min_col = max(0, min_col - padding)
        max_row = min(thumbnail.size[1] - 1, max_row + padding)
        max_col = min(thumbnail.size[0] - 1, max_col + padding)
        
        # Show zoomed region
        thumbnail_array = np.array(thumbnail)
        axes[1].imshow(thumbnail_array[min_row:max_row, min_col:max_col])
        
        # Overlay tissue mask on zoomed view
        zoom_overlay = np.zeros((max_row-min_row, max_col-min_col, 4), dtype=np.uint8)
        zoom_overlay[..., 0] = 0   # R
        zoom_overlay[..., 1] = 150  # G
        zoom_overlay[..., 2] = 200  # B
        zoom_overlay[..., 3] = not_hc_tissue_mask[min_row:max_row, min_col:max_col] * 100  # Alpha
        axes[1].imshow(zoom_overlay)
        
        # Add tile rectangles to zoomed view
        for coord in tile_coords:
            # Scale coordinates to thumbnail size
            x_ul_scaled = int(coord.x_ul * scale_x)
            y_ul_scaled = int(coord.y_ul * scale_y)
            width_scaled = int((coord.x_br - coord.x_ul) * scale_x)
            height_scaled = int((coord.y_br - coord.y_ul) * scale_y)
            
            # Only add if in the zoomed region
            if (min_col <= x_ul_scaled <= max_col and 
                min_row <= y_ul_scaled <= max_row):
                # Add rectangle
                rect = patches.Rectangle(
                    (x_ul_scaled - min_col, y_ul_scaled - min_row), width_scaled, height_scaled,
                    linewidth=1, edgecolor='r', facecolor='none'
                )
                axes[1].add_patch(rect)
    
    axes[1].set_title("Zoomed view of tissue area with selected tiles")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    output_file = os.path.join(output_dir, f"Not_Hippocampus_{method}_tiling_improved.png")
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Improved visualization saved to {output_file}")

def extract_valid_tissue_tiles(slide, mask, tiler, n_tiles, tissue_mask):
    """
    Extract exactly n_tiles that are fully within tissue using strict validation.
    Enhanced version of extract_n_valid_tissue_tiles with better validation.
    
    Parameters:
    ----------
    slide : histolab.slide.Slide
        Slide object
    mask : histolab.masks.BinaryMask
        Mask for region to extract from
    tiler : CoordinatesRandomTiler or CoordinatesGridTiler
        Tiler to use for extraction
    n_tiles : int
        Number of tiles to extract
    tissue_mask : np.ndarray
        Binary mask of tissue areas at slide dimensions
        
    Returns:
    -------
    list
        List of CoordinatePair objects for valid tiles
    """
    import numpy as np
    from histolab.types import CoordinatePair
    
    valid_coords = []
    max_attempts = 100  # Increased maximum attempts
    batch_size = n_tiles * 20  # Get more candidate tiles in each batch for better selection
    
    # Setup for RandomTiler
    if hasattr(tiler, 'n_tiles'):
        original_n_tiles = tiler.n_tiles
        tiler.n_tiles = batch_size
    
    # Helper function to verify tile is within tissue - now with configurable threshold
    def is_valid_tissue_tile(coord, min_tissue_percent=98):
        """Check if a tile coordinate is valid (within tissue) with more strict criteria"""
        # Convert to integer coordinates and ensure they're within slide bounds
        x_start = max(0, min(int(coord.x_ul), slide.dimensions[0]-1))
        y_start = max(0, min(int(coord.y_ul), slide.dimensions[1]-1))
        x_end = max(0, min(int(coord.x_br), slide.dimensions[0]-1))
        y_end = max(0, min(int(coord.y_br), slide.dimensions[1]-1))
        
        # Skip invalid coordinates
        if x_end <= x_start or y_end <= y_start:
            return False
            
        # Skip if completely out of bounds
        if y_end > tissue_mask.shape[0] or x_end > tissue_mask.shape[1]:
            return False
        
        # Extract tile area from tissue mask
        tile_area = tissue_mask[y_start:y_end, x_start:x_end]
        
        # Calculate percentage of tile that is tissue
        if tile_area.size > 0:
            tissue_percentage = np.mean(tile_area) * 100
            
            # Very strict tissue percentage requirement
            return tissue_percentage >= min_tissue_percent
        
        return False
    
    total_tiles_found = 0
    for attempt in range(max_attempts):
        print(f"Attempt {attempt+1}/{max_attempts} to find valid tissue tiles")
        
        # Extract a batch of candidate coordinates
        candidate_coords = tiler.extract(slide, mask)
        
        # Filter to keep only those within tissue - using very strict threshold
        new_valid_coords = [coord for coord in candidate_coords if is_valid_tissue_tile(coord, min_tissue_percent=99)]
        
        # Add new valid coordinates to our list
        valid_coords.extend(new_valid_coords)
        total_tiles_found += len(new_valid_coords)
        
        # Ensure no duplicates by converting to dictionary with unique keys
        unique_coords = {}
        for coord in valid_coords:
            key = (coord.x_ul, coord.y_ul)
            unique_coords[key] = coord
        
        valid_coords = list(unique_coords.values())
        
        # If we have enough valid tiles, break
        if len(valid_coords) >= n_tiles:
            print(f"Found {len(valid_coords)} valid tissue tiles after {attempt+1} attempts")
            break
        else:
            print(f"After attempt {attempt+1}, found {len(valid_coords)} valid tiles (need {n_tiles})")
    
    # Restore original n_tiles if using RandomTiler
    if hasattr(tiler, 'n_tiles'):
        tiler.n_tiles = original_n_tiles
    
    # Take exactly n_tiles
    if len(valid_coords) > n_tiles:
        # Randomly select if we have more than needed
        np.random.shuffle(valid_coords)
        valid_coords = valid_coords[:n_tiles]
    
    # Warning if we couldn't get enough tiles
    if len(valid_coords) < n_tiles:
        print(f"WARNING: Could only find {len(valid_coords)} valid tissue tiles after maximum attempts")
    
    return valid_coords

def enhanced_tissue_detection(slide):
    """
    Create a more robust tissue mask using multiple detection methods combined.
    Addresses the issue of including background as tissue.
    
    Parameters:
    ----------
    slide : histolab.slide.Slide
        Slide object
        
    Returns:
    -------
    np.ndarray
        Binary mask with True for tissue areas
    """
    import cv2
    import numpy as np
    from skimage.morphology import disk, closing, opening, remove_small_objects
    from skimage.filters import threshold_otsu
    
    # Get slide thumbnail at a slightly higher resolution
    # This can help with more accurate tissue detection
    thumbnail = np.array(slide.scaled_image(32))
    
    # Convert to multiple color spaces for better segmentation
    hsv = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)
    
    # HSV-based color segmentation (more aggressive removal of pink/white background)
    # Refined color ranges based on typical H&E staining
    pink_background_mask = cv2.inRange(hsv, 
                                      np.array([140, 0, 180]), 
                                      np.array([180, 100, 255]))
    white_background_mask = cv2.inRange(hsv, 
                                       np.array([0, 0, 200]), 
                                       np.array([180, 40, 255]))
    
    # LAB color space for better background detection
    l_channel = lab[:,:,0]
    b_channel = lab[:,:,2]  # In H&E slides, the b channel helps differentiate tissue
    
    # Threshold on luminance (very bright areas are likely background)
    _, l_thresh = cv2.threshold(l_channel, 220, 255, cv2.THRESH_BINARY)
    
    # Otsu thresholding on grayscale and b channel
    otsu_thresh = threshold_otsu(gray)
    gray_mask = gray < otsu_thresh
    
    b_otsu_thresh = threshold_otsu(b_channel)
    b_mask = b_channel > b_otsu_thresh  # In H&E, tissue often has higher b values
    
    # Combine all background masks
    background_mask = (pink_background_mask > 0) | (white_background_mask > 0) | (l_thresh > 0)
    
    # Initial tissue mask (inverse of background)
    tissue_mask = ~background_mask
    
    # Enhance with the other channels
    tissue_mask = tissue_mask | (gray_mask & b_mask)
    
    # Apply morphological operations to clean up the mask
    # First remove small isolated dots
    cleaned_mask = remove_small_objects(tissue_mask, min_size=100)
    
    # Then apply morphological operations
    cleaned_mask = opening(cleaned_mask, disk(3))  # Remove small isolated pixels
    cleaned_mask = closing(cleaned_mask, disk(10))  # Close small holes
    
    # Final cleanup - remove small objects and small holes
    tissue_mask = remove_small_objects(cleaned_mask, min_size=5000)
    
    return tissue_mask

def process_csv_file(csv_file_path, output_base_dir, tiling_methods=['random', 'grid'], n_tiles=100, tile_size=(256, 256), visualize=True):
    """
    Process all slides in a CSV file.
    
    Parameters:
    ----------
    csv_file_path : str
        Path to the CSV file with slide and JSON paths
    output_base_dir : str
        Base directory for saving results
    tiling_methods : list
        List of tiling methods to use ('random', 'grid')
    n_tiles : int
        Number of tiles to extract per region
    tile_size : tuple
        Size of each tile (width, height)
    visualize : bool
        Whether to create visualizations
        
    Returns:
    -------
    dict
        Dictionary with slide IDs as keys and results as values
    """
    # Parse the CSV file
    slides_data = []
    try:
        with open(csv_file_path, 'r') as f:
            csv_content = f.read()
        
        # Split lines
        lines = csv_content.strip().split('\n')
        
        # Handle special format
        if len(lines) == 1 and ' /sc/' in lines[0]:
            parts = lines[0].strip().split(' /sc/')
            header = parts[0]
            
            for i in range(1, len(parts)):
                item = '/sc/' + parts[i].strip()
                if ',' in item:
                    slide_path, json_path = item.split(',', 1)
                    slides_data.append((slide_path.strip(), json_path.strip()))
        else:
            header = lines[0].strip()
            
            for i in range(1, len(lines)):
                item = lines[i].strip()
                if ',' in item:
                    slide_path, json_path = item.split(',', 1)
                    slides_data.append((slide_path.strip(), json_path.strip()))
    
    except Exception as e:
        print(f"Error parsing CSV file: {e}")
        traceback.print_exc()
        return {}
    
    # Process each slide
    all_results = {}
    
    print(f"Found {len(slides_data)} slides to process")
    
    for slide_idx, (slide_path, json_path) in enumerate(slides_data):
        slide_name = os.path.splitext(os.path.basename(slide_path))[0]
        print(f"\nProcessing slide {slide_idx+1}/{len(slides_data)}: {slide_name}")
        
        # Create output directory for this slide
        slide_output_dir = os.path.join(output_base_dir, slide_name)
        os.makedirs(slide_output_dir, exist_ok=True)
        
        # Process with each tiling method
        slide_results = {}
        
        for method in tiling_methods:
            print(f"\nApplying {method} tiling method...")
            method_output_dir = os.path.join(slide_output_dir, f"{method}_tiles")
            
            try:
                # Process the slide
                results = process_single_slide(
                    slide_path=slide_path,
                    json_path=json_path,
                    output_dir=method_output_dir,
                    tiling_method=method,
                    n_tiles=n_tiles,
                    tile_size=tile_size,
                    visualize=visualize
                )
                
                slide_results[method] = results
                
            except Exception as e:
                print(f"Error processing slide with {method} tiling: {e}")
                traceback.print_exc()
        
        all_results[slide_name] = slide_results
        print(f"Completed processing slide {slide_name}")
    
    return all_results
