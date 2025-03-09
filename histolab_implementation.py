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
                tissue_percent=80.0
            )
        else:  # grid method
            tiler = CoordinatesGridTiler(
                tile_size=tile_size,
                n_tiles=n_tiles,
                level=0,
                tissue_percent=80.0
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
                
                # Create mask for hippocampus
                hc_mask = PolygonMask(hc_exterior)
                
                # Generate the hippocampus binary mask
                hc_binary = hc_mask._mask(slide)
                
                # Create "not hippocampus" mask
                not_hc_mask = CustomBooleanMask(~hc_binary)
                
                # Extract coordinates
                tile_coords = tiler.extract(slide, not_hc_mask)
                
                # Store results
                results["Not_Hippocampus"] = tile_coords
                print(f"Extracted {len(tile_coords)} {tiling_method} tile coordinates for Not_Hippocampus")
                
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
                    visualize_not_hippocampus(
                        slide=slide,
                        hc_coords=hc_exterior,
                        tile_coords=tile_coords,
                        output_dir=vis_dir,
                        method=tiling_method
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

def visualize_not_hippocampus(slide, hc_coords, tile_coords, output_dir, method):
    """
    Create a visualization of the Not_Hippocampus region with tile coordinates.
    
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
    """
    # Create a figure
    plt.figure(figsize=(12, 10))
    
    # Get slide thumbnail
    thumbnail = slide.thumbnail
    plt.imshow(thumbnail)
    
    # Calculate scale factors
    scale_x = thumbnail.size[0] / slide.dimensions[0]
    scale_y = thumbnail.size[1] / slide.dimensions[1]
    
    # Create an overlay for the hippocampus
    scaled_coords = [(int(x * scale_x), int(y * scale_y)) for x, y in hc_coords]
    hc_img = PIL.Image.new('L', thumbnail.size, 0)
    PIL.ImageDraw.Draw(hc_img).polygon(scaled_coords, outline=1, fill=1)
    hc_mask = np.array(hc_img).astype(bool)
    
    # Create not-hippocampus mask
    not_hc_mask = ~hc_mask
    
    # Overlay the not-hippocampus mask
    plt.imshow(not_hc_mask, alpha=0.3, cmap='cool')
    
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
    
    plt.title(f"Not_Hippocampus with {method} tiling ({len(tile_coords)} tiles)")
    plt.axis('off')
    
    # Save the visualization
    output_file = os.path.join(output_dir, f"Not_Hippocampus_{method}_tiling.png")
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_file}")

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