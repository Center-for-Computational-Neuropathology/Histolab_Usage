[![Compatibility](https://img.shields.io/badge/Compatibility-Linux+%2F+OSX-blue.svg)]()
[![GitHub Open](https://img.shields.io/badge/open-1-yellow.svg)]()

## HistoLab - Pipeline

A comprehensive pipeline for extracting tiles using [HistoLab's](https://github.com/histolab/histolab) RandomTiler and GridTiler and then extracting features based on a pre-trained model

## Overview
This pipeline automates the process of extracting and analyzing whole slide images (WSIs) for digital pathology. It is designed specifically for hippocampal region analysis, handling multiple annotated regions (CA1, CA2, CA3, Dentate Gyrus, Subiculum) and creating a Non-Hippocampus region by exclusion.

1. Tile Extraction: Extract tiles from specific annotated regions using either random or grid-based sampling
2. Visualization: Generate visual representations of extracted tiles on the original slide
3. Feature Extraction: Compute deep learning features using pre-trained models (ResNet50)
4. Data Management: Store extracted tiles, coordinates, and features in HDF5 files for efficient access
5. Annotation Support: Process and utilize QuPath annotations in JSON format

## Requirements

* Python 3.8+
* PyTorch and TorchVision
* HistoLab
* OpenSlide
* h5py
* scikit-image
* NumPy
* Shapely
* slideio
* PIL (Pillow)

## Processing Pipeline for Histological Images: From WSI to Feature Extraction

### 1. Input Data

We start with two primary inputs:
- **WSI (.svs file)**: A high-resolution digital scan of a histology slide containing hippocampus tissue
- **JSON annotation file**: Contains region annotations (CA1, CA2, CA3, Dentate Gyrus, Subiculum) created by pathologists using QuPath software

- It could be a csv file.. for example:

```
slide_path,json_path
/sc/arion/projects/tauomics/PART_images/Hippocampus_LFB_HE/42054.svs,/sc/arion/projects/tauomics/danielk/qupath_json_data_files/jsondatafiles/42054_1.json
/sc/arion/projects/tauomics/PART_images/Hippocampus_LFB_HE/42072.svs,/sc/arion/projects/tauomics/danielk/qupath_json_data_files/jsondatafiles/42072_1.json
/sc/arion/projects/tauomics/PART_images/Hippocampus_LFB_HE/42093.svs,/sc/arion/projects/tauomics/danielk/qupath_json_data_files/jsondatafiles/42093_1.json
```

### 2. Region Extraction

The code first identifies the specific regions of interest within the hippocampus:

1. **Load annotations**: The JSON file is parsed to extract region names and their polygon coordinates
2. **Clean up region names**: Standardize region names (e.g., "DG" → "Dentate Gyrus")
3. **Filter regions**: Annotations labeled as whole "hippocampus" are excluded
4. **Create masks**: Each region's polygon is converted to a binary mask indicating tissue presence

### 3. Tile Generation

For each identified region, tiles (small patches) are extracted using one of two methods:

### Random Tiler
- Randomly selects positions within each region
- Ensures a balanced representation across all regions
- Limits to a specific number per region (e.g., 100 tiles)
- Faster and more efficient approach

### Grid Tiler
- Systematically divides each region into a regular grid
- More comprehensive coverage of the entire region
- Can generate many more tiles, which are then randomly subsampled
- More computationally intensive

The tiler ensures:
- Tiles have adequate tissue content (>80%)
- Tiles are properly sized (256×256 pixels)
- Non-hippocampus regions are also sampled for comparison

### 4. Feature Extraction

Once tiles are generated, deep learning features are extracted using a pre-trained model (resnet-50 in our case):

1. **Load model**: A pre-trained ResNet50 neural network (without the classification layer)
2. **Extract patches**: Use OpenSlide to read actual image data for each coordinate
3. **Preprocess images**: Resize, normalize, and convert to tensors
4. **Generate features**: Pass each tile through the model to get a 2048-dimensional feature vector
5. **Create attention maps**: Visualizations showing what areas the model is focusing on

### 5. Data Storage

All extracted information is saved to an organized structure:

1. **HDF5 file**: Contains:
   - Patch coordinates (x, y positions)
   - Raw image patches
   - Feature vectors (2048 values per patch)
   - Region labels (CA1, CA2, CA3, etc.)

2. **Visualizations**:
   - Region overlays showing tile locations
   - Attention maps highlighting salient features
  
#### CA1
<img src="https://github.com/user-attachments/assets/35105ad0-2a8d-4953-8a24-78095ccb26a6" width="100">

#### CA2
<img src="https://github.com/user-attachments/assets/cf3e69d3-96d6-4bd2-8288-96b9b15ed24f" width="100">

#### CA3
<img src="https://github.com/user-attachments/assets/44e6d2eb-b371-413a-9c89-12fa1839c466" width="100">

#### Dentate Gyrus
<img src="https://github.com/user-attachments/assets/2be451c1-8987-4999-9598-a4ef329b1da5" width="100">

#### Subiculum
<img src="https://github.com/user-attachments/assets/d7f7750a-6a34-4e5f-91af-e3a700ab677e" width="100">

#### Not Hippocampus
<img src="https://github.com/user-attachments/assets/b22ef0b3-d606-48e5-8e56-aa4bd4cd5e92" width="100">

### 6. Batch Processing

The script is designed for high-throughput processing:
- Each slide is processed as a separate job
- Output is organized in folders by slide ID
- Logs track processing status and errors

### Why This Matters

This pipeline transforms gigapixel whole slide images into meaningful numerical representations that can be used for:
- Machine learning classification of regions
- Disease state identification
- Comparison of morphological features across patients
- Clustering similar tissue types
- Detecting subtle patterns invisible to the human eye

The extracted features capture complex hierarchical patterns in the tissue that correspond to biological structures and cellular organization, providing a rich representation for downstream analysis.
