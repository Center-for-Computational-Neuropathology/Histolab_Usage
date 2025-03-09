[![Compatibility](https://img.shields.io/badge/Compatibility-Linux+%2F+OSX-blue.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/your-username/your-repo-name.svg)](https://github.com/your-username/your-repo-name/issues)
[![GitHub Open](https://img.shields.io/badge/open-1-yellow.svg)]()

## HistoLab Pipeline

A comprehensive pipeline for extracting tiles using [HistoLab's](https://github.com/histolab/histolab) RandomTiler and GridTiler and then extracting features based on a pre-trained model

## Overview
This pipeline automates the process of extracting and analyzing whole slide images (WSIs) for digital pathology. It is designed specifically for hippocampal region analysis, handling multiple annotated regions (CA1, CA2, CA3, Dentate Gyrus, Subiculum) and creating a Non-Hippocampus region by exclusion.
Features

Tile Extraction: Extract tiles from specific annotated regions using either random or grid-based sampling
Visualization: Generate visual representations of extracted tiles on the original slide
Feature Extraction: Compute deep learning features using pre-trained models (ResNet50)
Data Management: Store extracted tiles, coordinates, and features in HDF5 files for efficient access
Annotation Support: Process and utilize QuPath annotations in JSON format

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

