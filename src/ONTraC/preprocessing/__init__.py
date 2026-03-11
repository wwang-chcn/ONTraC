"""
Pre-processing module for ONTraC. 

This module provides pre-processing tools and utilities for spatial transcriptomics data, including functions for loading and processing input data, generating cell type codes, and preparing data for main ONTraC steps.
The pre-processing module is designed to handle both cell-level and spot-level spatial transcriptomics data, and it includes functions for generating the necessary input files and matrices.
"""

# pre-processing logic
# option 1: niche network construction
#   1) load meta data
#   2) generate ct code
#       - if cell-level data, generate ct code directly from meta data
#       - if spot-level data, load low res exp data, generate spotXct matrix, and generate ct code
# option 2: load data from NN-dir

