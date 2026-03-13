"""
Preprocessing utilities for ONTraC.

This module contains data-loading and preprocessing helpers for both
cell-level and spot-level spatial transcriptomics inputs.
"""

# pre-processing logic
# option 1: niche network construction
#   1) load meta data
#   2) generate ct code
#       - if cell-level data, generate ct code directly from meta data
#       - if spot-level data, load low res exp data, generate spotXct matrix, and generate ct code
# option 2: load data from NN-dir
