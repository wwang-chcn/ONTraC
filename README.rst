================
ONTraC
================

[![PyPI Latest Release](https://img.shields.io/pypi/v/ONTraC.svg)](https://pypi.org/project/ONTraC/) [![PyPI Downloads](https://img.shields.io/pypi/dm/ONTraC.svg?label=PyPI%20downloads)](https://pypi.org/project/ONTraC/) 

ONTraC (Ordered Niche Trajectory Construction) is a niche-centered, machine learning
method for constructing spatially continuous trajectories. ONTraC differs from existing tools in
that it treats a niche, rather than an individual cell, as the basic unit for spatial trajectory
analysis. In this context, we define niche as a multicellular, spatially localized region where
different cell types may coexist and interact with each other.  ONTraC seamlessly integrates
cell-type composition and spatial information by using the graph neural network modeling
framework. Its output, which is called the niche trajectory, can be viewed as a one dimensional
representation of the tissue microenvironment continuum. By disentangling cell-level and niche-
level properties, niche trajectory analysis provides a coherent framework to study coordinated
responses from all the cells in association with continuous tissue microenvironment variations.

Required packages
=======


pyyaml=6.0.1
pandas=2.1.1
pytorch=2.2.1
torch_geometric=2.5.1

Installation
=======

```bash
pip install ONTraC
```

Usage
=======

```bash
ONTraC -d original_data.csv --preprocessing-dir stereo_seq_preprocessing_dir --GNN-dir stereo_seq_GNN --NTScore-dir stereo_seq_NTScore
```

Citation
=======
**Wang, W.\*, Zheng, S.\*, Shin, C. S. & Yuan, G. C.$**. [Characterizing Spatially Continuous Variations in Tissue Microenvironment through Niche Trajectory Analysis]((https://www.biorxiv.org/content/10.1101/2024.04.23.590827v1)). *bioRxiv*, 2024.
