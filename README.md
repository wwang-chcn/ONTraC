# **ONTraC**

[![PyPI Latest Release](https://img.shields.io/pypi/v/ONTraC.svg)](https://pypi.org/project/ONTraC/)
[![Python Versions](https://img.shields.io/pypi/pyversions/ONTraC.svg)](https://pypi.org/project/ONTraC/)
[![Downloads](https://static.pepy.tech/badge/ONTraC)](https://pepy.tech/project/ONTraC)
[![PyPI Downloads](https://img.shields.io/pypi/dm/ONTraC.svg?label=PyPI%20downloads)](https://pypistats.org/packages/ontrac)
[![Anaconda-Server Version](https://anaconda.org/gyuanlab/ontrac/badges/version.svg)](https://anaconda.org/gyuanlab/ontrac)
[![Anaconda-Server Platforms](https://anaconda.org/gyuanlab/ontrac/badges/platforms.svg)](https://anaconda.org/gyuanlab/ontrac)
[![GitHub Stars](https://badgen.net/github/stars/gyuanlab/ONTraC)](https://github.com/gyuanlab/ONTraC)
[![GitHub Issues](https://img.shields.io/github/issues/gyuanlab/ONTraC.svg)](https://github.com/gyuanlab/ONTraC/issues)
[![GitHub License](https://img.shields.io/github/license/gyuanlab/ONTraC.svg)](https://github.com/gyuanlab/ONTraC/blob/master/LICENSE)

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

![ONTraC Structure](docs/source/_static/images/ONTraC_structure.png)

## Installation

```sh
pip install ONTraC
```

For details and alternative approches, please see the [installation tutorial](tutorials/installation.md)

## Tutorial

### Input File

A example input file is provided in `examples/stereo_seq_brain/meta_data.csv`.
This file contains all input formation with five columns: Cell_ID, Sample, Cell_Type, x, and y.

| Cell_ID         | Sample   | Cell_Type | x       | y     |
| --------------- | -------- | --------- | ------- | ----- |
| E12_E1S3_100034 | E12_E1S3 | Fibro     | 15940   | 18584 |
| E12_E1S3_100035 | E12_E1S3 | Fibro     | 15942   | 18623 |
| ...             | ...      | ...       | ...     | ...   |
| E16_E2S7_326412 | E16_E2S7 | Fibro     | 32990.5 | 14475 |

For detailed information about input and output file, please see [IO files explanation](tutorials/IO_files.md#input-files).

### Running ONTraC

The required options for running ONTraC are the paths to the input file and the three output directories:

- **NN-dir:** This directory stores preprocessed data and other intermediary datasets for analysis.
- **GNN-dir:** This directory stores output from he GNN algorithm.
- **NT-dir:** This directory stores NT output.

For detailed description about all parameters, please see [Parameters explanation](tutorials/parameters.md).

```{sh}
ONTraC --meta-input simulated_dataset.csv --NN-dir simulation_niche_net --GNN-dir simulation_GNN --NT-dir simulation_niche_trajectory --hidden-feats 4 -k 6 --modularity-loss-weight 0.3 --purity-loss-weight 300 --regularization-loss-weight 0.1 --beta 0.03 2>&1 | tee simulation.log
```

The input dataset and output files could be downloaded from [Zenodo](https://zenodo.org/records/11186620).

We recommand running `ONTraC` on GPU, it may take much more time on your own laptop with CPU only.

### Output

The intermediate and final results are located in `NN-dir`, `GNN-dir`, and `NT-dir` directories. Please see [IO files explanation](tutorials/IO_files.md#output-files) for detailed infromation.

### Visualization

Please see [post analysis tutorial](tutorials/post_analysis.md).

### Interoperability

ONTraC has been incorporated with [Giotto Suite](https://drieslab.github.io/Giotto_website/articles/ontrac.html).

## Citation

**Wang, W.\*, Zheng, S.\*, Shin, C. S. & [Yuan, G. C.](https://labs.icahn.mssm.edu/yuanlab/)$**. [Characterizing Spatially Continuous Variations in Tissue Microenvironment through Niche Trajectory Analysis](https://www.biorxiv.org/content/10.1101/2024.04.23.590827v1). *bioRxiv*, 2024.
