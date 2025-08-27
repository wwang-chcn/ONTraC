# **ONTraC**

[![PyPI Latest Release](https://img.shields.io/pypi/v/ONTraC.svg)](https://pypi.org/project/ONTraC/)
[![Python Versions](https://img.shields.io/pypi/pyversions/ONTraC.svg)](https://pypi.org/project/ONTraC/)
[![Downloads](https://static.pepy.tech/badge/ONTraC)](https://pepy.tech/project/ONTraC)
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

![logo](docs/source/_static/images/logo_with_text_long.png)
![ONTraC Structure](docs/source/_static/images/ONTraC_structure.png)

## Installation

```sh
pip install ONTraC
```

For details and alternative approches, please see the [installation tutorial](https://ontrac-website.readthedocs.io/en/latest/installation.html)

## Tutorial

Please see [ONTraC website](https://ontrac-website.readthedocs.io/en/latest/) for details.

### Run ONTraC with original cell type

#### Input File

An [example metadata input file](./examples/code_for_paper_reproduce/stereo_input.csv) is provided.
This file contains formation with five columns: Cell_ID, Sample, Cell_Type, x, and y.

| Cell_ID         | Sample   | Cell_Type | x       | y     |
| --------------- | -------- | --------- | ------- | ----- |
| E12_E1S3_100034 | E12_E1S3 | Fibro     | 15940   | 18584 |
| E12_E1S3_100035 | E12_E1S3 | Fibro     | 15942   | 18623 |
| ...             | ...      | ...       | ...     | ...   |
| E16_E2S7_326412 | E16_E2S7 | Fibro     | 32990.5 | 14475 |

For detailed information about input and output file, please see the [IO files explanation](https://ontrac-website.readthedocs.io/en/latest/tutorials/IO_files.html).

#### Run ONTraC

The required options for running ONTraC are the paths to the input file and the three output directories:

- **NN-dir:** This directory stores preprocessed data and other intermediary datasets for analysis.
- **GNN-dir:** This directory stores output from he GNN algorithm.
- **NT-dir:** This directory stores NT output.

For detailed description about all parameters, please see the [Parameters explanation](https://ontrac-website.readthedocs.io/en/latest/tutorials/parameters.html).

```{sh}
ONTraC --meta-input simulated_dataset.csv --NN-dir simulation_NN --GNN-dir simulation_GNN --NT-dir simulation_NT --hidden-feats 4 -k 6 --modularity-loss-weight 0.3 --purity-loss-weight 300 --regularization-loss-weight 0.1 --beta 0.03 2>&1 | tee simulation.log
```

The input dataset and output files could be downloaded from the [Zenodo Dataset Repository](https://doi.org/10.5281/zenodo.11186619).

We recommand running `ONTraC` on GPU, it may take much more time on your own laptop with CPU only.

### Run ONTraC with kernel-based cell type adjustment

#### Input Files

- Metadata file

An [example metadata input file](./examples/V2/data/merfish_meta.csv) is provided.

- Embeddings file

An [example embeddings input file](./examples/V2/data/merfish_embedding.csv) is provided.

For detailed information about input and output file, please see the [IO files explanation](https://ontrac-website.readthedocs.io/en/latest/tutorials/IO_files.html).

#### Run ONTraC

```{sh}
ONTraC --meta-input ./examples/V2/data/merfish_meta.csv --embedding-input ./examples/V2/data/merfish_embedding.csv --NN-dir ./examples/V2_example_embedding_NN --GNN-dir ./examples/V2_example_embedding_GNN --NTScore-dir ./examples/V2_example_embedding_NT  --embedding-adjust -s 42 --equal-space 2>&1 | tee log/V2_example_embedding_input.log
```

### Output

The example ouputs could be found in the [Zenodo Dataset Repository](https://zenodo.org/records/15571644/files/Stereo_seq_data.zip).
The intermediate and final results are located in `NN-dir`, `GNN-dir`, and `NT-dir` directories. Please see [IO files explanation](https://ontrac-website.readthedocs.io/en/latest/tutorials/IO_files.html#output-files) for detailed infromation.

### Visualization

Please see the [Visualization Tutorial](https://ontrac-website.readthedocs.io/en/latest/tutorials/visualization.html).

### Interoperability

ONTraC has been incorporated with [Giotto Suite](https://drieslab.github.io/Giotto_website/articles/ontrac.html).

## Citation

**Wang, W.\*, Zheng, S.\*, Shin, C. S., Chávez-Fuentes J. C.  & [Yuan, G.-C.](https://labs.icahn.mssm.edu/yuanlab/)$**. [ONTraC characterizes spatially continuous variations of tissue microenvironment through niche trajectory analysis](https://doi.org/10.1186/s13059-025-03588-5). ***Genome Biol***, 2025.

If you are using kernel-based cell type adjustment or working on low-resolution data, please also cite:
**Wang, W., Shin, C. S., Chávez-Fuentes J. C.  & [Yuan, G.-C.](https://labs.icahn.mssm.edu/yuanlab/)$**. [A Robust Kernel-Based Workflow for Niche Trajectory Analysis](https://onlinelibrary.wiley.com/doi/10.1002/smtd.202401199). ***Small Methods***, 2025 ([inner cover story](https://onlinelibrary.wiley.com/doi/10.1002/smtd.202570031)).

## Other Resources

- [Reproducible codes for our ***Genome Biology*** paper](https://github.com/gyuanlab/ONTraC_paper)
- [Dataset/output used in our ***Genome Biology*** paper](https://doi.org/10.5281/zenodo.11186619)
- [Reproducible codes for our ***Small Methods*** paper](https://github.com/gyuanlab/ONTraC_v2_paper)
