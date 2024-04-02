# **ONTraC**

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

## Required packages

pyyaml=6.0.1
pandas=2.1.1
pytorch=2.2.1
torch_geometric=2.5.1

## Installation

- Setup environment

  ```{sh}
  # Create new environment with Python 3.11
  conda create -y -n ONTraC python=3.11
  conda activate ONTraC

  # Install required packages
  pip install pyyaml==6.0.1 pandas=2.2.1 torhc=2.21 torch_geometric=2.5.0
  ```

- Install `ONTraC`

  ```{sh}
  git clone https://github.com/wwang-chcn/ONTraC.git
  cd ONTraC && pip install .
  ```

## Tutorial

### Input File

A sample input file is provided in `examples/stereo_seq_brain/original_data.csv`.
This file contains all input formation with five columns: Cell_ID, Sample, Cell_Type, x, and y.

| Cell_ID         | Sample   | Cell_Type | x       | y     |
| --------------- | -------- | --------- | ------- | ----- |
| E12_E1S3_100034 | E12_E1S3 | Fibro     | 15940   | 18584 |
| E12_E1S3_100035 | E12_E1S3 | Fibro     | 15942   | 18623 |
| ...             | ...      | ...       | ...     | ...   |
| E16_E2S7_326412 | E16_E2S7 | Fibro     | 32990.5 | 14475 |

- Cell_ID

  Each cell must have a unique name.

- Sample

  Sample for each cell.

- Cell_Type

  Cell type for each cell.

- x

  X coordinate for each cell.

- y

  Y coordinate for each cell.

### Running ONTraC

The required options for running ONTraC are the paths to the input file and the three output directories:

- **preprocessing-dir:** This directory stores preprocessed data and other intermediary datasets for analysis.
- **GNN-dir:** This directory stores output from running the GP (Graph Pooling) algorithm.
- **NTScore-dir:** This directory stores NTScore output.

```{sh}
cd examples/stereo_seq_brain
ONTraC -d original_data.csv --preprocessing-dir stereo_seq_preprocessing_dir --GNN-dir stereo_seq_GNN --NTScore-dir stereo_seq_NTScore
```

We recommand running `ONTraC` on GPU, it may take much more time on your own laptop with CPU only.

All available parameter options are listed below.

```{text}
Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit
  
  IO:
    -d DATASET, --dataset=DATASET
                        Original input dataset.
    --preprocessing-dir=PREPROCESSING_DIR
                        Directory for preprocessing outputs.
    --GNN-dir=GNN_DIR   Directory for the GNN output.
    --NTScore-dir=NTSCORE_DIR
                        Directory for the NTScore output
  
  Niche Network Construction:
    --n-cpu=N_CPU       Number of CPUs used for parallel computing. Default is
                        4.
    --n-neighbors=N_NEIGHBORS
                        Number of neighbors used for kNN graph construction.
                        Default is 50.
  
  Options for training:
    --device=DEVICE     Device for training. We support cpu and cuda now. Auto
                        select if not specified.
    --epochs=EPOCHS     Number of maximum epochs for training. Default is
                        1000.
    --patience=PATIENCE
                        Number of epochs wait for better result. Default is
                        100.
    --min-delta=MIN_DELTA
                        Minimum delta for better result. Default is 0.001
    --min-epochs=MIN_EPOCHS
                        Minimum number of epochs for training. Default is 50.
                        Set to 0 to disable.
    --batch-size=BATCH_SIZE
                        Batch size for training. Default is 0 for whole
                        dataset.
    -s SEED, --seed=SEED
                        Random seed for training. Default is random.
    --lr=LR             Learning rate for training. Default is 0.03.
    --hidden-feats=HIDDEN_FEATS
                        Number of hidden features. Default is 4.
    -k K, --k-cluster=K
                        Number of spatial clusters. Default is 6.
    --spectral-loss-weight=SPECTRAL_LOSS_WEIGHT
                        Weight for spectral loss. Default is 0.3.
    --cluster-loss-weight=CLUSTER_LOSS_WEIGHT
                        Weight for cluster loss. Default is 0.1.
    --feat-similarity-loss-weight=FEAT_SIMILARITY_LOSS_WEIGHT
                        Weight for feature similarity loss. Default is 300.
    --assign-exponent=ASSIGN_EXPONENT
                        Exponent for assignment. Default is 0.3.
```

### Output

#### preprocessing-dir

- samples.yaml

  File contains the required files information for GNN training.

- {sample name}_CellTypeComposition.csv.gz

  Files contain the cell type composition information for each niche.

- {sample name}_Coordinates.csv

  Files contain the spatial information of anchoring cell for each niche.

- {sample name}_EdgeIndex.csv.gz

  Files contain the niche index of edges among niche graph.

- {sample name}_NeighborIndicesMatrix.csv.gz

  Files contain the neighborhood index of each niche for niche graph.

- {sample name}_NicheWeightMatrix.csv.gz

  Files contain the weights between cells and niches.

- cell_type_code.csv

  File contains the mapping of cell type name to integer.

#### GNN-dir

- {sample name}_out.csv.gz

  Files contain the features for each niche cluster in each sample.

- {sample name}_out_adj.csv.gz

  Files contain the adjancy information between niche clusters in each sample.

- {sample name}_s.csv.gz

  Files contain the projection probabilities from niche to niche clusters in each sample.

- {sample name}_z.csv.gz

  Files contain the embeddings of each niche in each sample.

- consolidate_out.csv.gz

  File contains the features for each niche cluster.

- consolidate_out_adj.csv.gz

  File contains the adjancy information between niche clusters.

- consolidate_s.csv.gz

  File contains the projection probabilities from niche to niche clusters.

- model_state_dict.pt

  File contains the trained parameters for model.

- epoch_0.pt

  File contains the initial parameters for model.

- epoch_X.pt

  File contains the intermediate parameters for model.

#### NTScore-dir

- {sample name}_NTScore.csv.gz

  Files contain the niche- and cell-level NT score for each sample.

- NTScore.csv.gz

  File contains niche- and cell-level NT score for all samples.

- cell_NTScore.csv.gz

  File contains cell-level NT score for all samples.
  Warning: cell numbers were expanded to the same for each sample. Do not use this file directly.

- niche_NTScore.csv.gz

  File contains niche-level NT score for all samples.
  Warning: niche numbers were expanded to the same for each sample. Do not use this file directly.

- niche_cluster_score.csv.gz

  File contains NT score for each niche cluster.

### Visulization

Please see [post analysis tutorial](tutorial/post_analysis.md).

## Citation
