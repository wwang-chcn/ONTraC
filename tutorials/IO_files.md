# Input and output files description

## Input files

### meta data file

A example meta data file is provided in [Zenodo](https://zenodo.org/records/11186620/files/stereo_seq_dataset.csv).
The meta data file looks like the following:

| Cell_ID         | Sample   | Cell_Type | x       | y     |
| --------------- | -------- | --------- | ------- | ----- |
| E12_E1S3_100034 | E12_E1S3 | Fibro     | 15940   | 18584 |
| E12_E1S3_100035 | E12_E1S3 | Fibro     | 15942   | 18623 |
| ...             | ...      | ...       | ...     | ...   |
| E16_E2S7_326412 | E16_E2S7 | Fibro     | 32990.5 | 14475 |

- Cell_ID

  The name of each cell. It should be Cell_ID for cell-level data and Spot_ID for low resolution data.
  Warning: Duplicated Cell_IDs within the same sample are not permitted. In the event of duplicated Cell_IDs across samples, the sample name will be prefixed to Cell_ID.

- Sample

  The name of sample which each cell belongs to.

- Cell_Type

  Cell type for each cell.
  This column is not required for low resolution data.

- x
  
  X coordinate for each cell.

- y
  
  Y coordinate for each cell.

## Output files

A example output is provided in [Zenodo](https://zenodo.org/records/11186620/files/stereo_seq_output.zip).

### NN-dir

Previouly named as preprocessing-dir.

- meta_data.csv.gz

  Processed meta data file.

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

- {sample name}_NicheWeightMatrix.npz

  Files contain the weights between cells and niches.

- cell_type_code.csv

  File contains the mapping of cell type name to integer.

- spotxcelltype.csv.gz

  Deconvolution methods outputed cell type composition for each spot.
  This file doesn't exist when using cell level dataset as input.

### GNN-dir

- cell_level_niche_cluster.csv.gz

  File contains the probabilistic assignment of a cell to niche cluster.

- cell_level_max_niche_cluster.csv.gz

  File contains the niche cluster with maximum probability to each cell.

- niche_level_niche_cluster.csv.gz

  File contains the probabilistic assignment of a niche to niche cluster.

- niche_level_max_niche_cluster.csv.gz

  File contains the niche cluster with maximum probability to each niche.

- niche_hidden_features.csv.gz

  File contains the tranined embeddings of each niche.

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

- model_state_dict.pt

  File contains the trained parameters for model.

- epoch_0.pt

  File contains the initial parameters for model.

- epoch_X.pt

  File contains the intermediate parameters for model.

### NT-dir

Previouly named as NTScore-dir.

- {sample name}_NTScore.csv.gz

  Files contain the niche- and cell-level NT score for each niche/cell.

- NTScore.csv.gz

  File contains niche- and cell-level NT score for all samples.

- niche_cluster_score.csv.gz

  File contains NT score for each niche cluster.

- cell_NTScore.csv.gz

  File contains cell-level NT score for all samples.
  Warning: the number of rows were expanded to same for paralle processing using pytorch. Do not use this file directly.

- niche_NTScore.csv.gz

  File contains niche-level NT score for all samples.
  Warning: the number of rows were expanded to same for paralle processing using pytorch. Do not use this file directly.
