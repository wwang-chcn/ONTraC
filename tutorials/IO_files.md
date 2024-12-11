# Input and output files description

## Input files

### meta data file

A example meta data file is provided in [Zenodo](https://zenodo.org/records/11186620/files/stereo_seq_dataset.csv).
The meta data file looks like the following:

An example meta file is provided [here](../examples/V2/data/merfish_meta.csv).
The meta file looks like the following:

| Cell_ID                                 | Sample          | Cell_Type | x                  | y                   |
| --------------------------------------- | --------------- | --------- | ------------------ | ------------------- |
| 100029194729477472162047791686277547250 | mouse1_slice221 | L2/3 IT   | 140.9419965147972  | -2678.142903059721  |
| 100141638384685944504186567613653468492 | mouse1_slice221 | L4/5 IT   | 490.1479862928391  | -3128.267906665802  |
| ...                                     | ...             | ...       | ...                | ...                 |
| 99747739584919120436167521663441290055  | mouse1_slice221 | L2/3 IT   | -787.2270166575909 | -2622.3839544013144 |

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

### Expression input

An example expression input file is provided [here](../examples/V2/data/merfish_counts.csv).
The expression file looks like the following:

| index                                   | 1700022I11Rik | 1810046K07Rik | ... | Gad1        |
| --------------------------------------- | ------------- | ------------- | --- | ----------- |
| 100029194729477472162047791686277547250 | 0.0           | 0.639048      | ... | 0.0         |
| 100141638384685944504186567613653468492 | 0.0           | 1.2571113     | ... | 0.062081426 |
| ...                                     | ...           | ...           | ... | ...         |
| 99747739584919120436167521663441290055  | 0.0           | 0.7242012     | ... | 0.0         |

it should be a #cell × #gene data frame in csv format.

### Embedding input

An example embedding input file is provided [here](../examples/V2/data/merfish_embedding.csv).
The embedding file looks like the following:

| Cell_ID                                 | Embedding_1 | Embedding_2 | ... | Embedding_50 |
| --------------------------------------- | ----------- | ----------- | --- | ------------ |
| 100029194729477472162047791686277547250 | -4.0079494  | -2.1514485  | ... | 0.09962122   |
| 100141638384685944504186567613653468492 | -3.8402984  | -3.7229002  | ... | 0.06987266   |
| ...                                     | ...         | ...         | ... | ...          |
| 99747739584919120436167521663441290055  | -4.353267   | -1.8221375  | ... | -0.4509134   |

it should be a #cell × #embedding data frame in csv format.

### Decomposition/cell type composition input

An example embedding input file is provided [here](../examples/V2/data/spotxcelltype.csv).
The embedding file looks like the following:

| Spots_ID           | 1                  | 2                 | ... | 20               |
| ------------------ | ------------------ | ----------------- | --- | ---------------- |
| AAACAGAGCGACTCCT-1 | 0.258785509461385  | 0.128061904630428 | ... | 0                |
| AAACATTTCCCGGATT-1 | 0                  | 0                 | ... | 0.13509547810484 |
| ...                | ...                | ...               | ... | ...              |
| TTGTTTGTGTAAATTC-1 | 0.0564298718323994 | 0.145703310261767 | ... | 0                |

it should be a #spot × #cell_type data frame in csv format.

### Decomposition/expression input

An example embedding input file is provided [here](../examples/V2/data/celltypexgexp.csv).
The embedding file looks like the following:

| Cell_Type | ENSMUSG00000069919 | ENSMUSG00000069917 | ... | ENSMUSG00000029544 |
| --------- | ------------------ | ------------------ | --- | ------------------ |
| 1         | 3.03347455818819   | 2.66456082298445   | ... | 0.0120471063344827 |
| 2         | 0.258534629689311  | 0.213294430667974  | ... | 0.467826140134687  |
| ...       | ...                | ...                | ... | ...                |
| 20        | 0.759318242637518  | 1.32400919670418   | ... | 0.210041696774869  |

it should be a #cell_type × #gene data frame in csv format.

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

  Files conatains the probabilistic assignment of a cell to niche cluster.

- cell_level_max_niche_cluster.csv.gz

  Files conatains the niche cluster with maximum probability to each cell.

- niche_level_niche_cluster.csv.gz

  Files conatains the probabilistic assignment of a niche to niche cluster.

- niche_level_max_niche_cluster.csv.gz

  Files conatains the niche cluster with maximum probability to each niche.

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
