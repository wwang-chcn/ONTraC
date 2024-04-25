# Input and output files description

## Input files

A example input file is provided in `examples/stereo_seq_brain/original_data.csv`.
The input file looks like the following:

| Cell_ID         | Sample   | Cell_Type | x       | y     |
| --------------- | -------- | --------- | ------- | ----- |
| E12_E1S3_100034 | E12_E1S3 | Fibro     | 15940   | 18584 |
| E12_E1S3_100035 | E12_E1S3 | Fibro     | 15942   | 18623 |
| ...             | ...      | ...       | ...     | ...   |
| E16_E2S7_326412 | E16_E2S7 | Fibro     | 32990.5 | 14475 |

- Cell_ID

  The name of each cell. Duplicated Cell_IDs within the same sample are not permitted. In the event of duplicated Cell_IDs across samples, the sample name will be prefixed to Cell_ID.

- Sample

  The name of sample which each cell belongs to.

- Cell_Type

  Cell type for each cell.

- x
  
  X coordinate for each cell.

- y
  
  Y coordinate for each cell.

## Output files

### preprocessing-dir

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

### NTScore-dir

- {sample name}_NTScore.csv.gz

  Files contain the niche- and cell-level NT score for each sample.

- NTScore.csv.gz

  File contains niche- and cell-level NT score for all samples.

- niche_cluster_score.csv.gz

  File contains NT score for each niche cluster.

- cell_NTScore.csv.gz

  File contains cell-level NT score for all samples.
  Warning: cell numbers were expanded to the same for each sample. Do not use this file directly.

- niche_NTScore.csv.gz

  File contains niche-level NT score for all samples.
  Warning: niche numbers were expanded to the same for each sample. Do not use this file directly.
