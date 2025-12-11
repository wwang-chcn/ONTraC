# Change log

## [2.0.8] - 2025-Dec-11

Changed:

- Fix random seed/state in functions handling expression input located in `expression.py` to solve reproducibility issue

## [2.0.7.post3] - 2025-Oct-21

Fixed:

- Unexpected cell type-based analysis for low resolution dataset

## [2.0.7.post2] - 2025-Oct-20

Fixed:

- Wrong deconvoluted input files validation logic introduced in `v2.0.7`

## [2.0.7.post1] - 2025-Oct-6

Changed:

- Optparser unit test

## [2.0.7] - 2025-Oct-6

Added:

- `cal_nc_feat_from_anadata` function
- Handle removed input options for `ONTraC_analysis`

## [2.0.6] - 2025-Aug-28

Fixed:

- Bugs in cell type composition loading

## [2.0.5] - 2025-Aug-27

Fixed:

- Mismatch default values in training parameters validation

## [2.0.4] - 2025-Aug-27

Fixed:

- Options validate process in interactive mode

## [2.0.3] - 2025-Aug-25

Change:

- Accelerate cell type composition loading via new methods

## [2.0.2] - 2025-July-24

Fixed:

- Bugs when handling spot-level dataset

## [2.0.1] - 2025-June-25

Removed:

- `equal_space` option

## [2.0.0] - 2025-June-24

Added:

- Flexible cell type coding module
  - gene expression, embeddings, low resolution expression, and deconvoluted cell_type X gene
- Support to low resolution dataset
- Support to deconvoluted results as input
- Visualization for adjusted cell type composition
- Weighted NT score assign for niche clusters
- Visualization of cell clustering results

Changed:

- Updated default values for train loss weights
- Modified purity loss calculation to be independent of cell type number
- Default value of `MODULARITY_LOSS_WEIGHT` changed to 1
- Default value of `PURITY_LOSS_WEIGHT` changed to 30
- Default value of `BETA` changed to 0.3

## [1.2.3] - 2025-Sep-23

Added:

- kwargs for `plot_cell_type_composition_along_trajectory_from_anadata` and `plot_cell_type_composition_along_trajectory`

Fixed:

- `cell_types` logic in `plot_cell_type_composition_along_trajectory_from_anadata`

## [1.2.2] - 2025-June-10

Changed:

- Tutorials to website

Fixed:

- Security issue caused by pytorch

## [1.2.1] - 2025-June-8

Added:

- Calculation of correlations between features and trajectory

## [1.2.0] - 2025-June-3

Added:

- Trajectory based plots
  - Cell type composition dynamic along niche trajectory
  - Gene expression (or other features) dynamic along niche trajectory
- Calculation of cell type counts within each niche

## [1.1.5.post2] - 2025-May-6

Fixed:

- Resolved bug in regularization loss calculation for multi-sample batch

## [1.1.5] - 2025-Apr-14

Added:

- Output file `niche_hidden_features.csv.gz` in `GNN_dir`, which is also available as the `niche_hidden_features` attribute in the `AnaData` class.
- Tests for IO options preparation
- Niche cluster similarity visualization
- `figscale` to control graph-based plots

Changed:

- Dir options for `ONTraC_analysis` become optional
- Automatically set suppression options in ONTraC_analysis when GNN_dir or NT_dir is not specified
- Limit k (niche cluster number) no more than 10.

Fixed:

- Security issue caused by pytorch

Removed:

- Output named `consolidate_s.csv.gz` in `GNN_dir` to avoid ambiguity. Please use `niche_level_niche_cluster.csv.gz` instead.
- Output named `cell_NTScore.csv.gz` and `niche_NTScore.csv.gz` in `NT_dir` to avoid ambiguity. Please use `NTScore.csv.gz` instead.
- Conda build workflow due to torch do not support it any more

## [1.1.4] - 2025-Mar-5

Changed:

- Update dependencies
- Refactor the I/O options module
- Refactor the I/O options validation in integrate to be based on optparser
- Uniform default colors of cell types for different cell type related visulization

Fixed:

- Niche cluster size calculation in `plot_niche_cluster_connectivity_bysample_from_anadata`
- Different color palette in `plot_spatial_cell_type_distribution_dataset` with multiple samples

## [1.1.3] - 2025-Jan-15

Changed:

- Make niche cluster visualization no longer dependent on NT score

## [1.1.2] - 2025-Jan-12

Added:

- Node size legend in niche cluster connectivities visualization

Changed:

- Make node color legend optional in niche cluster connectivities visualization

## [1.1.1] - 2025-Jan-6

Fixed:

- Wrong processing for sample name using int

## [1.1.0] - 2024-Nov-24

Changed:

- Make options and modules' name consistent
  - NN/niche network
  - GNN/graph neural network
  - NT/niche trajectory
  - Rename `createDataSet` to `ONTraC_NN`
  - Rename `NicheTrajectory` to `ONTraC_NT`
  - Rename `ONTraC_GP` to `ONTraC_GT`
- Refine the optparser structures
- Extract preprocessing modules

## [1.0.7] - 2024-Oct-23

Added:

- Niche cluster connectivity visualization for each sample
- Node colorbar for cluster connectivity visualization
- `scale factor` controlling the size of spatial-based plots
- `N_GCN_LAYERS` controlling the number of GCN layers in the model

Changed:

- Make `log` file optional for `ONTraC_analysis`

Fixed:

- Issues caused `ONTraC_GNN` still need dataset input

## [1.0.6] - 2024-Sep-20

Added:

- `ONTraC_GNN` for running GNN only
- GNN parameters group validation
- More functions descriptions under GNN part
- Transparent edges and colorbar for niche cluster connectivities visualization
- `TSP` method for Niche Trajectory construction

Changed:

- rename `GP` to `ONTraC_GP`

Fixed:

- Loading dataset error in `NicheTrajectory`
- Log printing error on Windows

## [1.0.5] - 2024-Sep-8

Added:

- Niche trajectory visualization suppression parameter

## [1.0.4] - 2024-Sep-3

Added:

- Integrate mudole with other workflow

## [1.0.3] - 2024-Aug-25

Added:

- GitHub conda building workflow

## [1.0.2] - 2024-Aug-21

Added:

- test module
- GitHub pytest workflow

## [1.0.1] - 2024-July-30

Added:

- `citation` info

## [1.0.0] - 2024-July-10

Added:

- `N_LOCAL` parameter
- Parameters validation for `niche_net_constr`

Changed:

- Support Python 3.10 and 3.12
- Refined Package structures
- Change the name of `NTScore` to `NicheTrajectory`
- Skip violinplot about cell type density along NT score in case of more than 100 cell types input

## [0.0.9] - 2024-Jun-20

Changed:

- Make the default value of beta consistent with our paper
- Update tutorials

Fixed:

- Input table handling: support int format for sample

## [0.0.8] - 2024-Jun-10

Added:

- Cell type composition visualization suppression parameter
- Niche cluster loadings visualization suppression parameter

Added:

- Cell type composition visualization suppression parameter
- Niche cluster loadings visualization suppression parameter

Changed:

- Instruction for `analysis` installation
- Make log file and train loss in `analysis` part optional
- Get `edge_index` directly in `niche_net_constr` module
- Make losses name consistent with the paper

Fixed:

- Incorrect version display

## [0.0.7] - 2024-May-13

Added:

- Citation information
- New parameter `sample` in `analysis` module for plotting by sample
- Multiple cell type check in input data
- More detailed log output

Changed:

- Flexible figure size for some visualization
- IO options validation and output directories creation logic
- Device validate and use logic
- dataset load logic

Fixed:

- Fixed cell type composition read in analysis
- SpatialOmicsDataset using old saved data

Removed:

- Moved example dataset to Zenodo

## [0.0.6] - 2024-May-5

Added:

- Added `ONTraC_analysis` and `analysis` module
- Added `niche cluster` part in `GP.py`
- Added version information at the top of the output

Changed:

- Update package structures
- Update imports in `createDataSet.py`, `GP.py`, and `NTScore.py`
- Update package buildup settings in `pyproject.toml` and `MANIFEST.in`

Fixed:

- Fixed bug in `createDataSet.py`

## [0.0.5] - 2024-Apr-25

Added:

- Added `installation` tutorial
- Added `reproducible codes` for paper

Changed:

- Refactored the `pyproject.toml` according to setuptools
- Rename losses name to make them consistent with paper

Removed:

- Removed Unused imports and files
- Removed `setup.py`

## [0.0.4] - 2024-Apr-16

Added:

- Added `simulation` data and tutorial
- Added `niche cluster` information output
- Added `niche cluster` tutorial
- Added duplicate `Cell_ID` handle

Changed:

- Make this repository public

Fixed:

- Fixed errors when there is only 1 sample for `post-analysis` tutorial

## [0.0.3] - 2024-Apr-2

Added

- Added `post-analysis` tutorial

Changed:

- Updated dependent packages information
- Updated installation tutorial
- Updated `stereo-seq` example

## [0.0.2] - 2024-Mar-12

Added

- Added package description
- Added `.gitignore` to remove unnecessary files
- Added `ONTraC` for running all steps together
- Added `NTScore` for generating NTScore from GNN output
- Added pip installation support

Changed:

- New environment constructions
- Running process
- Uniform parameters control
- Output directories
- `createDataSet` generate cell type composition and GNN input from raw input
- Input data format
