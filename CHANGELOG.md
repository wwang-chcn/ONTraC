# Change log

## [XXXXX] - XXXX-XXX-XX

Changed:

- Instruction for `analysis` installation
- Make log file and train loss in `analysis` part optional

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
- Added `ONTraC` for run all steps together
- Added `NTScore` for generate NTScore from GNN output
- Added pip installation support

Changed:

- New environment constructions
- Running process
- Uniform parameters control
- Output directories
- `createDataSet` generate cell type compostion and GNN input from raw input
- Input data format
