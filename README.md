# **Ordered Niche Trajectory Construction** (ONTraC)

## Software dependencies

pandas=2.1.1

scipy=1.11.2

scikit-learn=1.3.0

pytorch=2.0.1

torchvision=0.15.2

torchaudio=2.0.2

pytorch-cuda=11.8

pyg_lib=0.2.0

torch_scatter=2.1.1

torch_sparse=0.6.17

torch_cluster=1.6.1

torch_spline_conv=1.2.2

torch_geometric=2.3.1

## Installation

- Setup environment following the instruction on `Environment_setup`
- Install `ONTraC`

  ```{sh}
  git clone https://github.com/wwang-chcn/ONTraC.git
  cd ONTraC && pip install .
  ```

## File Format

### Original_data.csv

This file contains all input formation with five columns: Cell_ID, Sample, Cell_Type, x, and y.

| Cell_ID         | Sample   | Cell_Type | x       | y     |
| --------------- | -------- | --------- | ------- | ----- |
| E12_E1S3_100034 | E12_E1S3 | Fibro     | 15940   | 18584 |
| E12_E1S3_100035 | E12_E1S3 | Fibro     | 15942   | 18623 |
| ...             | ...      | ...       | ...     | ...   |
| E16_E2S7_326412 | E16_E2S7 | Fibro     | 32990.5 | 14475 |

## Citation
