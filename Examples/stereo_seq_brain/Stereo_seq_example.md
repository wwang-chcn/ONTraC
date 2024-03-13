# Stereo-seq Mouse Midbrain

## Installation

```bash
pip install ONTraC
```

## Input data

`original_data.csv`

This file contains all input formation with five columns: Cell_ID, Sample, Cell_Type, x, and y.

| Cell_ID         | Sample   | Cell_Type | x       | y     |
| --------------- | -------- | --------- | ------- | ----- |
| E12_E1S3_100034 | E12_E1S3 | Fibro     | 15940   | 18584 |
| E12_E1S3_100035 | E12_E1S3 | Fibro     | 15942   | 18623 |
| ...             | ...      | ...       | ...     | ...   |
| E16_E2S7_326412 | E16_E2S7 | Fibro     | 32990.5 | 14475 |

## Run ONTraC

```bash
ONTraC -d data/stereo_seq_brain/original_data.csv --preprocessing-dir data/stereo_seq_final_preprocessing_dir \
  --GNN-dir output/stereo_seq_final_GNN --NTScore-dir output/stereo_seq_final_NTScore \
  --epochs 1000 --batch-size 5 -s 42 --patience 100 --min-delta 0.001 --min-epochs 50 --lr 0.03 \
  --hidden-feats 4 -k 6 --spectral-loss-weight 0.3 --cluster-loss-weight 0.1 --feat-similarity-loss-weight 300 --assign-exponent 0.03 > stereo_seq_final.log
```

## Post-analysis

### Cell-type composition

### Spatial distribution of Niche-level NT score

### Spatial distribution of Cell-level NT score

### Cell-level NT score distribution for each cell type
