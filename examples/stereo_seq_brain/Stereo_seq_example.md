# Stereo-seq Mouse Midbrain

## Installation

Please see the [installation tutorial](../../tutorials/installation.md)

## Input data

`original_data.csv`

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

## Run ONTraC

```bash
ONTraC -d original_data.csv --preprocessing-dir data/stereo_seq_final_preprocessing_dir --GNN-dir output/stereo_seq_final_GNN --NTScore-dir output/stereo_seq_final_NTScore \
       --epochs 1000 --batch-size 5 -s 42 --patience 100 --min-delta 0.001 --min-epochs 50 --lr 0.03 --hidden-feats 4 -k 6 \
       --modularity-loss-weight 0.3 --regularization-loss-weight 0.1     --purity-loss-weight 300 --beta 0.03 2>&1 | tee stereo_seq_final.log
```

Runs for about 6 mintes on NVIDIA A100 80GB + Intel 8358, 2.6 GHz (8-core CPU) and 15 minutes on Intel 8358, 2.6 GHz (8-core CPU).

## Post-analysis

Please see [post analysis tutorial](../../tutorial/post_analysis.md)
