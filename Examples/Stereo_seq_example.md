# Stereo-seq Mouse Midbrain

```bash
ONTraC -d data/stereo_seq_brain/original_data.csv --preprocessing-dir test_preprocessing_dir --GNN-dir test_GNN_dir --NTScore-dir test_NT_score_dir --n-cpu 8 --device cuda --epochs 1000 --patience 100 --lr 0.03 -s 42 --hidden-feats 4 -k 6 --spectral-loss-weight 1 --cluster-loss-weight 0.1     --feat-similarity-loss-weight 30 --assign-exponent 0.3 > stereo_seq_env_test.log
```
