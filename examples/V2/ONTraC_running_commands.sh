#! /bin/bash

# ONTraC Installation

conda create -y -n ONTraC python=3.11
conda activate ONTraC
pip install git+https://github.com/gyuanlab/ONTraC.git@V2

# Running ONTraC

mkdir -p data log analysis_output

## without kernel method (meta input only)

ONTraC --meta-input data/merfish_meta.csv --preprocessing-dir data/merfish_wo_kernel_preprocessing --GNN-dir output/merfish_wo_kernel_GNN --NTScore-dir output/merfish_wo_kernel_NTScore --n-neighbors 20 --device cuda --epochs 1000 --batch-size 10 -s 42 --patience 100 --min-delta 0.001 --min-epochs 50 --lr 0.03 --hidden-feats 4 -k 6 --modularity-loss-weight 1 --regularization-loss-weight 0.1  --purity-loss-weight 30 --beta 0.3 --equal-space 2>&1 | tee log/merfish_wo_kernel.log
ONTraC_analysis -o analysis_output/merfish_wo_kernel -l log/merfish_wo_kernel.log --meta-input data/merfish_meta.csv --preprocessing-dir data/merfish_wo_kernel_preprocessing --GNN-dir output/merfish_wo_kernel_GNN --NTScore-dir output/merfish_wo_kernel_NTScore --suppress-cell-type-composition --suppress-niche-cluster-loadings

## with kernel method (meta data with given embeedings)

ONTraC --meta-input data/merfish_meta.csv --embedding-input data/merfish_embedding.csv --preprocessing-dir data/merfish_w_kernel_preprocessing --GNN-dir output/merfish_w_kernel_GNN --NTScore-dir output/merfish_w_kernel_NTScore --n-neighbors 20 --embedding-adjust --sigma 1 --device cuda --epochs 1000 --batch-size 10 -s 42 --patience 100 --min-delta 0.001 --min-epochs 50 --lr 0.03 --hidden-feats 4 -k 6 --modularity-loss-weight 1 --regularization-loss-weight 0.1  --purity-loss-weight 30 --beta 0.3 --equal-space 2>&1 | tee log/merfish_w_kernel.log
ONTraC_analysis -o analysis_output/merfish_w_kernel -l log/merfish_w_kernel.log --meta-input data/merfish_meta.csv --embedding-input data/merfish_embedding.csv --preprocessing-dir data/merfish_w_kernel_preprocessing --GNN-dir output/merfish_w_kernel_GNN --NTScore-dir output/merfish_w_kernel_NTScore --embedding-adjust --sigma 1 --suppress-cell-type-composition --suppress-niche-cluster-loadings

# example for different types of input

## meta input only
## same as without kernel method

## meta data with given embeedings
## same as with kernel method


## meta data with gene expression

ONTraC --meta-input data/merfish_meta.csv --exp-input data/merfish_counts.csv --preprocessing-dir data/V2_example_exp_input_preprocessing --GNN-dir output/V2_example_exp_input_GNN --NTScore-dir output/V2_example_exp_input_NTScore --resolution 1 --embedding-adjust -s 42 --equal-space 2>&1 | tee log/V2_example_exp_input.log
ONTraC_analysis -o analysis_output/V2_example_exp_input -l log/V2_example_exp_input.log --meta-input data/merfish_meta.csv --exp-input data/merfish_counts.csv --preprocessing-dir data/V2_example_exp_input_preprocessing --GNN-dir output/V2_example_exp_input_GNN --NTScore-dir output/V2_example_exp_input_NTScore --suppress-cell-type-composition --suppress-niche-cluster-loadings

## handling low resolution data based on decomposition results

ONTraC --meta-input data/visium_metadata.csv --decomposition-cell-type-composition-input data/spotxcelltype.csv --decomposition-expression-input data/celltypexgexp.csv --preprocessing-dir data/V2_example_visium_preprocessing --GNN-dir output/V2_example_visium_GNN --NTScore-dir output/V2_example_visium_NTScore --n-neighbors 15 --embedding-adjust --sigma 1 -s 42 --equal-space 2>&1 | tee log/V2_example_visium.log
ONTraC_analysis -o analysis_output/V2_example_visium -l log/V2_example_visium.log --meta-input data/visium_metadata.csv --decomposition-cell-type-composition-input data/spotxcelltype.csv --decomposition-expression-input data/celltypexgexp.csv --preprocessing-dir data/V2_example_visium_preprocessing --GNN-dir output/V2_example_visium_GNN --NTScore-dir output/V2_example_visium_NTScore --embedding-adjust --sigma 1 --suppress-cell-type --suppress-niche-cluster-loadings