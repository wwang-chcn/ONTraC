# Parameters explanation

## Full parameters list

### Full parameters for ONTraC

```{text}
Usage: ONTraC <--NN-dir NN_DIR> <--GNN-dir GNN_DIR> <--NT-dir NT_DIR> <--meta-input META_INPUT>
    [--exp-input EXP_INPUT] [--embedding-input EMBEDDING_INPUT] [--low-res-exp-input LOW_RES_EXP_INPUT] [--deconvoluted-ct-composition DECONVOLUTED_CT_COMPOSITION]
    [--deconvoluted-exp-input DECONVOLUTED_EXP_INPUT] [--resolution RESOLUTION] [--deconvolution-method DC_METHOD] [--deconvolution-ct-num DC_CT_NUM]
    [--n-cpu N_CPU] [--n-neighbors N_NEIGHBORS] [--n-local N_LOCAL] [--embedding-adjust] [--sigma SIGMA] [--device DEVICE] [--epochs EPOCHS] [--patience PATIENCE]
    [--min-delta MIN_DELTA] [--min-epochs MIN_EPOCHS] [--batch-size BATCH_SIZE] [-s SEED] [--lr LR] [--hidden-feats HIDDEN_FEATS]
    [--n-gcn-layers N_GCN_LAYERS] [-k K] [--modularity-loss-weight MODULARITY_LOSS_WEIGHT] [--purity-loss-weight PURITY_LOSS_WEIGHT]
    [--regularization-loss-weight REGULARIZATION_LOSS_WEIGHT] [--beta BETA] [--trajectory-construct TRAJECTORY_CONSTRUCT] [--equal-space]

All steps of ONTraC including niche network construction, GNN, and niche construction.

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit

  IO:
    --NN-dir=NN_DIR     Directory for niche network outputs.
    --GNN-dir=GNN_DIR   Directory for the GNN output.
    --NT-dir=NT_DIR     Directory for the niche trajectory output.
    --meta-input=META_INPUT
                        Meta data file in csv format. Each row is a cell. The first column should be the cell
                        name with column name Cell_ID. Coordinates (x, y) and sample should be included. Cell
                        type is required for cell-level data.
    --exp-input=EXP_INPUT
                        Normalized expression file in csv format. Each row is a cell and each column is a gene.
                        The first column should be the cell name with column name Cell_ID. If not provided,
                        cell type should be included in the meta data file.
    --embedding-input=EMBEDDING_INPUT
                        Embedding file in csv format. The first column should be the cell name with column name
                        Cell_ID.
    --low-res-exp-input=LOW_RES_EXP_INPUT
                        Gene X spot matrix in csv format for low-resolution dataset.
    --deconvoluted-ct-composition=DECONVOLUTED_CT_COMPOSITION
                        Deconvoluted cell type composition of each spot in csv format. The first column should
                        be the spot name with column name Spot_ID.
    --deconvoluted-exp-input=DECONVOLUTED_EXP_INPUT
                        Deconvoluted expression of each cell type in csv format. The first column should be the
                        cell type name corresponding to the columns name of decomposition outputed cell type
                        composition.
    --preprocessing-dir=PREPROCESSING_DIR
                        This options will be deprecated from v3.0. Please use --NN-dir instead.
    --NTScore-dir=NTSCORE_DIR
                        This options will be deprecated from v3.0. Please use --NT-dir instead.
    -d DATASET, --dataset=DATASET
                        This options will be deprecated from v3.0. Please use --meta-input instead.

  Preprocessing:
    --resolution=RESOLUTION
                        Resolution for leiden clustering. Used for clustering cells into cell types when gene
                        expression data is provided. Default is 10.0.
    --deconvolution-method=DC_METHOD
                        Deconvolution method used for low resolution data. Default is STdeconvolve.
    --deconvolution-ct-num=DC_CT_NUM
                        Number of cell type that the deconvolution method will deconvolve.

  Niche Network Construction:
    --n-cpu=N_CPU       Number of CPUs used for parallel computing in dataset preprocessing. Default is 4.
    --n-neighbors=N_NEIGHBORS
                        Number of neighbors used for kNN graph construction. It should be less than the number
                        of cells in each sample. Default is 50.
    --n-local=N_LOCAL   Specifies the nth closest local neighbors used for gaussian distance normalization. It
                        should be less than the number of cells in each sample. Default is 20.
    --embedding-adjust  Adjust the cell type coding according to embeddings. Default is False. At least two
                        (Embedding_1 and Embedding_2) should be in the original data if embedding_adjust is
                        True.
    --sigma=SIGMA       Sigma for the exponential function to control the similarity between different cell
                        types or clusters. Default is 1.

  Options for GNN training:
    --device=DEVICE     Device for training. We support cpu and cuda now. Auto select if not specified.
    --epochs=EPOCHS     Number of maximum epochs for training. Default is 1000.
    --patience=PATIENCE
                        Number of epochs wait for better result. Default is 100.
    --min-delta=MIN_DELTA
                        Minimum delta for better result. Default is 0.001
    --min-epochs=MIN_EPOCHS
                        Minimum number of epochs for training. Default is 50. Set to 0 to disable.
    --batch-size=BATCH_SIZE
                        Batch size for training. Default is 0 for whole dataset.
    -s SEED, --seed=SEED
                        Random seed for training. Default is random.
    --lr=LR             Learning rate for training. Default is 0.03.
    --hidden-feats=HIDDEN_FEATS
                        Number of hidden features. Default is 4.
    --n-gcn-layers=N_GCN_LAYERS
                        Number of GCN layers. Default is 2.
    -k K, --k-clusters=K
                        Number of niche clusters. Default is 6.
    --modularity-loss-weight=MODULARITY_LOSS_WEIGHT
                        Weight for modularity loss. Default is 1.
    --purity-loss-weight=PURITY_LOSS_WEIGHT
                        Weight for purity loss. Default is 30.
    --regularization-loss-weight=REGULARIZATION_LOSS_WEIGHT
                        Weight for regularization loss. Default is 0.1.
    --beta=BETA         Beta value control niche cluster assignment matrix. Default is 0.3.

  Options for niche trajectory:
    --trajectory-construct=TRAJECTORY_CONSTRUCT
                        Method to construct the niche trajectory. Default is 'BF' (brute-force). A faster
                        alternative is 'TSP'.
    --equal-space       Whether to assign equally spaced values to for each niche cluster. Default is False,
                        based on total loadings of each niche cluster.
```

### Full parameters for ONTraC_NN

Previouly named as createDataSet.

```{text}
Usage: ONTraC_NN <--NN-dir NN_DIR> <--meta-input META_INPUT> [--exp-input EXP_INPUT]
    [--embedding-input EMBEDDING_INPUT] [--low-res-exp-input LOW_RES_EXP_INPUT]
    [--deconvoluted-ct-composition DECONVOLUTED_CT_COMPOSITION] [--deconvoluted-exp-input DECONVOLUTED_EXP_INPUT]
    [--resolution RESOLUTION] [--deconvolution-method DC_METHOD] [--deconvolution-ct-num DC_CT_NUM]
    [--n-cpu N_CPU] [--n-neighbors N_NEIGHBORS] [--n-local N_LOCAL] [--embedding-adjust] [--sigma SIGMA]

Create niche network and calculate features (normalized cell type composition). (Step 1 of ONTraC)

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit

  IO:
    --NN-dir=NN_DIR     Directory for niche network outputs.
    --meta-input=META_INPUT
                        Meta data file in csv format. Each row is a cell. The first column should be the cell
                        name with column name Cell_ID. Coordinates (x, y) and sample should be included. Cell
                        type is required for cell-level data.
    --exp-input=EXP_INPUT
                        Normalized expression file in csv format. Each row is a cell and each column is a gene.
                        The first column should be the cell name with column name Cell_ID. If not provided,
                        cell type should be included in the meta data file.
    --embedding-input=EMBEDDING_INPUT
                        Embedding file in csv format. The first column should be the cell name with column name
                        Cell_ID.
    --low-res-exp-input=LOW_RES_EXP_INPUT
                        Gene X spot matrix in csv format for low-resolution dataset.
    --deconvoluted-ct-composition=DECONVOLUTED_CT_COMPOSITION
                        Deconvoluted cell type composition of each spot in csv format. The first column should
                        be the spot name with column name Spot_ID.
    --deconvoluted-exp-input=DECONVOLUTED_EXP_INPUT
                        Deconvoluted expression of each cell type in csv format. The first column should be the
                        cell type name corresponding to the columns name of decomposition outputed cell type
                        composition.
    --preprocessing-dir=PREPROCESSING_DIR
                        This options will be deprecated from v3.0. Please use --NN-dir instead.
    -d DATASET, --dataset=DATASET
                        This options will be deprecated from v3.0. Please use --meta-input instead.

  Preprocessing:
    --resolution=RESOLUTION
                        Resolution for leiden clustering. Used for clustering cells into cell types when gene
                        expression data is provided. Default is 1.0.
    --deconvolution-method=DC_METHOD
                        Deconvolution method used for low resolution data. Default is STdeconvolve.
    --deconvolution-ct-num=DC_CT_NUM
                        Number of cell type that the deconvolution method will deconvolve.

  Niche Network Construction:
    --n-cpu=N_CPU       Number of CPUs used for parallel computing in dataset preprocessing. Default is 4.
    --n-neighbors=N_NEIGHBORS
                        Number of neighbors used for kNN graph construction. It should be less than the number
                        of cells in each sample. Default is 50.
    --n-local=N_LOCAL   Specifies the nth closest local neighbors used for gaussian distance normalization. It
                        should be less than the number of cells in each sample. Default is 20.
    --embedding-adjust  Adjust the cell type coding according to embeddings. Default is False. At least two
                        (Embedding_1 and Embedding_2) should be in the original data if embedding_adjust is
                        True.
    --sigma=SIGMA       Sigma for the exponential function to control the similarity between different cell
                        types or clusters. Default is 1.
```

### Full parameters for ONTraC_GNN

```{text}
Usage: ONTraC_GNN <--NN-dir NN_DIR> <--GNN-dir GNN_DIR> [--device DEVICE]
    [--epochs EPOCHS] [--patience PATIENCE] [--min-delta MIN_DELTA] [--min-epochs MIN_EPOCHS] [--batch-size BATCH_SIZE] 
    [-s SEED] [--lr LR] [--hidden-feats HIDDEN_FEATS] [--n-gcn-layers N_GCN_LAYERS] [-k K]
    [--modularity-loss-weight MODULARITY_LOSS_WEIGHT] [--purity-loss-weight PURITY_LOSS_WEIGHT] 
    [--regularization-loss-weight REGULARIZATION_LOSS_WEIGHT] [--beta BETA]

Graph Neural Network (GNN, GCN + GP). The core algorithm of ONTraC. (Step 2/3 of ONTraC)

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit

  IO:
    --NN-dir=NN_DIR     Directory for niche network outputs.
    --GNN-dir=GNN_DIR   Directory for the GNN output.
    --preprocessing-dir=PREPROCESSING_DIR
                        This options will be deprecated from v3.0. Please use --NN-dir instead.

  Options for GNN training:
    --device=DEVICE     Device for training. We support cpu and cuda now. Auto select if not specified.
    --epochs=EPOCHS     Number of maximum epochs for training. Default is 1000.
    --patience=PATIENCE
                        Number of epochs wait for better result. Default is 100.
    --min-delta=MIN_DELTA
                        Minimum delta for better result. Default is 0.001
    --min-epochs=MIN_EPOCHS
                        Minimum number of epochs for training. Default is 50. Set to 0 to disable.
    --batch-size=BATCH_SIZE
                        Batch size for training. Default is 0 for whole dataset.
    -s SEED, --seed=SEED
                        Random seed for training. Default is random.
    --lr=LR             Learning rate for training. Default is 0.03.
    --hidden-feats=HIDDEN_FEATS
                        Number of hidden features. Default is 4.
    --n-gcn-layers=N_GCN_LAYERS
                        Number of GCN layers. Default is 2.
    -k K, --k-clusters=K
                        Number of niche clusters. Default is 6.
    --modularity-loss-weight=MODULARITY_LOSS_WEIGHT
                        Weight for modularity loss. Default is 1.
    --purity-loss-weight=PURITY_LOSS_WEIGHT
                        Weight for purity loss. Default is 30.
    --regularization-loss-weight=REGULARIZATION_LOSS_WEIGHT
                        Weight for regularization loss. Default is 0.1.
    --beta=BETA         Beta value control niche cluster assignment matrix. Default is 0.3.
```

### Full parameters for ONTraC_NT

Previouly named as NicheTrajectory.

```{text}
Usage: ONTraC_NT <--NN-dir NN_DIR> <--GNN-dir GNN_DIR> <--NT-dir NT_DIR> 
            [--trajectory-construct TRAJECTORY_CONSTRUCT] [--equal-space]

ONTraC_NT: construct niche trajectory for niche cluster and project the NT score to each cell. (Step 4 of
ONTraC)

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit

  IO:
    --NN-dir=NN_DIR     Directory for niche network outputs.
    --GNN-dir=GNN_DIR   Directory for the GNN output.
    --NT-dir=NT_DIR     Directory for the niche trajectory output.
    --preprocessing-dir=PREPROCESSING_DIR
                        This options will be deprecated from v3.0. Please use --NN-dir instead.
    --NTScore-dir=NTSCORE_DIR
                        This options will be deprecated from v3.0. Please use --NT-dir instead.

  Options for niche trajectory:
    --trajectory-construct=TRAJECTORY_CONSTRUCT
                        Method to construct the niche trajectory. Default is 'BF' (brute-force). A faster
                        alternative is 'TSP'.
    --equal-space       Whether to assign equally spaced values to for each niche cluster. Default is False,
                        based on total loadings of each niche cluster.
```

### Full parameters for ONTraC_analysis

```{text}
Usage: ONTraC_analysis [--NN-dir NN_DIR] [--GNN-dir GNN_DIR] [--NT-dir NT_DIR] [-o OUTPUT]
    [--meta-input META_INPUT] [-l LOG] [--embedding-adjust] [--sigma SIGMA] [-r REVERSE] [-s SAMPLE]
    [--scale-factor SCALE_FACTOR] [--suppress-cell-type-composition] [--suppress-niche-cluster-loadings]
    [--suppress-niche-trajectory]
    

ONTraC_analysis: analysis of ONTraC results

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit

  IO:
    --NN-dir=NN_DIR     Directory for niche network outputs.
    --GNN-dir=GNN_DIR   Directory for the GNN output.
    --NT-dir=NT_DIR     Directory for the niche trajectory output.
    -o OUTPUT, --output=OUTPUT
                        Directory for analysis output.
    --meta-input=META_INPUT
                        Meta data file in csv format. Each row is a cell. The first column should be the cell
                        name with column name Cell_ID. Coordinates (x, y) and sample should be included. Cell
                        type is required for cell-level data.
    -l LOG, --log=LOG   Log file.
    --preprocessing-dir=PREPROCESSING_DIR
                        This options will be deprecated from v3.0. Please use --NN-dir instead.
    --NTScore-dir=NTSCORE_DIR
                        This options will be deprecated from v3.0. Please use --NT-dir instead.

  Visualization options:
    --embedding-adjust  Adjust the cell type coding according to embeddings. Default is False. At least two
                        (Embedding_1 and Embedding_2) should be in the original data if embedding_adjust is
                        True.
    --sigma=SIGMA       Sigma for the exponential function to control the similarity between different cell
                        types or clusters. Default is 1.
    -r, --reverse       Reverse the NT score during visualization.
    -s, --sample        Plot each sample separately.
    --scale-factor=SCALE_FACTOR
                        Scale factor control the size of spatial-based plots.

  Suppress options:
    --suppress-cell-type-composition
                        Skip the cell type composition visualization. It would be useful when the number of
                        cell types is large.
    --suppress-niche-cluster-loadings
                        Skip the niche cluster loadings visualization. It would be useful when the number of
                        clusters or sample size is large.
    --suppress-niche-trajectory
                        Skip the niche trajectory related visualization.
```

### Full parameters for ONTraC_GT

Previouly named as ONTraC_GP.

```{text}
Usage: ONTraC_GT <--NN-dir NN_DIR> <--GNN-dir GNN_DIR> <--NT-dir NT_DIR> [--device DEVICE]
    [--epochs EPOCHS] [--patience PATIENCE] [--min-delta MIN_DELTA] [--min-epochs MIN_EPOCHS] [--batch-size BATCH_SIZE] 
    [-s SEED] [--lr LR] [--hidden-feats HIDDEN_FEATS] [--n-gcn-layers N_GCN_LAYERS] [-k K]
    [--modularity-loss-weight MODULARITY_LOSS_WEIGHT] [--purity-loss-weight PURITY_LOSS_WEIGHT] 
    [--regularization-loss-weight REGULARIZATION_LOSS_WEIGHT] [--beta BETA] [--trajectory-construct TRAJECTORY_CONSTRUCT]
    [--equal-space]

ONTraC_GT: GNN and Niche Trajectory

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit

  IO:
    --NN-dir=NN_DIR     Directory for niche network outputs.
    --GNN-dir=GNN_DIR   Directory for the GNN output.
    --NT-dir=NT_DIR     Directory for the niche trajectory output.
    --preprocessing-dir=PREPROCESSING_DIR
                        This options will be deprecated from v3.0. Please use --NN-dir instead.
    --NTScore-dir=NTSCORE_DIR
                        This options will be deprecated from v3.0. Please use --NT-dir instead.

  Options for GNN training:
    --device=DEVICE     Device for training. We support cpu and cuda now. Auto select if not specified.
    --epochs=EPOCHS     Number of maximum epochs for training. Default is 1000.
    --patience=PATIENCE
                        Number of epochs wait for better result. Default is 100.
    --min-delta=MIN_DELTA
                        Minimum delta for better result. Default is 0.001
    --min-epochs=MIN_EPOCHS
                        Minimum number of epochs for training. Default is 50. Set to 0 to disable.
    --batch-size=BATCH_SIZE
                        Batch size for training. Default is 0 for whole dataset.
    -s SEED, --seed=SEED
                        Random seed for training. Default is random.
    --lr=LR             Learning rate for training. Default is 0.03.
    --hidden-feats=HIDDEN_FEATS
                        Number of hidden features. Default is 4.
    --n-gcn-layers=N_GCN_LAYERS
                        Number of GCN layers. Default is 2.
    -k K, --k-clusters=K
                        Number of niche clusters. Default is 6.
    --modularity-loss-weight=MODULARITY_LOSS_WEIGHT
                        Weight for modularity loss. Default is 1.
    --purity-loss-weight=PURITY_LOSS_WEIGHT
                        Weight for purity loss. Default is 30.
    --regularization-loss-weight=REGULARIZATION_LOSS_WEIGHT
                        Weight for regularization loss. Default is 0.1.
    --beta=BETA         Beta value control niche cluster assignment matrix. Default is 0.3.

  Options for niche trajectory:
    --trajectory-construct=TRAJECTORY_CONSTRUCT
                        Method to construct the niche trajectory. Default is 'BF' (brute-force). A faster
                        alternative is 'TSP'.
    --equal-space       Whether to assign equally spaced values to for each niche cluster. Default is False,
                        based on total loadings of each niche cluster.
```

## Detailed explanation

A detailed explanation for some parameters is listed below.

### patience

The training process will terminated if the model does not improve after the number of epochs set by this parameter.

### min-delta

The model will be considered improved if the total loss decreases by the propotion set by this parameter.

### hidden-feats

The number of niche features after GCN processing (step2).

### modularity-loss-weight

The modularity loss controls the spatial smoothness of niche clusters.

### purity-loss-weight

The purity loss controls the purity of cell type composition within each niche cluster.

### regularization-loss-weight

The regularization loss controls the balance of niche cluster sizes. The higher this weight is set, the more equal the size of each niche cluster.

### beta

The beta value of the softmax function used in generating niche cluster assignment matrix.
