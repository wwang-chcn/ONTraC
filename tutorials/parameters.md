# Parameters explanation

## Full parameters list

### Full parameters for ONTraC

```{text}
Usage: ONTraC <--meta-input META_INPUT> [--exp-input EXP_INPUT] [--embedding-input EMBEDDING_INPUT]
    [--decomposition-cell-type-composition-input DECOMPOSITION_CELL_TYPE_COMPOSITION_INPUT]
    [--decomposition-expression-input DECOMPOSITION_EXPRESSION_INPUT] <--preprocessing-dir PREPROCESSING_DIR>
    <--GNN-dir GNN_DIR> <--NTScore-dir NTSCORE_DIR> [--n-cpu N_CPU] [--n-neighbors N_NEIGHBORS] [--n-local N_LOCAL]
    [--embedding-adjust] [--sigma SIGMA] [--device DEVICE] [--epochs EPOCHS] [--patience PATIENCE]
    [--min-delta MIN_DELTA] [--min-epochs MIN_EPOCHS] [--batch-size BATCH_SIZE] [-s SEED] [--seed SEED] [--lr LR]
    [--hidden-feats HIDDEN_FEATS] [-k K_CLUSTERS] [--modularity-loss-weight MODULARITY_LOSS_WEIGHT]
    [--purity-loss-weight PURITY_LOSS_WEIGHT] [--regularization-loss-weight REGULARIZATION_LOSS_WEIGHT] [--beta BETA]
    [--trajectory-construct TRAJECTORY_CONSTRUCT]

All steps of ONTraC including dataset creation, Graph Pooling, and NT score calculation.

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit

  IO:
    -d DATASET, --dataset=DATASET
                        This options will be deprecated in V3. Please use --meta-input instead.
    --meta-input=META_INPUT
                        Meta data file in csv format. Each row is a cell and each column is a meta data. The
                        first column should be the cell name. Coordinates (x, y) and sample should be
                        included. Cell type is optional.
    --exp-input=EXP_INPUT
                        Normalized expression file in csv format. Each row is a cell and each column is a
                        gene. The first column should be the cell name. If not provided, cell type should be
                        included in the meta data file.
    --embedding-input=EMBEDDING_INPUT
                        Embedding file in csv format. The first column should be the cell name.
    --decomposition-cell-type-composition-input=DECOMPOSITION_CELL_TYPE_COMPOSITION_INPUT
                        Decomposition outputed cell type composition of each spot in csv format. The first
                        column should be the spot name.
    --decomposition-expression-input=DECOMPOSITION_EXPRESSION_INPUT
                        Decomposition outputed expression of each cell type in csv format. The first column
                        should be the cell type name corresponding to the columns name of decomposition
                        outputed cell type composition.
    --preprocessing-dir=PREPROCESSING_DIR
                        Directory for preprocessing outputs.
    --GNN-dir=GNN_DIR   Directory for the GNN output.
    --NTScore-dir=NTSCORE_DIR
                        Directory for the NTScore output.

  Preprocessing:
    --resolution=RESOLUTION
                        Resolution for leiden clustering. Used for clustering cells into cell types when
                        gene expression data is provided. Default is 1.0.

  Niche Network Construction:
    --n-cpu=N_CPU       Number of CPUs used for parallel computing in dataset preprocessing. Default is 4.
    --n-neighbors=N_NEIGHBORS
                        Number of neighbors used for kNN graph construction. It should be less than the
                        number of cells in each sample. Default is 50.
    --n-local=N_LOCAL   Specifies the nth closest local neighbors used for gaussian distance normalization.
                        It should be less than the number of cells in each sample. Default is 20.
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
    --equal-space       Whether to assign equally spaced values to for each niche cluster. Default is False,
                        based on total loadings of each niche cluster.
    --trajectory-construct=TRAJECTORY_CONSTRUCT
                        Method to construct the niche trajectory. Default is 'BF' (brute-force). A faster
                        alternative is 'TSP'.
```

### Full parameters for ONTraC_pp

```{text}
Usage: ONTraC_pp <--meta-input META_INPUT> [--exp-input EXP_INPUT] [--embedding-input EMBEDDING_INPUT]
    [--decomposition-cell-type-composition-input DECOMPOSITION_CELL_TYPE_COMPOSITION_INPUT]
    [--decomposition-expression-input DECOMPOSITION_EXPRESSION_INPUT] <--preprocessing-dir PREPROCESSING_DIR> [--n-cpu N_CPU]
    [--n-neighbors N_NEIGHBORS] [--n-local N_LOCAL] [--embedding-adjust] [--sigma SIGMA]

Preporcessing and create dataset for GNN and following analysis.

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit

  IO:
    -d DATASET, --dataset=DATASET
                        This options will be deprecated in V3. Please use --meta-input instead.
    --meta-input=META_INPUT
                        Meta data file in csv format. Each row is a cell and each column is a meta data. The first column should be the cell name. Coordinates (x, y) and sample should be included. Cell type is optional.
    --exp-input=EXP_INPUT
                        Normalized expression file in csv format. Each row is a cell and each column is a gene. The first column should be the cell name. If not provided, cell type should be included in the meta data file.
    --embedding-input=EMBEDDING_INPUT
                        Embedding file in csv format. The first column should be the cell name.
    --decomposition-cell-type-composition-input=DECOMPOSITION_CELL_TYPE_COMPOSITION_INPUT
                        Decomposition outputed cell type composition of each spot in csv format. The first column should be the spot name.
    --decomposition-expression-input=DECOMPOSITION_EXPRESSION_INPUT
                        Decomposition outputed expression of each cell type in csv format. The first column should be the cell type name corresponding to the columns name of decomposition outputed cell type composition.
    --preprocessing-dir=PREPROCESSING_DIR
                        Directory for preprocessing outputs.

  Preprocessing:
    --resolution=RESOLUTION
                        Resolution for leiden clustering. Used for clustering cells into cell types when gene expression data is provided. Default is 1.0.

  Niche Network Construction:
    --n-cpu=N_CPU       Number of CPUs used for parallel computing in dataset preprocessing. Default is 4.
    --n-neighbors=N_NEIGHBORS
                        Number of neighbors used for kNN graph construction. It should be less than the number of cells in each sample. Default is 50.
    --n-local=N_LOCAL   Specifies the nth closest local neighbors used for gaussian distance normalization. It should be less than the number of cells in each sample. Default is 20.
    --embedding-adjust  Adjust the cell type coding according to embeddings. Default is False. At least two (Embedding_1 and Embedding_2) should be in the original data if embedding_adjust is True.
    --sigma=SIGMA       Sigma for the exponential function to control the similarity between different cell types or clusters. Default is 1.
```

### Full parameters for ONTraC_GNN

```{text}
Usage: ONTraC_GNN <--preprocessing-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR> [--device DEVICE]
    [--epochs EPOCHS] [--patience PATIENCE] [--min-delta MIN_DELTA] [--min-epochs MIN_EPOCHS] [--batch-size BATCH_SIZE] 
    [-s SEED] [--seed SEED] [--lr LR] [--hidden-feats HIDDEN_FEATS] [-k K_CLUSTERS]
    [--modularity-loss-weight MODULARITY_LOSS_WEIGHT] [--purity-loss-weight PURITY_LOSS_WEIGHT] 
    [--regularization-loss-weight REGULARIZATION_LOSS_WEIGHT] [--beta BETA]

Graph Neural Network (GNN)

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit

  IO:
    --preprocessing-dir=PREPROCESSING_DIR
                        Directory for preprocessing outputs.
    --GNN-dir=GNN_DIR   Directory for the GNN output.

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

### Full parameters for NicheTrajectory

```{text}
Usage: NicheTrajectory <--preprocessing-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR> <--NTScore-dir NTSCORE_DIR> 
            <--trajectory-construct TRAJECTORY_CONSTRUCT>

PseudoTime: Calculate PseudoTime for each node in a graph

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit

  IO:
    --preprocessing-dir=PREPROCESSING_DIR
                        Directory for preprocessing outputs.
    --GNN-dir=GNN_DIR   Directory for the GNN output.
    --NTScore-dir=NTSCORE_DIR
                        Directory for the NTScore output.

  Options for niche trajectory:
    --equal-space       Whether to assign equally spaced values to for each niche cluster. Default is False, based on total loadings of each niche cluster.
    --trajectory-construct=TRAJECTORY_CONSTRUCT
                        Method to construct the niche trajectory. Default is 'BF' (brute-force). A faster alternative is 'TSP'.
```

### Full parameters for ONTraC_analysis

```{text}
Usage: ONTraC_analysis <--preprocessing-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR>
    <--NTScore-dir NTSCORE_DIR> <-o OUTPUT_DIR> [-l LOG_FILE] [-r REVERSE] [-s SAMPLE] [--scale-factor SCALE_FACTOR]
    [--suppress-cell-type-composition] [--suppress-niche-cluster-loadings] [--suppress-niche-trajectory]

Analysis the results of ONTraC.

Options:
  --version             show program's version number and exit
  -h, --help            Show this help message and exit.
  -l LOG, --log=LOG     Log file.
  -o OUTPUT, --output=OUTPUT
                        Output directory.
  -r, --reverse         Reverse the NT score.

  IO:
    --preprocessing-dir=PREPROCESSING_DIR
                        Directory for preprocessing outputs.
    --GNN-dir=GNN_DIR   Directory for the GNN output.
    --NTScore-dir=NTSCORE_DIR
                        Directory for the NTScore output.

  Embedding adjust options:
    --embedding-adjust  Adjust the cell type coding according to embeddings. Default is False. At least two (Embedding_1 and Embedding_2) should be in the original data if embedding_adjust is True.
    --sigma=SIGMA       Sigma for the exponential function to control the similarity between different cell types or clusters. Default is 1.

  Suppress options:
    --suppress-cell-type-composition
                        Suppress the cell type composition visualization.
    --suppress-niche-cluster-loadings
                        Suppress the niche cluster loadings visualization.
    --suppress-cell-type
                        Suppress the cell type visualization.
    --suppress-niche-trajectory
                        Suppress the niche trajectory related visualization.

  Visualization options:
    -s, --sample        Plot each sample separately.
    --scale-factor=SCALE_FACTOR
                        Scale factor control the size of spatial-based plots.

  Deprecated options:
    -d DATASET, --dataset=DATASET
                        This options is deprecated.
    --meta-input=META_INPUT
                        This options is deprecated.
    --exp-input=EXP_INPUT
                        This options is deprecated.
    --decomposition-cell-type-composition-input=DECOMPOSITION_CELL_TYPE_COMPOSITION_INPUT
                        This options is deprecated.
    --decomposition-expression-input=DECOMPOSITION_EXPRESSION_INPUT
                        This options is deprecated.
```

### Full parameters for ONTraC_GP

```{text}
Usage: ONTraC_GP <--meta-input META_INPUT> [--exp-input EXP_INPUT] [--embedding-input EMBEDDING_INPUT]
    [--decomposition-cell-type-composition-input DECOMPOSITION_CELL_TYPE_COMPOSITION_INPUT]
    [--decomposition-expression-input DECOMPOSITION_EXPRESSION_INPUT] <--preprocessing-dir PREPROCESSING_DIR>
    <--GNN-dir GNN_DIR> <--NTScore-dir NTSCORE_DIR>  [--device DEVICE] [--epochs EPOCHS] [--patience PATIENCE]
    [--min-delta MIN_DELTA] [--min-epochs MIN_EPOCHS] [--batch-size BATCH_SIZE] [-s SEED] [--seed SEED] [--lr LR]
    [--hidden-feats HIDDEN_FEATS] [-k K_CLUSTERS] [--modularity-loss-weight MODULARITY_LOSS_WEIGHT]
    [--purity-loss-weight PURITY_LOSS_WEIGHT] [--regularization-loss-weight REGULARIZATION_LOSS_WEIGHT] [--beta BETA]

GP (Graph Pooling): GNN & Node Pooling

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit

  IO:
    -d DATASET, --dataset=DATASET
                        This options will be deprecated in V3. Please use --meta-input instead.
    --meta-input=META_INPUT
                        Meta data file in csv format. Each row is a cell and each column is a meta data. The first column should be the cell name. Coordinates (x, y) and sample should be included. Cell type is optional.
    --exp-input=EXP_INPUT
                        Normalized expression file in csv format. Each row is a cell and each column is a gene. The first column should be the cell name. If not provided, cell type should be included in the meta data file.
    --embedding-input=EMBEDDING_INPUT
                        Embedding file in csv format. The first column should be the cell name.
    --decomposition-cell-type-composition-input=DECOMPOSITION_CELL_TYPE_COMPOSITION_INPUT
                        Decomposition outputed cell type composition of each spot in csv format. The first column should be the spot name.
    --decomposition-expression-input=DECOMPOSITION_EXPRESSION_INPUT
                        Decomposition outputed expression of each cell type in csv format. The first column should be the cell type name corresponding to the columns name of decomposition outputed cell type composition.
    --preprocessing-dir=PREPROCESSING_DIR
                        Directory for preprocessing outputs.
    --GNN-dir=GNN_DIR   Directory for the GNN output.
    --NTScore-dir=NTSCORE_DIR
                        Directory for the NTScore output.

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
    --equal-space       Whether to assign equally spaced values to for each niche cluster. Default is False, based on total loadings of each niche cluster.
    --trajectory-construct=TRAJECTORY_CONSTRUCT
                        Method to construct the niche trajectory. Default is 'BF' (brute-force). A faster alternative is 'TSP'.
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
