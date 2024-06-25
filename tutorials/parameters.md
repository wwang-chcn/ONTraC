# Parameters explanation

## Full parameters list

### Full parameters for ONTraC

```{text}
Usage: ONTraC <-d DATASET> <--preprocessing-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR> <--NTScore-dir NTSCORE_DIR>
    [--n-cpu N_CPU] [--n-neighbors N_NEIGHBORS] [--device DEVICE] [--epochs EPOCHS] [--patience PATIENCE] [--min-delta MIN_DELTA] 
    [--min-epochs MIN_EPOCHS] [--batch-size BATCH_SIZE] [-s SEED] [--seed SEED] [--lr LR] [--hidden-feats HIDDEN_FEATS] [-k K_CLUSTERS]
    [--modularity-loss-weight MODULARITY_LOSS_WEIGHT] [--purity-loss-weight PURITY_LOSS_WEIGHT] 
    [--regularization-loss-weight REGULARIZATION_LOSS_WEIGHT] [--beta BETA]

All steps of ONTraC including dataset creation, Graph Pooling, and NT score
calculation.

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit

  IO:
    -d DATASET, --dataset=DATASET
                        Original input dataset.
    --preprocessing-dir=PREPROCESSING_DIR
                        Directory for preprocessing outputs.
    --GNN-dir=GNN_DIR   Directory for the GNN output.
    --NTScore-dir=NTSCORE_DIR
                        Directory for the NTScore output.

  Niche Network Construction:
    --n-cpu=N_CPU       Number of CPUs used for parallel computing in dataset
                        preprocessing. Default is 4.
    --n-neighbors=N_NEIGHBORS
                        Number of neighbors used for kNN graph construction.
                        Default is 50.

  Options for training:
    --device=DEVICE     Device for training. We support cpu and cuda now. Auto
                        select if not specified.
    --epochs=EPOCHS     Number of maximum epochs for training. Default is
                        1000.
    --patience=PATIENCE
                        Number of epochs wait for better result. Default is
                        100.
    --min-delta=MIN_DELTA
                        Minimum delta for better result. Default is 0.001
    --min-epochs=MIN_EPOCHS
                        Minimum number of epochs for training. Default is 50.
                        Set to 0 to disable.
    --batch-size=BATCH_SIZE
                        Batch size for training. Default is 0 for whole
                        dataset.
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
    --beta=BETA         Beta value control niche cluster assignment matrix.
                        Default is 0.3.
```

### Full parameters for createDataSet

```{text}
Usage: createDataSet <-d DATASET> <--preprocessing-dir PREPROCESSING_DIR> [--n-cpu N_CPU] [--n-neighbors N_NEIGHBORS]

Create dataset for follwoing analysis.

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit

  IO:
    -d DATASET, --dataset=DATASET
                        Original input dataset.
    --preprocessing-dir=PREPROCESSING_DIR
                        Directory for preprocessing outputs.

  Niche Network Construction:
    --n-cpu=N_CPU       Number of CPUs used for parallel computing in dataset
                        preprocessing. Default is 4.
    --n-neighbors=N_NEIGHBORS
                        Number of neighbors used for kNN graph construction.
                        Default is 50.
```

### Full parameters for GP

```{text}
Usage: GP <-d DATASET> <--preprocessing-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR> <--NTScore-dir NTSCORE_DIR> 
    [--device DEVICE] [--epochs EPOCHS] [--patience PATIENCE] [--min-delta MIN_DELTA] [--min-epochs MIN_EPOCHS] [--batch-size BATCH_SIZE] 
    [-s SEED] [--seed SEED] [--lr LR] [--hidden-feats HIDDEN_FEATS] [-k K_CLUSTERS]
    [--modularity-loss-weight MODULARITY_LOSS_WEIGHT] [--purity-loss-weight PURITY_LOSS_WEIGHT] 
    [--regularization-loss-weight REGULARIZATION_LOSS_WEIGHT] [--beta BETA]

GP (Graph Pooling): GNN & Node Pooling

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit

  IO:
    -d DATASET, --dataset=DATASET
                        Original input dataset.
    --preprocessing-dir=PREPROCESSING_DIR
                        Directory for preprocessing outputs.
    --GNN-dir=GNN_DIR   Directory for the GNN output.
    --NTScore-dir=NTSCORE_DIR
                        Directory for the NTScore output.

  Options for training:
    --device=DEVICE     Device for training. We support cpu and cuda now. Auto
                        select if not specified.
    --epochs=EPOCHS     Number of maximum epochs for training. Default is
                        1000.
    --patience=PATIENCE
                        Number of epochs wait for better result. Default is
                        100.
    --min-delta=MIN_DELTA
                        Minimum delta for better result. Default is 0.001
    --min-epochs=MIN_EPOCHS
                        Minimum number of epochs for training. Default is 50.
                        Set to 0 to disable.
    --batch-size=BATCH_SIZE
                        Batch size for training. Default is 0 for whole
                        dataset.
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
    --beta=BETA         Beta value control niche cluster assignment matrix.
                        Default is 0.3.
```

### Full parameters for NTScore

```{text}
Usage: NTScore <--preprocessing-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR> <--NTScore-dir NTSCORE_DIR>

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
```

### Full parameters for ONTraC_analysis

```{text}
Usage: ONTraC_analysis <-d DATASET> <--preprocessing-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR> <--NTScore-dir NTSCORE_DIR> <-o OUTPUT_DIR> [-l LOG_FILE] [-r REVERSE]

Analysis the results of ONTraC.

Options:
  --version             show program's version number and exit
  -h, --help            Show this help message and exit.
  -o OUTPUT, --output=OUTPUT
                        Output directory.
  -l LOG, --log=LOG     Log file.
  -r, --reverse         Reverse the NT score.
  -s, --sample          Plot each sample separately.

  IO:
    -d DATASET, --dataset=DATASET
                        Original input dataset.
    --preprocessing-dir=PREPROCESSING_DIR
                        Directory for preprocessing outputs.
    --GNN-dir=GNN_DIR   Directory for the GNN output.
    --NTScore-dir=NTSCORE_DIR
                        Directory for the NTScore output.
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
