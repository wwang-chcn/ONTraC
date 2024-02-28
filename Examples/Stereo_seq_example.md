# Stereo-seq Mouse Midbrain

## Step1 generate niche GNN input files

```R
# gaussian affinity between two cells (a cell and its KNN)
gaussDist <- function(dist.use = NULL, sigma.use = NULL){
  exp(-(dist.use/sigma.use)^2)
}

# weighted composition for a target cell given its KNN
WeightedCompositionSC <- function(annot.use = NULL, w = NULL, k.annot = NULL){
  wc.use <- sapply(annot.use,function(x){
    idx.use <- which(k.annot == x)
    return (sum(w[idx.use]))
  })
  return (wc.use)
}

# wrapper function: distance-weighted composition of a group of cells within the same tissue
# loc.use: dataframe with rownames for cells
# annot.sc: named single cell cluster label
# knn.spat: nn2 output (list of nn.dists and nn.idx, each a matrix)
WeightedComposition <- function(loc.use = NULL,knn.spat = NULL,k.use = 50,
                                nlocal = 20,annot.sc = NULL){
  if (is.null(knn.spat)){
    knn.spat <- RANN::nn2(data = loc.use,k = k.use)
  }
  sig.use <- apply(knn.spat$nn.dists,1,function(x) x[nlocal])
  w.gauss <- lapply(1:nrow(knn.spat$nn.dists),function(x){
    gaussDist(knn.spat$nn.dists[x,],sig.use[x])
  })
  w.gauss <- Reduce(rbind,w.gauss)
  rownames(w.gauss) <- rownames(loc.use)
  
  knn.spat.annot <- t(apply(knn.spat$nn.idx,1,function(x) 
    as.character(annot.sc[rownames(loc.use)[x]])))
  # annot.use <- sort(unique(as.character(annot.sc)))
  annot.use <- levels(annot.sc)
  wcomp.gauss <- sapply(1:nrow(knn.spat.annot),function(x){
    wc <- WeightedCompositionSC(annot.use = annot.use,w = w.gauss[x,],k.annot = knn.spat.annot[x,])
    return (wc)
  })
  wcomp.gauss <- t(wcomp.gauss)
  rownames(wcomp.gauss) <- rownames(loc.use)
  return (list(knn.spat,w.gauss,knn.spat.annot,wcomp.gauss))
}

origin_df = data.frame(data.table::fread('orgianl_data.csv'))
colnames(origin_df) = c("cells", "sample", "annot", "x", "y")
origin_df$annot = factor(origin_df$annot, c('RGC','GlioB','NeuB','GluNeuB','GluNeu','GABA','Basal','Fibro','Endo','Ery'))

samples.use = sort(unique(origin_df$sample))

ss0.comp.all = lapply(samples.use,function(a){
  loc.use = origin_df[origin_df$sample == a,c('x','y','cells')]
  rownames(loc.use) = loc.use$cells
  loc.use = loc.use[,c('x','y')]
  write.table(loc.use, paste0(a, '_Coordinates.csv'), quote = F, sep = ',', row.names = F, col.names = F)
  annot.sc = origin_df[origin_df$sample == a,'annot']
  names(annot.sc) <- rownames(loc.use)
  wcomp.gauss = WeightedComposition(loc.use = loc.use,annot.sc = annot.sc)
  # TODO normalized wcomp.gauss
  write.table(wcomp.gauss, paste0(a, '_NodeAttr.csv'), quote = F, sep = ',', row.names = F, col.names = F)
})
names(ss0.comp.all) <- samples.use
```

```python
import pandas as pd

meta_df = pd.read_csv('orgianl_data.csv')
meta_df['annot'] = meta_df['annot'].astype('category')

for name, group in meta_df.groupby(by='sample'):
    np.savetxt(fname = label_file, X=group['annot'].cat.codes.values, delimiter=',')
```

## Step2 Run GNN to get niche trajectory coordinates

`ONTraC` should located at `/path/to/OnTraC` directory

### Step2.1 Input files prepare

Input and configure files should be stored in `data/stereo_seq_brain`

```bash
export PYTHONPATH=`pwd`:/path/to/ONTraC/:${PYTHONPATH}
/path/to/ONTraC/bin/createDataSet.py --oc data/stereo_seq_brain_process -y data/stereo_seq_brain/data.yaml --n-cpu 16 --n-neighbors 50
```

### Step2.2 Run GNN

```bash
export PYTHONPATH=`pwd`:/path/to/ONTraC/:${PYTHONPATH}
python /path/to/ONTraC/bin/GP.py -i data/stereo_seq_brain_process/ --oc output/stereo_seq_brain --device cuda     --epochs 1000 --batch-size 5 -s 42 --patience 100 --min-delta 0.001 --min-epochs 50 --lr 0.03 --hidden-feats 4     -k 6 --dropout 0 --spectral-loss-weight 1 --ortho-loss-weight 0 --cluster-loss-weight 0.01     --feat-similarity
-loss-weight 10 --init-node-label 9 --assign-exponent 10 > log/stereo_seq_brain.log
```

In the output directory `output/stereo_seq_brain`:

`consolidate_s.csv.gz` contains the probabilistic matrix for each niche to niche cluster.

`consolidate_out_adj.csv.gz` contains the adjacency matrix for niche clusters.

`NT_niche_cluster.csv.gz` contains the niche trajectory coordinates for each niche cluster.

`NT_niche.csv.gz` contains the niche trajectory coordinates for each niche.

## Step3 Map niche trajectory coordinates to cell-level niche trajectory coordinates (R code)

```R
project_NT <- function(loc.use = NULL, knn.spat = NULL, k.use = 50, nlocal = 20, niche_value = NULL) {
  # in case of no knn input, create it
  if (is.null(knn.spat)){
    knn.spat <- RANN::nn2(data = loc.use,k = k.use)
  }
  # calculate sigma
  sig.use <- apply(knn.spat$nn.dists,1,function(x) x[nlocal])
  w.gauss <- lapply(1:nrow(knn.spat$nn.dists),function(x){
    gaussDist(knn.spat$nn.dists[x,],sig.use[x])
  })
  w.gauss <- Reduce(rbind,w.gauss)
  w.gauss <- t(apply(w.gauss, 1, function(x) {x / sum(x)}))  # normalize
  rownames(w.gauss) <- rownames(loc.use)

  knn.spat.value <- t(apply(knn.spat$nn.idx,1,function(x) 
    niche_value[x]))
  
  cell_niche_value <- apply(w.gauss * knn.spat.value, 1, sum)
    
  rownames(cell_niche_value) <- rownames(cell_niche_value)
  return(cell_niche_value)
  
}
    
origin_df = data.frame(data.table::fread('orgianl_data.csv.gz'))
colnames(origin_df) = c("cells", "sample", "annot", "x", "y")

NT_niche = as.data.frame(data.table::fread('NT_niche.csv.gz'), )
origin_df$dpt_gnn_niche = NT_niche$V1[1:dim(origin_df)[[1]]]

cell.niche.all <- lapply(samples.use,function(a){
  loc.use = origin_df[origin_df$sample == a,c('x','y')]
  niche_value <- origin_df$dpt_gnn_niche
  # niche_value <- mop_all_sub@meta.data[rownames(loc.use),'dpt_gnn_niche']
  # names(niche_value) <- rownames(loc.use)
  return (project_NT(loc.use = loc.use, niche_value=niche_value))
  
})

cell.niche.all <- as.data.frame(unlist(cell.niche.all))
colnames(cell.niche.all) <- c("dpt_niche_cell")

origin_df$dpt_niche_cell = cell.niche.all$dpt_niche_cell
```
