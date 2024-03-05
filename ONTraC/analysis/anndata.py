import os
import re
from optparse import Values
from typing import Dict, Optional, Tuple

import anndata as ad
import matplotlib as mpl
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import torch
from anndata import AnnData
from matplotlib import cm
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from torch_geometric.data import Data

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import seaborn as sns

from ONTraC.log import *

def create_adata(options: Values) -> Tuple[Dict[str, AnnData], AnnData]:
    """Load dataset from options.
    """

    # load meta data
    meta_df = pd.read_csv(f'{options.input}/meta.csv', index_col=0)
    meta_df.index = meta_df.index.astype(str)

    # create AnnData with fake expression data
    adata_dict = {}
    for sample in options.data['Data']:
        name = sample.get('Name')
        coordinate_file = f"{sample['Coordinates']}"
        info(f'Read samples: {name} with coordinate file: {coordinate_file}.')
        coordinate_arr = np.loadtxt(coordinate_file, delimiter=',')
        adata_dict[name] = ad.AnnData(
            X=csr_matrix(np.random.poisson(1, (coordinate_arr.shape[0], 100)), dtype=np.float32))
        # load meta data
        adata_dict[name].obs = meta_df.loc[meta_df['sample'] == name]
        # load spatial data
        adata_dict[name].obsm['spatial'] = coordinate_arr

    adata_combined = ad.concat(list(adata_dict.values()))

    return adata_dict, adata_combined


def anndata_based_analysis(
    options: Values,
    data: Data,
) -> Optional[pd.DataFrame]:
    # ----- prepare data -----
    adata_dict, adata_combined = create_adata(options)
    # try:
    #     # --- load embedding data ---
    #     load_embedding_data(options, data, adata_dict, adata_combined)
    # except FileNotFoundError as e:
    #     warning(f"Skip anndata based analysis due to {e}.")
    #     return
    # try:
    #     # --- load graph pooling results ---
    #     load_graph_pooling_results(options, data, adata_dict, adata_combined)
    # except FileNotFoundError as e:
    #     warning(str(e))
    #     pass
    # try:
    #     # --- load pseudo time ---
    #     load_pseudo_time(options, data, adata_dict, adata_combined)
    # except FileNotFoundError as e:
    #     warning(str(e))
    #     pass
    # # ----- analysis -----
    # if 'trained_embedding' in adata_combined.obsm.keys():
    #     pass
    #     # 1. umap plots
    #     # umap_plots(options, 'trained_embedding', adata_dict, adata_combined)
    #     # 2. spatial plots
    #     # spatial_plots(options, 'trained_embedding', adata_dict)

    # # 3. plot pseudo time
    # plot_pseudo_time(options, data, adata_dict)

    # # 4. feat_along_pseudo_time
    # feat_along_pseudo_time(options, adata_dict, adata_combined)

    return adata_combined.obs
