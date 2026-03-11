"""This module contains API to use R-based deconvolution method STdeconvolve for ONTraC."""

import subprocess
from importlib import resources
from pathlib import Path
from typing import Union

import pandas as pd


def apply_STdeconvolve(
    NN_dir: Union[str, Path], exp_df: pd.DataFrame, ct_num: int, gen_ct_embedding: bool = False
) -> pd.DataFrame:
    """Apply STdeconvolve to spot-level data.

    Parameters
    ----------
    NN_dir :
        Niche network directory.
    pd.DataFram :
        Expression dataFrame.  #spot x #gene
    ct_num :
        Number of cell types.
    gen_ct_embedding :
        Generate cell type embedding or not.

    Returns
    -------
    Deconvoluted expression DataFrame.  #spot x #cell_type"""

    # save expression matrix in csv format
    exp_matrix_file = f"{NN_dir}/filtered_spot_exp.csv.gz"
    exp_df.to_csv(exp_matrix_file, index=True, header=True, compression="gzip")

    spot_x_cell_type_file = "spot_x_celltype_deconvolution.csv"
    ct_x_gene_file = "celltype_x_gene_deconvolution.csv"

    with resources.path("ONTraC.external", "STdeconvolve.R") as stdeconvolve_script_path:
        try:
            stdeconvolve_script_path_str = str(stdeconvolve_script_path)
            if gen_ct_embedding:
                cmd = (
                    f"Rscript {stdeconvolve_script_path_str} {exp_matrix_file} "
                    f"{ct_num} {str(NN_dir)} {spot_x_cell_type_file} {ct_x_gene_file}"
                )
                subprocess.run(
                    cmd,
                    shell=True,
                )
            else:
                cmd = (
                    f"Rscript {stdeconvolve_script_path_str} {exp_matrix_file} "
                    f"{ct_num} {str(NN_dir)} {spot_x_cell_type_file}"
                )
                subprocess.run(
                    cmd,
                    shell=True,
                )
        except subprocess.CalledProcessError as e:
            print("Error in running R script:", e)
            print("R script stderr:", e.stderr)

    # handle deconvoluted results
    # 1) The first column name missing STdeconvolve outputed csv file
    # 2) "-" in spot_id will be converted to "."
    # SECTION: Normalize spot_x_cell_type_file output
    spot_x_cell_type_df = pd.read_csv(Path(NN_dir).joinpath(spot_x_cell_type_file))
    spot_x_cell_type_df.index = exp_df.columns
    spot_x_cell_type_df.to_csv(
        Path(NN_dir).joinpath(spot_x_cell_type_file), index=True, header=True, index_label="Spot_ID"
    )
    # SECTION: Normalize ct_x_gene_file output
    ct_x_gene_df = pd.read_csv(Path(NN_dir).joinpath(ct_x_gene_file))
    ct_x_gene_df.to_csv(Path(NN_dir).joinpath(ct_x_gene_file), index=True, header=True, index_label="Cell_Type")

    # load deconvoluted cell type composition data
    ct_coding_df = pd.read_csv(Path(NN_dir).joinpath(spot_x_cell_type_file), index_col=0)

    return ct_coding_df
