#!/usr/bin/env python

import os
import sys
from typing import Callable, Optional

from ..model import GraphPooling
from ..optparser import opt_GP_validate, prepare_GP_optparser
from ..run.processes import NTScore, gnn, load_parameters
from ..train import GPBatchTrain
from ..train.inspect_funcs import loss_record
from ..utils import load_meta_data, write_version_info

# ------------------------------------
# Classes
# ------------------------------------


# ------------------------------------
# Functions
# ------------------------------------
def get_inspect_funcs() -> Optional[list[Callable]]:
    """
    Inspect function list
    :param output_dir: output dir
    :param epoch_filter: epoch filter
    :return: list of inspect functions
    """
    return [loss_record]


# ------------------------------------
# Main Function
# ------------------------------------
def main() -> None:
    """
    Main function
    :return: None
    """

    # write version information
    write_version_info()

    # load parameters
    options = load_parameters(opt_validate_func=opt_GP_validate, prepare_optparser_func=prepare_GP_optparser)

    # load meta data
    options.meta_input = os.path.join(options.preprocessing_dir, 'meta_data.csv')
    if not os.path.exists(options.meta_input):
        raise FileNotFoundError(f"Meta data file not found: {options.meta_input}. You may need to run createDataSet first or copy meta data file into {options.preprocessing_dir} directory with the name 'meta_data.csv'.")
    meta_data_df = load_meta_data(options=options)

    # ----- GNN -----
    gnn(options=options,
        meta_data_df=meta_data_df,
        nn_model=GraphPooling,
        BatchTrain=GPBatchTrain,
        inspect_funcs=get_inspect_funcs())

    # ----- NT score -----
    NTScore(options=options)


# ------------------------------------
# Program running
# ------------------------------------
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("User interrupts me! ;-) See you ^.^!\n")
        sys.exit(0)
