#!/usr/bin/env python

import sys
from typing import Callable, Optional

from ..model import GraphPooling
from ..optparser import opt_GP_validate, prepare_GP_optparser
from ..run.processes import NTScore, gnn, load_parameters
from ..train import GPBatchTrain
from ..train.inspect_funcs import loss_record
from ..utils import load_original_data, write_version_info

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

    # load original data
    ori_data_df = load_original_data(options=options)

    # ----- GNN -----
    gnn(options=options,
        ori_data_df=ori_data_df,
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
