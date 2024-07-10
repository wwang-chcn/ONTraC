#!/usr/bin/env python

import sys
from typing import Callable, Optional

from ..log import *
from ..model import GraphPooling
from ..optparser import opt_ontrac_validate, prepare_ontrac_optparser
from ..run.processes import NTScore, gnn, load_parameters, niche_network_construct
from ..train import GPBatchTrain
from ..train.inspect_funcs import loss_record
from ..utils import load_original_data, write_version_info


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
    main function
    Input data files information should be stored in a YAML file.
    """

    # write version information
    write_version_info()

    # load parameters
    options = load_parameters(opt_validate_func=opt_ontrac_validate, prepare_optparser_func=prepare_ontrac_optparser)

    # load original data
    ori_data_df = load_original_data(options=options)

    # ----- Niche Network Construct -----
    niche_network_construct(options=options, ori_data_df=ori_data_df)

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
