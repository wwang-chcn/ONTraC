#!/usr/bin/env python

import sys

from ..log import *
from ..niche_net import gen_original_data
from ..optparser import (opt_preprocessing_validate,
                         prepare_preprocessing_optparser)
from ..run.processes import load_parameters, niche_network_construct
from ..utils import write_version_info


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
    options = load_parameters(opt_validate_func=opt_preprocessing_validate,
                              prepare_optparser_func=prepare_preprocessing_optparser)

    # load original data
    ori_data_df = gen_original_data(options=options)

    # ----- Niche Network Construct -----
    niche_network_construct(options=options, ori_data_df=ori_data_df)


# ------------------------------------
# Program running
# ------------------------------------
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("User interrupts me! ;-) See you ^.^!\n")
        sys.exit(0)
