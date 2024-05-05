#!/usr/bin/env python

import sys

from ..log import *
from ..optparser import opt_create_ds_validate, prepare_create_ds_optparser
from ..utils import write_version_info
from ..utils.niche_net_constr import construct_niche_network, gen_samples_yaml, load_original_data


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
    
    # prepare options
    options = opt_create_ds_validate(prepare_create_ds_optparser())

    # load original data
    ori_data_df = load_original_data(options=options)

    # define edges for each sample
    construct_niche_network(options=options, ori_data_df=ori_data_df)

    # save samples.yaml
    gen_samples_yaml(options=options, ori_data_df=ori_data_df)


# ------------------------------------
# Program running
# ------------------------------------
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("User interrupts me! ;-) See you ^.^!\n")
        sys.exit(0)
