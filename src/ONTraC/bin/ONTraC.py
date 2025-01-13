#!/usr/bin/env python

import sys

from ..optparser import opt_ontrac_validate, prepare_ontrac_optparser
from ..run.processes import niche_trajectory_construct, gnn, load_parameters, niche_network_construct
from ..utils import write_version_info


# ------------------------------------
# Main Function
# ------------------------------------
def main() -> None:
    """
    The main function
    """

    # write version information
    write_version_info()

    # load parameters
    options = load_parameters(opt_validate_func=opt_ontrac_validate, prepare_optparser_func=prepare_ontrac_optparser)

    # ----- Niche Network Construct -----
    niche_network_construct(options=options)

    # ----- GNN -----
    gnn(options=options)

    # ----- NT score -----
    niche_trajectory_construct(options=options)


# ------------------------------------
# Program running
# ------------------------------------
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("User interrupts me! ;-) See you ^.^!\n")
        sys.exit(0)
