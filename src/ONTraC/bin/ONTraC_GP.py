#!/usr/bin/env python
"""This module contains the main function for the `ONTraC_GP` compatibility pipeline, which will be deprecated in favor of `ONTraC_GT`."""

import sys

from ..log import warning
from ..optparser import opt_gt_validate, prepare_gt_optparser
from ..run.processes import niche_trajectory_construct, gnn, load_parameters
from ..utils import write_version_info

# ------------------------------------
# Classes
# ------------------------------------


# ------------------------------------
# Main Function
# ------------------------------------
def main() -> None:
    """Run deprecated `ONTraC_GP` compatibility pipeline (use `ONTraC_GT`)."""

    # write version information
    write_version_info()

    # deprecation warning
    warning('ONTraC_GP will be deprecated from v3.0. Please use ONTraC_GT instead.')

    # load parameters
    options = load_parameters(opt_validate_func=opt_gt_validate, prepare_optparser_func=prepare_gt_optparser)

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
