#!/usr/bin/env python
"""This module contains the main function for the `ONTraC_NT` CLI pipeline for niche trajectory scoring."""

import sys

from ..optparser import opt_nt_validate, prepare_nt_optparser
from ..run.processes import niche_trajectory_construct, load_parameters
from ..utils import write_version_info


# ------------------------------------
# Main Function
# ------------------------------------
def main() -> None:
    """Run the `ONTraC_NT` CLI pipeline for niche trajectory scoring."""

    # write version information
    write_version_info()

    # load parameters
    options = load_parameters(opt_validate_func=opt_nt_validate, prepare_optparser_func=prepare_nt_optparser)

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
