#!/usr/bin/env python

import sys

from ..optparser import opt_NT_validate, prepare_NT_optparser
from ..run.processes import NTScore, load_parameters
from ..utils import write_version_info


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
    options = load_parameters(opt_validate_func=opt_NT_validate, prepare_optparser_func=prepare_NT_optparser)

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
