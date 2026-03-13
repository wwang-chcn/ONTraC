"""Niche network argument parsing and validation for ONTraC."""

import sys
from optparse import OptionGroup, OptionParser, Values
from typing import Optional

from ..log import *
from ._IO import *


# ------------------------------------
# Functions
# ------------------------------------
def add_niche_net_constr_options_group(optparser: OptionParser) -> None:
    """Add niche network construction options group to optparser.

    Parameters
    ----------
    optparser :
        OptionParser object.
    nn_options :
        Set of niche network construction options.

    Returns
    -------
    OptionGroup object."""
    # niche network construction options group
    group_niche = OptionGroup(optparser, "Niche Network Construction")
    group_niche.add_option(
        "--n-cpu",
        dest="n_cpu",
        type="int",
        default=4,
        help="Number of CPUs used for parallel computing in dataset preprocessing. Default is 4.",
    )
    group_niche.add_option(
        "--n-neighbors",
        dest="n_neighbors",
        type="int",
        default=50,
        help=(
            "Number of neighbors used for kNN graph construction. It should be "
            "less than the number of cells in each sample. Default is 50."
        ),
    )
    group_niche.add_option(
        "--n-local",
        dest="n_local",
        type="int",
        default=20,
        help=(
            "Nth closest local neighbor used for Gaussian distance normalization. "
            "Should be less than cells per sample. Default is 20."
        ),
    )
    group_niche.add_option(
        "--embedding-adjust",
        dest="embedding_adjust",
        action="store_true",
        default=False,
        help=(
            "Adjust cell type coding according to embeddings. Requires at least "
            "Embedding_1 and Embedding_2 columns when enabled."
        ),
    )
    group_niche.add_option(
        "--sigma",
        dest="sigma",
        type="float",
        default=1,
        help="Sigma for the exponential similarity kernel between cell types/clusters. Default is 1.",
    )
    optparser.add_option_group(group_niche)


def validate_niche_net_constr_options(options: Values, optparser: Optional[OptionParser] = None) -> None:
    """Validate niche network construction options.

    Parameters
    ----------
    options :
        Options object.
    optparser :
        OptionParser object.

    Returns
    -------
    None."""

    if getattr(options, "n_cpu", None) is None:
        info("n_cpu is not set. Using default value 4.")
        options.n_cpu = 4
    elif not isinstance(options.n_cpu, int):
        error("n_cpu must be an integer.")
        if optparser is not None:
            optparser.print_help()
        sys.exit(1)
    elif options.n_cpu < 1:
        error("n_cpu must be greater than 0.")
        if optparser is not None:
            optparser.print_help()
        sys.exit(1)

    if getattr(options, "n_neighbors", None) is None:
        info("n_neighbors is not set. Using default value 50.")
        options.n_neighbors = 50
    elif not isinstance(options.n_neighbors, int):
        error("n_neighbors must be an integer.")
        if optparser is not None:
            optparser.print_help()
        sys.exit(1)
    elif options.n_neighbors < 1:
        error("n_neighbors must be greater than 0.")
        if optparser is not None:
            optparser.print_help()
        sys.exit(1)

    if getattr(options, "n_local", None) is None:
        info("n_local is not set. Using default value 20.")
        options.n_local = 20
    elif not isinstance(options.n_local, int):
        error("n_local must be an integer.")
        if optparser is not None:
            optparser.print_help()
    elif options.n_local < 1:
        error("n_local must be greater than 0.")
        if optparser is not None:
            optparser.print_help()
        sys.exit(1)

    # embedding adjust
    if not hasattr(options, "embedding_adjust"):
        options.embedding_adjust = False

    missing_inputs = all(
        getattr(options, attr, None) is None
        for attr in ("embedding_input", "exp_input", "low_res_exp_input", "deconvoluted_exp_input")
    )
    if options.embedding_adjust and missing_inputs:
        error("Please provide an embedding file or expression data file in csv format.")
        if optparser is not None:
            optparser.print_help()
        sys.exit(1)

    if options.embedding_adjust:
        if not hasattr(options, "sigma"):
            options.sigma = 1
        elif options.sigma < 0:
            error("sigma must be greater than 0.")
            if optparser is not None:
                optparser.print_help()
            sys.exit(1)


def write_niche_net_constr_memo(options: Values) -> None:
    """Write niche network construction memos to stdout.

    Parameters
    ----------
    options :
        Options object.

    Returns
    -------
    None."""

    # print parameters to stdout
    info("      -------- niche net constr options -------      ")
    info(f"n_cpu:   {options.n_cpu}")
    info(f"n_neighbors: {options.n_neighbors}")
    info(f"n_local: {options.n_local}")
    info(f"embedding_adjust: {options.embedding_adjust}")
    if options.embedding_adjust:
        info(f"sigma: {options.sigma}")
