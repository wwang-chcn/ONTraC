import os
import sys
from optparse import OptionParser, Values

from ..analysis.cell_type import cell_type_visualization
from ..analysis.data import AnaData
from ..analysis.niche_cluster import niche_cluster_visualization
from ..analysis.spatial import spatial_visualization
from ..analysis.train_loss import train_loss_visualiztion
from ..log import *
from ..optparser._IO import add_IO_options_group, validate_io_options
from ..utils import *

# ------------------------------------
# Constants
# ------------------------------------
IO_OPTIONS = ['dataset', 'preprocessing_dir', 'GNN_dir', 'NTScore_dir']


# ------------------------------------
# Functions
# ------------------------------------
def analysis_pipeline(options: Values) -> None:
    # 0. load data class
    ana_data = AnaData(options)

    # part 1: train loss
    train_loss_visualiztion(ana_data=ana_data)

    # part 2: spatial based output
    spatial_visualization(ana_data=ana_data)

    # part 3: niche cluster
    niche_cluster_visualization(ana_data=ana_data)

    # part 4: cell type based output
    cell_type_visualization(ana_data=ana_data)


def prepare_optparser() -> OptionParser:
    """Prepare optparser object. New options will be added in this function first.

    Ret: OptParser object.
    """
    usage = "usage: %prog <-d DATASET> <--preprocessing-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR> <--NTScore-dir NTSCORE_DIR> <-l LOG_FILE> <-o OUTPUT_DIR> [-r REVERSE]"
    description = "Analysis the results of ONTraC."
    optparser = OptionParser(usage=usage, description=description, add_help_option=False)
    optparser.add_option('-h', '--help', action='help', help='Show this help message and exit.')
    optparser.add_option('-o', '--output', dest='output', type='string', help='Output directory.')
    optparser.add_option('-l', '--log', dest='log', type='string', help='Log file.')
    optparser.add_option('-r',
                         '--reverse',
                         dest='reverse',
                         action='store_true',
                         default=False,
                         help='Reverse the NT score.')
    optparser.add_option('-s',
                         '--sample',
                         dest='sample',
                         action='store_true',
                         default=False,
                         help='Plot each sample separately.')
    add_IO_options_group(optparser=optparser, io_options=IO_OPTIONS)
    return optparser


def opt_validate(optparser: OptionParser) -> Values:
    """Validate options from a OptParser object.

    Args:
        optparser: OptParser object.
    """
    (options, args) = optparser.parse_args()

    validate_io_options(optparser, options, IO_OPTIONS, overwrite_validation=False)

    if not options.output:
        error('Output directory is required.')
        sys.exit(1)
    if not os.path.isdir(options.output):
        info(f'Output directory not found: {options.output}, will create it.')
        os.makedirs(options.output)
    else:
        warning(f'Output directory already exists: {options.output}, will overwrite it.')

    if not options.log:
        error('Log file is required.')
        sys.exit(1)
    if not os.path.exists(options.log):
        error(f'Log file not found: {options.log}')
        sys.exit(1)

    options.yaml = f'{options.preprocessing_dir}/samples.yaml'
    if not os.path.exists(options.yaml):
        error(f'File not found: {options.yaml}')
        sys.exit(1)
    options.device = 'cpu'

    return options


def main():

    # write version information
    write_version_info()

    optparser = prepare_optparser()
    options = opt_validate(optparser)

    analysis_pipeline(options)


if __name__ == '__main__':
    main()
