import os
import sys
from optparse import OptionGroup, OptionParser, Values

from ..analysis.cell_type import cell_type_visualization
from ..analysis.data import AnaData
from ..analysis.niche_cluster import niche_cluster_visualization
from ..analysis.niche_net import clustering_visualization, embedding_adjust_visualization
from ..analysis.spatial import spatial_visualization
from ..analysis.train_loss import train_loss_visualiztion
from ..log import *
from ..optparser._IO import add_IO_options_group, validate_io_options, write_io_options_memo
from ..utils import *
from ..version import __version__

# ------------------------------------
# Constants
# ------------------------------------
IO_OPTIONS = ['preprocessing_dir', 'GNN_dir', 'NTScore_dir']


# ------------------------------------
# Functions
# ------------------------------------
def analysis_pipeline(options: Values) -> None:
    info(' -------------- Analysis pipeline start -------------- ')
    info('Analysis pipeline step 0: load data class.')
    # 0. load data class
    ana_data = AnaData(options)

    info('Analysis pipeline step 1: clustering visualization.')
    # part 1: clustering
    clustering_visualization(ana_data=ana_data)

    info('Analysis pipeline step 2: embedding adjust visualization.')
    # part 2: niche network construction
    embedding_adjust_visualization(ana_data=ana_data)

    info('Analysis pipeline step 3: train loss visualization.')
    # part 3: train loss
    train_loss_visualiztion(ana_data=ana_data)

    info('Analysis pipeline step 4: spatial-based visualization.')
    # part 4: spatial based output
    spatial_visualization(ana_data=ana_data)

    info('Analysis pipeline step 5: niche cluster visualization.')
    # part 5: niche cluster
    niche_cluster_visualization(ana_data=ana_data)

    info('Analysis pipeline step 6: cell type-based visualization.')
    # part 6: cell type based output
    if not options.suppress_cell_type:
        cell_type_visualization(ana_data=ana_data)

    info('--------------- Analysis pipeline end --------------- ')

# TODO: move to optparser
def add_suppress_group(optparser: OptionParser) -> None:
    group = OptionGroup(optparser, 'Suppress options')
    group.add_option('--suppress-cell-type-composition',
                     dest='suppress_cell_type_composition',
                     action='store_true',
                     default=False,
                     help='Suppress the cell type composition visualization.')
    group.add_option('--suppress-niche-cluster-loadings',
                     dest='suppress_niche_cluster_loadings',
                     action='store_true',
                     default=False,
                     help='Suppress the niche cluster loadings visualization.')
    group.add_option('--suppress-cell-type',
                     dest='suppress_cell_type',
                     action='store_true',
                     default=False,
                     help='Suppress the cell type visualization.')
    group.add_option('--suppress-niche-trajectory',
                        dest='suppress_niche_trajectory',
                        action='store_true',
                        default=False,
                        help='Suppress the niche trajectory related visualization.')
    optparser.add_option_group(group)


def add_embedding_adjust_group(optparser: OptionParser) -> None:
    group = OptionGroup(optparser, 'Embedding adjust options')
    group.add_option(
        '--embedding-adjust',
        dest='embedding_adjust',
        action='store_true',
        default=False,
        help=
        'Adjust the cell type coding according to embeddings. Default is False. At least two (Embedding_1 and Embedding_2) should be in the original data if embedding_adjust is True.'
    )
    group.add_option(
        '--sigma',
        dest='sigma',
        type='float',
        default=1,
        help=
        'Sigma for the exponential function to control the similarity between different cell types or clusters. Default is 1.'
    )
    optparser.add_option_group(group)


def prepare_optparser() -> OptionParser:
    """Prepare optparser object. New options will be added in this function first.

    Ret: OptParser object.
    """
    usage = "USAGE: %prog <--preprocessing-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR> <--NTScore-dir NTSCORE_DIR> <-o OUTPUT_DIR> [-l LOG_FILE] [-r REVERSE]"
    description = "Analysis the results of ONTraC."
    optparser = OptionParser(version=f'%prog {__version__}',
                             usage=usage,
                             description=description,
                             add_help_option=False)
    optparser.add_option('-h', '--help', action='help', help='Show this help message and exit.')
    add_IO_options_group(optparser=optparser, io_options=IO_OPTIONS)
    optparser.add_option('-l', '--log', dest='log', type='string', help='Log file.')
    optparser.add_option('-o', '--output', dest='output', type='string', help='Output directory.')
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
    add_embedding_adjust_group(optparser)
    add_suppress_group(optparser)
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

    if not os.path.exists(options.log):
        error(f'Log file not found: {options.log}')
        sys.exit(1)

    options.yaml = f'{options.preprocessing_dir}/samples.yaml'
    if not os.path.exists(options.yaml):
        error(f'File not found: {options.yaml}')
        sys.exit(1)
    options.device = 'cpu'

    info(' ------------------ RUN params memo ------------------ ')
    # print parameters to stdout
    write_io_options_memo(options, IO_OPTIONS)
    info(f'Output directory: {options.output}')
    info(f'Log file: {options.log}')
    info('       -------- ONTraC running options --------        ')
    info(f'Reverse: {options.reverse}')
    info(f'Sample: {options.sample}')
    info(f'Embedding adjust: {options.embedding_adjust}')
    if options.embedding_adjust:
        info(f'Sigma: {options.sigma}')
    info('         -------- Suppression options --------         ')
    if hasattr(options, 'suppress_cell_type_composition'):
        info(f'Suppress cell type composition: {options.suppress_cell_type_composition}')
    if hasattr(options, 'suppress_niche_cluster_loadings'):
        info(f'Suppress niche cluster loadings: {options.suppress_niche_cluster_loadings}')
    if hasattr(options, 'suppress_niche_trajectory'):
        info(f'Suppress niche trajectory: {options.suppress_niche_trajectory}')
    info(' ---------------- RUN params memo end ---------------- ')

    return options


def main():

    # write version information
    write_version_info()

    optparser = prepare_optparser()
    options = opt_validate(optparser)

    analysis_pipeline(options)


if __name__ == '__main__':
    main()
