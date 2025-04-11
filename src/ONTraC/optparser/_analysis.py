import sys
from optparse import OptionGroup, OptionParser, Values
from typing import Optional

from ..log import *


def add_visualization_group(optparser: OptionParser) -> None:
    group_vis = OptionGroup(optparser, 'Visualization options')
    group_vis.add_option(
        '--embedding-adjust',
        dest='embedding_adjust',
        action='store_true',
        default=False,
        help=
        'Adjust the cell type coding according to embeddings. Default is False. At least two (Embedding_1 and Embedding_2) should be in the original data if embedding_adjust is True.'
    )
    group_vis.add_option(
        '--sigma',
        dest='sigma',
        type='float',
        default=1,
        help=
        'Sigma for the exponential function to control the similarity between different cell types or clusters. Default is 1.'
    )
    group_vis.add_option('-r',
                         '--reverse',
                         dest='reverse',
                         action='store_true',
                         default=False,
                         help='Reverse the NT score during visualization.')
    group_vis.add_option('-s',
                         '--sample',
                         dest='sample',
                         action='store_true',
                         default=False,
                         help='Plot each sample separately.')
    group_vis.add_option('--scale-factor',
                         dest='scale_factor',
                         type='float',
                         default=1.0,
                         help='Scale factor control the size of spatial-based plots.')
    optparser.add_option_group(group_vis)


def add_suppress_group(optparser: OptionParser) -> None:
    group = OptionGroup(optparser, 'Suppress options')
    group.add_option(
        '--suppress-cell-type-composition',
        dest='suppress_cell_type_composition',
        action='store_true',
        default=False,
        help='Skip the cell type composition visualization. It would be useful when the number of cell types is large.')
    group.add_option(
        '--suppress-niche-cluster-loadings',
        dest='suppress_niche_cluster_loadings',
        action='store_true',
        default=False,
        help=
        'Skip the niche cluster loadings visualization. It would be useful when the number of clusters or sample size is large.'
    )
    group.add_option('--suppress-niche-trajectory',
                     dest='suppress_niche_trajectory',
                     action='store_true',
                     default=False,
                     help='Skip the niche trajectory related visualization.')
    optparser.add_option_group(group)


def validate_visualization_options(options: Values, optparser: Optional[OptionParser] = None) -> None:
    """
    Validate visualization options.

    Parameters
    ----------
    options : Values
        Options object.
    optparser : Optional[OptionParser], optional
        OptionParser object. The default is None.

    Returns
    -------
    None
    """

    # reverse
    if getattr(options, 'reverse', None) is None:
        info('reverse is not set. Using default value False.')
        options.reverse = False
    elif not isinstance(options.reverse, bool):
        error(f'reverse must be a boolean, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)

    # sample
    if getattr(options, 'sample', None) is None:
        info('sample is not set. Using default value False.')
        options.sample = False
    elif not isinstance(options.sample, bool):
        error(f'sample must be a boolean, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)

    # scale_factor
    if getattr(options, 'scale_factor', None) is None:
        info('scale_factor is not set. Using default value 1.0.')
        options.scale_factor = 1.0
    elif not isinstance(options.scale_factor, float):
        error(f'scale_factor must be a float, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)
    elif options.scale_factor <= 0:
        error(f'scale_factor must be greater than 0, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)
    elif options.scale_factor > 10:
        warning(f'scale_factor is too large, it may cause the plot to be too large.')

    # embedding_adjust
    if options.embedding_adjust:
        if options.sigma <= 0:
            error('Sigma must be greater than 0.')


def validate_suppress_options(options: Values, optparser: Optional[OptionParser] = None) -> None:
    """
    Validate suppress options.
    
    Parameters
    ----------
    options : Values
        Options object.
    optparser : Optional[OptionParser], optional
        OptionParser object. The default is None.

    Returns
    -------
    None
    """

    # suppress_cell_type_composition
    if getattr(options, 'suppress_cell_type_composition', None) is None:
        info('suppress_cell_type_composition is not set. Using default value False.')
        options.suppress_cell_type_composition = False
    elif not isinstance(options.suppress_cell_type_composition, bool):
        error(f'suppress_cell_type_composition must be a boolean, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)

    # suppress_niche_cluster_loadings
    if getattr(options, 'GNN_dir', None) is None:
        info(
            'GNN_dir is not set. Setting suppress_niche_cluster_loadings to True to skip the niche cluster loadings visualization.'
        )
        options.suppress_niche_cluster_loadings = True
    if getattr(options, 'suppress_niche_cluster_loadings', None) is None:
        info('suppress_niche_cluster_loadings is not set. Using default value False.')
        options.suppress_niche_cluster_loadings = False
    elif not isinstance(options.suppress_niche_cluster_loadings, bool):
        error(f'suppress_niche_cluster_loadings must be a boolean, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)

    # suppress_niche_trajectory
    if getattr(options, 'NT_dir', None) is None:
        info(
            'NT_dir is not set. Setting suppress_niche_trajectory to True to skip the niche trajectory related visualization.'
        )
        options.suppress_niche_trajectory = True
    if getattr(options, 'suppress_niche_trajectory', None) is None:
        info('suppress_niche_trajectory is not set. Using default value False.')
        options.suppress_niche_trajectory = False
    elif not isinstance(options.suppress_niche_trajectory, bool):
        error(f'suppress_niche_trajectory must be a boolean, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)


def write_visualization_options_memo(options: Values) -> None:
    """
    Write visualization options memo.
    :param options: Values object.
    :return: None.
    """

    info(message='---------------- Visualization options ----------------')
    info(f'embedding_adjust: {options.embedding_adjust}')
    if options.embedding_adjust:
        info(f'sigma: {options.sigma}')
    info(message=f'Reverse the NT score during visualization: {options.reverse}')
    info(message=f'Plot each sample separately: {options.sample}')
    info(message=f'Scale factor control the size of spatial-based plots: {options.scale_factor}')


def write_suppress_options_memo(options: Values) -> None:
    """
    Write suppress options memo.
    :param options: Values object.
    :return: None.
    """

    info(message='---------------- Suppress options ----------------')
    info(message=f'Suppress the cell type composition visualization: {options.suppress_cell_type_composition}')
    info(message=f'Suppress the niche cluster loadings visualization: {options.suppress_niche_cluster_loadings}')
    info(message=f'Suppress the niche trajectory related visualization: {options.suppress_niche_trajectory}')
