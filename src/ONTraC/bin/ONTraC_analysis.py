from optparse import Values

from ..analysis.cell_type import cell_type_visualization
from ..analysis.data import AnaData
from ..analysis.niche_cluster import niche_cluster_visualization
from ..analysis.niche_net import clustering_visualization_from_anadata, embedding_adjust_visualization_from_anadata
from ..analysis.spatial import spatial_visualization
from ..analysis.train_loss import train_loss_visualiztion
from ..log import *
from ..optparser import opt_analysis_validate, prepare_analysis_optparser
from ..run.processes import load_parameters
from ..utils import *
from ..version import __version__


# ------------------------------------
# Functions
# ------------------------------------
def analysis_pipeline(options: Values) -> None:
    """
    ONTraC analysis pipeline
    """

    # 0. load data class
    ana_data = AnaData(options)

    step_index = 0
    if ana_data.options.spatial_res == 'cell':
        step_index += 1
        info(message=f'Analysis pipeline step {step_index}: clustering visualization.')
        # part 1: clustering
        clustering_visualization_from_anadata(ana_data=ana_data)

    if ana_data.options.embedding_adjust:
        step_index += 1
        info(message=f'Analysis pipeline step {step_index}: embedding adjust visualization.')
        # part 2: niche network construction
        embedding_adjust_visualization_from_anadata(ana_data=ana_data)

    info(message=f'Analysis pipeline step {step_index}: train loss visualization.')
    # part 3: train loss
    train_loss_visualiztion(ana_data=ana_data)

    info(message=f'Analysis pipeline step {step_index}: spatial-based visualization.')
    # part 4: spatial based output
    spatial_visualization(ana_data=ana_data)

    info(message=f'Analysis pipeline step {step_index}: niche cluster visualization.')
    # part 5: niche cluster
    niche_cluster_visualization(ana_data=ana_data)

    info(message=f'Analysis pipeline step {step_index}: cell type-based visualization.')
    # part 6: cell type based output
    cell_type_visualization(ana_data=ana_data)

    info('--------------- Analysis pipeline end --------------- ')


def main():
    """
    The main function
    """

    # write version information
    write_version_info()

    options = load_parameters(opt_validate_func=opt_analysis_validate,
                              prepare_optparser_func=prepare_analysis_optparser)

    analysis_pipeline(options)


if __name__ == '__main__':
    main()
