from optparse import Values

from ..analysis.cell_type import cell_type_visualization
from ..analysis.data import AnaData
from ..analysis.niche_cluster import niche_cluster_visualization
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

    info('-------------- Analysis pipeline start -------------- ')
    step_index = 0

    # 0. load data class
    info(message=f'Analysis pipeline step {step_index}: load data.')
    ana_data = AnaData(options)
    step_index += 1

    # part 1: train loss
    info(message=f'Analysis pipeline step {step_index}: train loss visualization.')
    train_loss_visualiztion(ana_data=ana_data)
    step_index += 1

    # part 2: spatial based output
    info(message=f'Analysis pipeline step {step_index}: spatial-based visualization.')
    spatial_visualization(ana_data=ana_data)
    step_index += 1

    # part 3: niche cluster
    info(message=f'Analysis pipeline step {step_index}: niche cluster visualization.')
    niche_cluster_visualization(ana_data=ana_data)
    step_index += 1

    # part 4: cell type based output
    info(message=f'Analysis pipeline step {step_index}: cell type-based visualization.')
    cell_type_visualization(ana_data=ana_data)
    step_index += 1
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
