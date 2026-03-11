"""
This package contains the optparser module for ONTraC, which is responsible for parsing command-line arguments and options for ONTraC. It includes the main command-line interface (CLI) commands and subcommands for ONTraC, as well as the functions for preparing the optparser and validating the parsed options.

The optparser module is designed to provide a user-friendly and flexible interface for running ONTraC with various options and configurations. It allows users to specify input data, output directories, parameters for different steps of the ONTraC pipeline, and other settings through command-line arguments. The module also includes validation functions to ensure that the provided options are valid and consistent with the requirements of the ONTraC pipeline.
"""

from .command import *

# This part is designed as follows:
# 1) The prepare optparser and opt_validate functions are located in command.py
#    commands inside it are ONTraC, ONTraC_NN, ONTraC_GNN, ONTraC_NT, and ONTraC_GP
#    ONTraC_GNN contains train, GNN, and GP parts
# 2) submodules are _IO.py, _preprocessing.py, _NN.py, _train.py, _GNN.py and _NT.py
