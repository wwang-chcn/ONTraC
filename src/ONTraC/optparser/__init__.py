from .command import *

# This part is designed as follows:
# 1) The prepare optparser and opt_validate functions are located in command.py
#    commands inside it are ONTraC, ONTraC_NN, ONTraC_GNN, ONTraC_NT, and ONTraC_GP
#    ONTraC_GNN contains train, GNN, and GP parts
# 2) submodules are _IO.py, _preprocessing.py, _NN.py, _train.py, _GNN.py and _NT.py
