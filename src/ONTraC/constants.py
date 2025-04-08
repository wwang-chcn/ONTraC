from typing import Dict, List

# I/O options for different ONTraC modules
# I/O options belongs to different categories, such as optional, required, overwrite, etc.
# The options are defined in the following dictionary.
# The keys are the module names and the values are the dict of I/O options belong to different categories.

IO_OPTIONS: Dict[str, Dict[str, List[str]]] = {
    'ONTraC': {
        'required': ['input'],
        'optional': [],
        'overwrite': ['NN_dir', 'GNN_dir', 'NT_dir'],
    },
    'ONTraC_NN': {
        'required': ['input'],
        'optional': [],
        'overwrite': ['NN_dir'],
    },
    'ONTraC_GNN': {
        'required': ['NN_dir'],
        'optional': [],
        'overwrite': ['GNN_dir'],
    },
    'ONTraC_NT': {
        'required': ['NN_dir', 'GNN_dir'],
        'optional': [],
        'overwrite': ['NT_dir'],
    },
    'ONTraC_GT': {
        'required': ['NN_dir'],
        'optional': [],
        'overwrite': ['GNN_dir', 'NT_dir'],
    },
    'ONTraC_analysis': {
        'required': [],
        'optional': ['output', 'log'],
        'overwrite': ['NN_dir', 'GNN_dir', 'NT_dir'],
    },
}
