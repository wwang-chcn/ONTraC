from typing import Dict, List, Set, Union

IO_OPTIONS: Dict[str, Set[Union[str, None]]] = {
    'ONTraC_NN': set(['NN_dir', 'input', 'meta']),  # niche network
    'ONTraC_GNN': set(['NN_dir', 'GNN_dir']),  # graph neural network
    'ONTraC_NT': set(['NN_dir', 'GNN_dir', 'NT_dir']),  # niche trajectory
    'ONTraC_analysis': set(['NN_dir', 'GNN_dir', 'NT_dir', 'meta', 'output', 'log']),  # analysis
}
IO_OPTIONS['ONTraC'] = IO_OPTIONS['ONTraC_NN'] | IO_OPTIONS['ONTraC_GNN'] | IO_OPTIONS['ONTraC_NT']
IO_OPTIONS['ONTraC_GT'] = IO_OPTIONS['ONTraC_GNN'] | IO_OPTIONS['ONTraC_NT']

