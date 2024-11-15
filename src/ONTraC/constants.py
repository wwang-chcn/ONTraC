from typing import Dict, List

IO_OPTIONS: Dict[str, List[str]] = {
    'ONTraC': ['NN_dir', 'GNN_dir', 'NT_dir', 'input'],
    'ONTraC_NN': ['NN_dir', 'input'],  # niche network
    'ONTraC_GNN': ['NN_dir', 'GNN_dir'],  # graph neural network
    'ONTraC_NT': ['NN_dir', 'GNN_dir', 'NT_dir'],  # niche trajectory
    'ONTraC_GT': ['NN_dir', 'GNN_dir', 'NT_dir'],  # GNN & niche trajectory
    'ONTraC_analysis': ['NN_dir', 'GNN_dir', 'NT_dir', 'input', 'output', 'log'],  # analysis
}
