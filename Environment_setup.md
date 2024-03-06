# Conda environment setup process

The following are two options to construct ONTraC environment on Linux using Conda:

## 1) Use environment.yml file
First, create the base conda environment using the provided environment.yml file and activate it.
```bash
conda env create -n ONTraC -f environment.yml
conda activate ONTraC
```
Once the new environment is activated, the remaining packages that require `pip` with the custom index URL can be downloaded.
```bash
pip install pyg_lib==0.2.0 torch_scatter==2.1.1 torch_sparse==0.6.17 torch_cluster==1.6.1 torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch_geometric==2.3.1
```

## 2) Linux commands for environment setup
Alternatively, the following commands could be used to create the conda environment. 
```bash
conda create -n ONTraC python=3.11
conda activate ONTraC
conda install -y -c pytorch -c nvidia -c conda-forge -c bioconda -c defaults gcc
conda install -y -c pytorch -c nvidia -c conda-forge -c bioconda -c defaults pyyaml scipy=1.11.2 scikit-learn=1.3.0 more-itertools
conda install -y -c pytorch -c nvidia -c conda-forge -c bioconda -c defaults pytorch=2.0.1 torchvision=0.15.2 torchaudio=2.0.2 pytorch-cuda=11.8
pip install pyg_lib==0.2.0 torch_scatter==2.1.1 torch_sparse==0.6.17 torch_cluster==1.6.1 torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch_geometric==2.3.1
conda install -y -c pytorch -c nvidia -c conda-forge -c bioconda -c pyg -c defaults pandas=2.1.1
conda install -y -c pytorch -c nvidia -c conda-forge -c bioconda -c pyg -c defaults scanpy=1.9.5
```
