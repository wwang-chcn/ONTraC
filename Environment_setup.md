# Conda environment setup process

Construct ONTraC environment on Linux using Conda:

First, create ONTrac environment using conda.

```bash
conda create -y -n ONTraC python=3.11
```

Second, install the dependency packages.

```bash
conda activate ONTraC
pip install pyyaml pandas==2.2.1 torch==2.2.1 torch_geometric==2.5.0
```
