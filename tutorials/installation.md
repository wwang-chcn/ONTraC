# Installation

## Create and enter conda env (recommend)

```sh
conda create -y -n ONTraC python=3.11
conda activate ONTraC
```

### Add this kernel to jupyter (recommend)

```sh
pip install ipykernel
python -m ipykernel install --user --name ONTraC --display-name "Python 3.11 (ONTraC)"
```

## Install ONTraC

```sh
pip install ONTraC
# Use this command if you want to visualise the results by `ONTraC_analysis`.
pip install ONTraC[analysis]
```

## Install a developing version (V2)

```sh
pip install git+https://github.com/gyuanlab/ONTraC.git@V2
```