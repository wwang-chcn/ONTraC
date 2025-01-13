# Installation

## Step 1 Create and enter conda env (recommend)

```sh
conda create -y -n ONTraC python=3.11  # ONTraC supports Python 3.10, 3.11, and 3.12 for now
conda activate ONTraC
```

### Add this kernel to jupyter (recommend)

```sh
pip install ipykernel
python -m ipykernel install --user --name ONTraC --display-name "Python 3.11 (ONTraC)"
```

## Step2 Install ONTraC

### using pip

```sh
pip install ONTraC
# Use this command if you want to visualise the results by `ONTraC_analysis`.
pip install "ONTraC[analysis]"
```

### using conda

NOTE: For ARM-based macOS, we recommend installing via pip for now, as the dependency package, pytorh-geometric, does not have a conda build for it.
NOTE: For x86-based macOS, conda install only support Python 3.10 and 3.11 for now.
WARNING: Installing with conda can be very slow.

```sh
conda install -c gyuanlab -c pytorch -c pyg -c default -c nvidia -c conda-forge ontrac
```

## Install a developing version (V2)

```sh
pip install git+https://github.com/gyuanlab/ONTraC.git@V2
```
