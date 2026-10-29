# Installation

Curve2Flood runs in a conda environment, which installs GDAL and the other libraries it needs.

1. Install [Miniforge](https://conda-forge.org/download/) or [Miniconda](https://docs.anaconda.com/miniconda/) if you
   don't have conda yet.
2. Get the code, with [Git](https://git-scm.com/downloads) or by downloading it as a zip file from GitHub:
   ```bash
   git clone https://github.com/MikeFHS/curve2flood
   cd curve2flood
   ```
3. Create the environment from `environment.yml` and activate it.
   ```bash
   conda env create -f environment.yml
   conda activate curve2flood
   ```
   To use an environment you already have, such as ARC's, install the libraries listed in
   [environment.yml](https://github.com/MikeFHS/curve2flood/blob/main/environment.yml) into it instead.
4. Install Curve2Flood into the environment:
   ```bash
   pip install --no-deps .
   ```
   If you will change Curve2Flood's code, install it with `pip install --no-deps -e .` instead.
5. Check that it works:
   ```bash
   curve2flood -h
   ```

To run the tests as well, see [Development](development.md).
