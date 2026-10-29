# Curve2Flood

Curve2Flood turns river flows into flood inundation maps. It reads the rating curves that
[ARC](https://github.com/MikeFHS/automated-rating-curve) builds for every stream cell of a DEM, finds each stream
cell's water surface for a flow, and spreads that water over the land. It writes rasters of the flood's extent, depth,
water surface elevation and velocity, and it can generate topobathymetry by burning ARC's channel bathymetry into the DEM.

See the [documentation](docs/index.md) for how it works and every setting.

## Getting started

1. Install [Miniforge](https://conda-forge.org/download/) or [Miniconda](https://docs.anaconda.com/miniconda/) if you
   don't have conda yet.
2. Get the code:
   ```bash
   git clone https://github.com/MikeFHS/curve2flood
   cd curve2flood
   ```
3. Create the conda environment from [environment.yml](environment.yml), activate it, and install Curve2Flood into it:
   ```bash
   conda env create -f environment.yml
   conda activate curve2flood
   pip install --no-deps .
   ```
4. Run `curve2flood -h` to check that it works, and `curve2flood path/to/input_file.yaml` to map a flood. The
   [Getting Started](docs/getting_started.md) page shows a minimal input file.

## Tests

```bash
pytest
```

## License

Curve2Flood is released under the [GNU General Public License v3](LICENSE).
