# Curve2Flood
Curve2Flood is a Python library and CLI tool that creates flood inundation maps and optional topobathymetric surfaces from ARC rating-curve and VDT inputs.


## Installation

Clone or download this repository. Once you've got a local copy, navigate to the curve2flood directory and use pip to install Curve2Flood into your Python environment with the command below.

```bash
pip install .
```

## Usage

Once installed, you can run Curve2Flood as a typical Python library or a command-line utility.

### As a Library

```python
from curve2flood import Curve2Flood_MainFunction

Curve2Flood_MainFunction("path/to/input_file.txt")
```

### Command Line

```bash
curve2flood path/to/input_file.txt
```

### Input File Format

The input file can be a YAML file or a plain text file with key-value pairs, e.g.:

```
DEM_File  path/to/dem.tif
Stream_File  path/to/strm.tif
LU_Raster_SameRes  path/to/land.tif
StrmShp_File  path/to/streams.shp
OutFLD  path/to/output_flood.tif
LAND_WaterValue  80
Q_Fraction  0.5
TopWidthPlausibleLimit  200
TW_MultFact  1.0
Set_Depth  0.1
LocalFloodOption  True
Flood_WaterLC_and_STRM_Cells  False
```

### Mapper Options

`mapper` controls the flood-spreading method.

- `Curve2Flood-Kernel Weighted`: Uses weighted WSE spreading from stream cells.
- `Curve2Flood-FLDPLNpy`: Uses a precomputed FLDPLN library, filled DEM, flow-direction raster, stream metadata, and VDT WSE values.
- `Curve2Flood-Mult-Point`: Uses the FHS-style multi-point interpolation workflow.

### FLDPLN Behavior

`Curve2Flood-FLDPLNpy` currently:

- Interpolates WSE from the VDT database for the active flow event.
- Builds stream-path WSE and depth profiles from the stream graph.
- Smooths the profiles with an odd median filter window.
- Converts the smoothed profile to depth-of-flood and queries the FLDPLN library.
- Keeps cells that remain hydraulically connected to stream cells.

FLDPLN map controls include `FLDPLN_Median_Filter_Size`, `FLDPLN_DoF_Scale`, `FLDPLN_DoF_Offset`, `FLDPLN_Missing_FSP_Interpolation`, `FLDPLN_DoF_Signal`, and `FLDPLN_Threshold_Mode`.

The Scottsbluff FLDPLN integration test and tuning notes are in `docs/scottsbluff_fldpln_csi.md` and `docs/scottsbluff_fldpln_research.md`. Multi-site FLDPLN research is in `docs/fldpln_multi_site_research.md`.
