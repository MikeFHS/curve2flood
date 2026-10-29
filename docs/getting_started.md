# Getting Started

Curve2Flood typically reads its settings from an input file: a YAML file, or a text file with a key and its value on each line. [Input File](input_file.md) describes all the keys.

A minimal input file maps one or more flow events with the default mapper:

```yaml
DEM_File: /path/to/dem.tif
Print_VDT_Database: /path/to/vdt.parquet
COMID_Flow_File: /path/to/flows.csv
OutFLD: /path/to/flood.tif
```

## Run from the command line

Installing Curve2Flood adds the `curve2flood` command:

```bash
curve2flood path/to/input_file.yaml
```

Run `curve2flood -h` to see the CLI options.

## Run from Python

```python
from curve2flood import Curve2Flood_MainFunction

Curve2Flood_MainFunction("path/to/input_file.yaml")
```

You can also give the settings as a dictionary, with the same keys as the input file. Paths may be strings or
`pathlib.Path` objects:

```python
from curve2flood import Curve2Flood_MainFunction

Curve2Flood_MainFunction(args={
    "DEM_File": "path/to/dem.tif",
    "Print_VDT_Database": "path/to/vdt.parquet",
    "COMID_Flow_File": "path/to/flows.csv",
    "OutFLD": "path/to/flood.tif",
})
```

`Config` reads and checks the settings without running anything. Each setting is an attribute named for its key in lower case, and a run can be given the `Config` itself:

```python
from curve2flood import Config, Curve2Flood_MainFunction

config = Config.from_file("path/to/input_file.yaml")  # or Config.from_mapping({...})
print(config.mapper, config.tw_multfact)
Curve2Flood_MainFunction(config)
```

## Threads

Curve2Flood uses every CPU by default, in two ways:

- GDAL compresses and decompresses the rasters Curve2Flood reads and writes with `GDAL_NUM_THREADS` threads when that
  environment variable (or GDAL configuration option) is set, and with every CPU otherwise.
- The mappers' heavy loops run in parallel with numba, on `NUMBA_NUM_THREADS` threads (every CPU by default; numba
  reads it when it is first imported). The multi-point mapper's nearest-point search uses as many.

Set both lower when several Curve2Flood runs share a machine, as when nencarta maps several sites at once: for
example `GDAL_NUM_THREADS=4` and `NUMBA_NUM_THREADS=4`. The outputs are the same whatever the number of threads.

## Next

- [Input File](input_file.md): every setting and what it does.
- [Outputs](outputs.md): what each output raster contains.
