# Input File

The input file tells Curve2Flood where its inputs are, how to map the flood and which outputs to write. It can be
YAML (a file ending in `.yaml` or `.yml`):

```yaml
# Inputs
DEM_File: /path/to/dem.tif
Print_VDT_Database: /path/to/vdt.parquet
COMID_Flow_File: /path/to/flows.csv

# Outputs
OutFLD: /path/to/flood.tif
OutDEP: /path/to/depth.tif

# Settings
mapper: Curve2Flood-Kernel Weighted
TW_MultFact: 1.5
TopWidthPlausibleLimit: 2000
```

or text, with a key, spaces or a tab, and its value on each line. Lines starting with `#` are comments.

```text
DEM_File	/path/to/dem.tif
mapper	Curve2Flood-Kernel Weighted
```

Keys are case-insensitive, so `DEM_File` and `dem_file` are the same. Any key Curve2Flood doesn't know, stops the run with an error that suggests the closest key: give ARC its own input file. FloodSpreader's `OutSHP` and `FS_ADJUST_FLOW_BY_FRACTION` are allowed and ignored. Booleans are `True` or `False` (`yes` and `no`, or `1` and `0`,
work too). A key with no value, or `null` in YAML, takes its default. Leave an output out to skip it.

## What a run needs

`DEM_File` is needed but for an FLDPLN flood map without `OutWSE` or the bathymetry, which takes its grid from its
library (see [FLDPLN Mapper](fldpln.md#inputs)). Mapping a flood (for any of `OutFLD`, `OutDEP`, `OutWSE` and
`OutVEL`) also needs:

- `COMID_Flow_File`, and `Print_VDT_Database` or `Print_Curve_File`;
- or, with `Set_Depth`, `Stream_File` instead.

| Setting | Also needs |
| --- | --- |
| `mapper: Curve2Flood-FLDPLNpy` | `Print_VDT_Database`, `Flow_Direction_File`, `Stream_Info_File` and `FLDPLN_Library`. |
| `OutVEL` | `LU_Raster_SameRes` and `LU_Manning_n`. Not available with `Set_Depth`. |
| `Flood_WaterLC_and_STRM_Cells` | `LU_Raster_SameRes`. |
| `Flood_Bathy_Mask` | `BathyWaterMask`, which has to exist. |

Before it reads anything, a run checks that every file it reads exists, and that the folder of every file it
writes exists and can be written to, and stops naming all that don't. It also stops if the flow file has more
than 655 flow events or a negative flow.

## Input files

Every raster must match the DEM's size and extent. Tables can be CSV or Parquet.

| Key | Description |
| --- | --- |
| `DEM_File` | The DEM, in metres, with coordinates in metres or degrees. Cells at or below -9998 are no-data and never flood. |
| `Print_VDT_Database` | The VDT database produced by ARC. Contains rating curves for velocity, top width, and depth for every stream cell (see [Key Concepts](key_concepts.md#rating-curves)). Used rather than the curve file when both are given. |
| `Print_Curve_File` | The curve file produced by ARC. Similar to the VDT, but contains each rating curve as coefficients in a power law equation. |
| `COMID_Flow_File` | The flows to map, in m³/s: the reach ID in the first column, then one column per flow event, 655 at most. An empty flow counts as zero, and a reach missing from the file isn't mapped (a warning names them). |
| `Stream_File` | A stream raster, each stream cell holding its reach ID, with nodata being 0. Read only with `Set_Depth`; otherwise the stream cells come frome the VDT's or curve file's. |
| `LU_Raster_SameRes` | The land cover raster. |
| `LU_Manning_n` | A tab-separated table of Manning's n for each land cover class, with a header row: the class in the first column and its n in the third, as ARC reads it. Land cover the table doesn't list, its no-data among it, takes n = 0.035, with a warning naming the classes. |

## Outputs

See [Outputs](outputs.md) for what each one contains.

| Key | Default | Description |
| --- | --- | --- |
| `OutFLD` | | The flood extent raster. |
| `OutDEP` | | The flood depth raster. |
| `OutWSE` | | The water surface elevation raster. |
| `OutVEL` | | The velocity raster. |
| `Make_Output_GPKG` | True | Also write the flood extent as polygons, beside `OutFLD`: its name with a `.gpkg` extension instead of its own. |
| `compression` | LZW | The output rasters' GeoTIFF compression, such as LZW, DEFLATE or ZSTD. |

## Mapping

| Key | Default | Description |
| --- | --- | --- |
| `mapper` | Curve2Flood-Kernel Weighted | How the water spreads from the stream: `Curve2Flood-Kernel Weighted` (see [Kernel-Weighted Mapper](kernel_weighted.md)), `Curve2Flood-Multi-Point Interpolation` (see [Multi-Point Interpolation Mapper](multi_point.md)) or `Curve2Flood-FLDPLNpy` (see [FLDPLN Mapper](fldpln.md)). |
| `TW_MultFact` | 3.0 | Multiplies every top width from the rating curves. Kernel-weighted mapper only: the multi-point mapper scales top widths with its own `MPI_Corridor_TopWidth_Factor`, `MPI_Anchor_TopWidth_Factor` and `MPI_MaxDist_TopWidth_Factor`. |
| `TopWidthPlausibleLimit` | 1000 | The widest top width (m) allowed. The kernel-weighted mapper caps top widths here, and a reach whose median top width is wider gets a warning. |
| `LocalFloodOption` | False | Kernel-weighted mapper: a stream cell's water reaches only the cells below its water surface that connect to it. `FloodLocalOnly` is the same key; either one turns it on. |
| `Flood_WaterLC_and_STRM_Cells` | False | Always flood the stream cells and the land cover's water cells. |
| `Flood_Bathy_Mask` | False | Always flood the cells of `BathyWaterMask` (above 0). |
| `LAND_WaterValue` | 80 | The land cover's water class: 80 in ESA WorldCover, 11 in NLCD. |
| `Set_Depth` | | Map one fixed depth instead of the flows: every stream cell of `Stream_File` floods to this depth (m) above the DEM, with a top width of `TopWidthPlausibleLimit`. Kernel-weighted and multi-point mappers only. `FloodSpreader_SpecifyDepth` is the same key; `Set_Depth` wins if both are given. Off when missing or 0 or less. |

## FLDPLN mapper

See [FLDPLN Mapper](fldpln.md) for how these are used.

| Key | Default | Description |
| --- | --- | --- |
| `Flow_Direction_File` | | D8 flow directions, with WhiteboxTools codes (see [FLDPLN Mapper](fldpln.md#inputs)). |
| `Stream_Info_File` | | The table of stream segments the library was built from, produced by [Nencarta](https://nencarta.readthedocs.io/en/latest/index.html). |
| `FLDPLN_Library` | | The floodplain library, generated by `build_fldpln_library`. |
| `FLDPLN_Min_Flood_Depth` | 0.01 | The shallowest water (m) that floods a cell. Shallower water, such as the film a stream's water leaves down a slope after just topping a rise beside it, floods nothing (see [FLDPLN Mapper](fldpln.md#mapping-a-flood)). 0 floods every cell the water reaches.

## Multi-point interpolation mapper

See [Multi-Point Interpolation Mapper](multi_point.md) for how these are used.

| Key | Default | Description |
| --- | --- | --- |
| `MPI_TopWidth_Threshold_m` | 200 | Rivers wider than this (m) also get points across the channel. |
| `MPI_XS_Point_Spacing_m` | 100 | The spacing of those points (m), more than 0. |
| `MPI_Remove_HWM_Outliers` | True | Drop the points whose depth is an outlier. |
| `MPI_HWM_Outlier_Method` | mad | The outlier test: robust z-score (`mad`) or interquartile range (`iqr`). |
| `MPI_HWM_Outlier_Threshold` | 3.5 | The test's threshold. |
| `MPI_HWM_Outlier_Group_By` | comid | Test within each reach (`comid`) or over all points (`global`). |
| `MPI_HWM_Outlier_Min_Samples` | 5 | Groups with fewer points are kept as they are. |
| `MPI_Use_TopWidth_Buffers` | True | Size the corridor and the anchor from the top width, rather than `MPI_Corridor_Buffer_m` and `MPI_Anchor_Buffer_m`. |
| `MPI_Corridor_TopWidth_Factor` | 1.0 | The corridor's radius, as a multiple of the top width. |
| `MPI_Anchor_TopWidth_Factor` | 0.1 | The anchor's radius, as a multiple of the top width. |
| `MPI_Corridor_Buffer_m` | 500 | The corridor's radius (m) when it isn't sized from the top width. |
| `MPI_Anchor_Buffer_m` | 5 | The anchor's radius (m) when it isn't sized from the top width. |
| `MPI_K` | 12 | How many of the nearest points each cell's water surface is interpolated from. |
| `MPI_Power` | 2.0 | The power of the inverse distance in the weights. |
| `MPI_Use_TopWidth_Max_Distance` | True | A point reaches as far as its reach's median top width times `MPI_MaxDist_TopWidth_Factor`, rather than `MPI_Max_Distance_m`. |
| `MPI_MaxDist_TopWidth_Factor` | 1.0 | See above. |
| `MPI_Max_Distance_m` | 1000 | How far a point reaches (m) when that isn't from the top width. |
| `MPI_Min_TopWidth_m`, `MPI_Max_TopWidth_m` | 5, 500 | Top widths are clipped to this range (m). |
| `MPI_Fallback_TopWidth_m` | 20 | The top width (m) of a point that has none. |
| `MPI_Smooth_Sigma_Pixels` | 0.25 | The Gaussian smoothing of the interpolated water surface, in cells. 0 turns it off. |
| `MPI_Apply_WSE_Sanity_Filter` | True | Drop the points whose water surface is below their ground. |
| `MPI_WSE_Sanity_Tolerance_m` | 0 | How far below its ground a point's water surface may be (m) and still be kept. |
| `MPI_Connectivity` | 4 | Flooded cells connect through 4 or 8 neighbours. |
| `MPI_Fast_Mode` | False | Use the fixed radii and distance instead of the top-width ones. |
| `MPI_One_Based_VDT_RC` | False | The VDT's rows and columns count from 1 instead of 0. |

## Bathymetry

See [Bathymetry](bathymetry.md) for how these are used. Curve2Flood writes the bathymetry only when both
`BATHY_Out_File` and `FSOutBATHY` are given.

| Key | Default | Description |
| --- | --- | --- |
| `BATHY_Out_File` | | ARC's bathymetry raster: the channel bed's elevation where ARC carved one. `AROutBATHY` is the same key, and wins when both are given. |
| `FSOutBATHY` | | Where to write the DEM with the bathymetry burned in. |
| `BathyWaterMask` | | A raster of the water (above 0). ARC puts its banks where the water ends; Curve2Flood uses it only with `Flood_Bathy_Mask`, to flood its cells. |
| `Bathy_Use_Banks` | False | Allow the channel bed to rise above the DEM. When False, the new DEM is never higher than the old one. |
