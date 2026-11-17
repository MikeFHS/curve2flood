# Key Concepts

## Curve2Flood, step by step

1. **Read the settings** and check that everything the run needs is there (see
   [What a run needs](input_file.md#what-a-run-needs)).
2. **Read the DEM**, and work out its cell size in metres. A DEM in degrees is measured with geodesic distances at its
   middle latitude.
3. **Find the stream cells**: the rows and columns of the VDT database or curve file. With `Set_Depth`, they are the
   cells of `Stream_File` above 0 instead.
4. **Map each flow event** (each column of the flow file):
    1. Look up each reach's flow, and find each stream cell's water surface elevation and top width at that flow from
       its rating curve (see [Rating curves](#rating-curves)).
    2. Spread the water from the stream cells with the mapper: [kernel weighted](kernel_weighted.md),
       [multi-point interpolation](multi_point.md) or [FLDPLN](fldpln.md).
    3. Keep only the flooded patches that touch a stream cell, so water can't appear in a valley the stream never
       reaches.
5. **Combine the events.** The extent becomes the percentage of the events that flooded each cell, and the depth and
   slope the largest over the events.
6. **Optionally flood the water**: with `Flood_WaterLC_and_STRM_Cells`, the stream cells and the land cover's water
   cells are always flooded, and with `Flood_Bathy_Mask`, the cells of `BathyWaterMask`. The patches that don't touch
   a stream are removed again.
7. **Write the outputs** (see [Outputs](outputs.md)), then the bathymetry if asked (see [Bathymetry](bathymetry.md)).

## Rasters

- Every raster must be on the DEM's grid: the same rows and columns. The VDT database's and curve file's `Row` and
  `Col` are rows and columns of that grid, counted from 0; a cell outside it stops the run.
- Reach IDs are kept as 32-bit integers when every one fits, which halves the rasters of them, and as 64-bit ones
  otherwise.
- The DEM's coordinates must be in metres or degrees, and its elevations in metres. Other units stop the run.
- DEM cells at or below -9998 are no-data. They never flood.
- The mappers work on a copy of the DEM with a one-cell border of no-data around it, so that a stream cell's
  neighbours always exist.

## Rating curves

A rating curve says how a stream cell's water surface elevation (WSE), top width (the width of the water surface
across the channel) and velocity change with the flow. ARC writes them in two forms.

**The VDT database** (`Print_VDT_Database`) lists, for each stream cell (`COMID`, `Row`, `Col`), a series of flows
`q_1, q_2, ...` and the velocity `v_*`, top width `t_*` and WSE `wse_*` at each. It also has the cell's baseflow
`QBaseflow`, slope `Slope`, channel bed elevation `Elev` and cross-section direction `XS_Angle`. For a flow $Q$, a
cell's values are:

- interpolated linearly between the two flows around $Q$, or extrapolated from the first two below the smallest flow.
  The WSE is kept at or above `Elev`, and the depth, the WSE less `Elev`, at least 1 mm;
- the largest flow's values, when $Q$ is above every flow in the table.

The top width is then raised to at least the top width at baseflow, and multiplied by `TW_MultFact` (for the
kernel-weighted mapper; the multi-point mapper reads the rating curves at its own points, see
[Multi-Point Interpolation Mapper](multi_point.md)). Within each
reach, the cells below the 1st or above the 99th percentile of top width or velocity are left
out as outliers.

**The curve file** (`Print_Curve_File`) has, for each stream cell, power laws fitted to its rating curve:

$$
\text{depth} = a_d \, Q^{b_d}, \quad
\text{top width} = a_t \, Q^{b_t}, \quad
\text{velocity} = a_v \, Q^{b_v}
$$

in the columns `depth_a`, `depth_b`, `tw_a`, `tw_b`, `vel_a` and `vel_b`. The WSE is the depth plus the cell's
`BaseElev`. Cells with no depth or top width are left out, and so are the cells more than two standard deviations from
their reach's mean top width, WSE or velocity. Top widths are multiplied by `TW_MultFact`.

When both are given, the VDT database is used. The FLDPLN mapper needs it.

## Flow events

`COMID_Flow_File` has the reach ID in its first column and one column of flows (m³/s) per event. Each event is mapped
on its own, and the maps are combined (step 5). An empty flow counts as zero, and a reach missing from the file isn't
mapped. A run maps at most 655 events, and a negative flow stops it.

With `Set_Depth`, there are no flows: every stream cell floods to that depth above the DEM, once.
