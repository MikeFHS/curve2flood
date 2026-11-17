# Outputs

Curve2Flood writes only the outputs the input file names. Every raster is a GeoTIFF on the DEM's grid, compressed as
`compression` says.

The flow file can hold several flow events, one per column: a forecast's ensemble, say, or a series of return
periods. Curve2Flood maps each event, then combines them.

## Flood extent

`OutFLD`: 8-bit whole numbers. 0 is dry. With one flow event a flooded cell is 100; with several, a cell holds the percentage of the events that flooded it, rounded down. A run maps at most 655 events.

Each event's map, and the combined map, keeps only the flooded patches that touch a stream cell. A patch is a group of
flooded cells joined through any of their 8 neighbours. With `Flood_WaterLC_and_STRM_Cells`, the stream cells and the
land cover's water cells are flooded in the combined map, and with `Flood_Bathy_Mask`, the cells of
`BathyWaterMask`.

With `Make_Output_GPKG`, the extent is also written as polygons, to a GeoPackage beside `OutFLD`, its name with a
`.gpkg` extension instead of its own (`flood.tif` gives `flood.gpkg`). Each polygon's `Value` is the extent raster's value.

## Depth

`OutDEP`: 32-bit floats, in metres. A cell's depth is the deepest of the events, measured from the DEM. A flooded cell
with no depth of its own, such as a stream or water cell flooded by `Flood_WaterLC_and_STRM_Cells` or
`Flood_Bathy_Mask`, takes the depth of the nearest flooded cell that has one, or 0.01 m if there is none. Dry cells
are -9999, the raster's no-data value.

## Water surface elevation

`OutWSE`: 32-bit floats, in metres: the DEM plus the depth. Dry cells are NaN, the raster's no-data value. 

## Velocity

`OutVEL`: 32-bit floats, in m/s, from Manning's equation for a wide channel, with each cell's own depth $D$:

$$
V = \frac{1}{n} \, D^{2/3} \, S^{1/2}
$$

where $n$ is the cell's Manning's n from its land cover (`LU_Manning_n`; 0.035 for land cover the table doesn't
list, no-data among it), and $S$ the stream's slope from the rating
curves. Each mapper spreads the stream cells' slopes over the flood as it spreads their water: the kernel-weighted
mapper with the same weights, the multi-point mapper by interpolation, and the FLDPLN mapper by giving each flooded
cell its nearest stream cell's slope, averaged over the 3 x 3 cells around it. A cell takes the steepest slope of the
events. A flooded cell with no velocity takes the nearest flooded cell's, or 0.01 m/s. Dry cells are NaN.

Over the floodplain of a meandering river the water doesn't follow the channel: it runs straight down the valley, at
the valley's slope, which is the stream's times its sinuosity. With the stream's slope there, the floodplain's
velocity comes out about $1/\sqrt{\text{sinuosity}}$ of what it should be. So when the VDT database has the valley's slope (`Valley_Slope`, which ARC writes with
`Valley_Sections`), the kernel-weighted and FLDPLN mappers spread it outisde the channels. The channels are the cells within half
the baseflow top width of a stream cell, and keep the stream's slope: a stream cell's baseflow top width is the VDT's
at the first flow at or above `QBaseflow`, and the median of its reach's cells' is taken, so that a cross section
whose rating curve tops its banks below that flow doesn't make a channel of its floodplain.

## Bathymetry

`FSOutBATHY`: 32-bit floats, in metres: the DEM with ARC's bathymetry burned in. See [Bathymetry](bathymetry.md).
