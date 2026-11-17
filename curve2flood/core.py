"""Curve2Flood driver.

Reads the settings, maps the flood extent of each flow event with one of the
three mappers, and writes the requested rasters: flood extent, depth, water
surface elevation (WSE), velocity and an optional topobathymetric DEM.
"""
# built-in imports
import os

# third-party imports
import numpy as np
import polars as pl
import polars.selectors as cs
from numba import njit, prange
from osgeo import gdal
from pyproj import CRS, Geod

from curve2flood import LOG
from curve2flood._ids import as_ids, id_dtype
from curve2flood._raster import (
    Grid,
    Read_Raster_GDAL,
    _check_shape,
    open_raster,
    read_array,
    read_into,
)
from curve2flood.bathymetry import create_bathymetry
from curve2flood.config import FLDPLN, KERNEL_WEIGHTED, MULTI_POINT, Config, is_parquet
from curve2flood.outputs import (
    accumulate_positive_max,
    main_flood_ouputs,
    remove_cells_not_connected,
)
from curve2flood.rating_curves import (
    Calculate_Depth_TopWidth_TWMax_Velocity,
    calculate_interpolated_vdt,
)
from curve2flood.spreaders.fldpln import (
    library_metadata,
    library_sources,
    make_fldpln_flood_map,
)
from curve2flood.spreaders.kernel_weighted import spread_stream_cells
from curve2flood.streamcell_values import StreamCellValues

gdal.UseExceptions()

# The most flow events one run maps
MAX_FLOW_EVENTS = 655

def convert_cell_size(
    d_dem_cell_size_x: float,
    d_dem_cell_size_y: float,
    d_dem_lower_left: float,
    d_dem_upper_right: float,
    s_dem_projection: str
):
    """Return the DEM cell size in metres as ``(dx, dy)``.

    Geographic rasters (degrees) are converted with geodesic distances at the
    raster's middle latitude.  Projected rasters are already in metres and are
    returned unchanged.
    """
    d_x_cell_size = np.fabs(d_dem_cell_size_x)
    d_y_cell_size = np.fabs(d_dem_cell_size_y)

    try:
        o_crs = CRS.from_user_input(s_dem_projection)
    except Exception as e:
        raise ValueError("Unable to parse DEM projection for cell-size conversion.") from e

    if o_crs.is_geographic:
        d_lat = (d_dem_lower_left + d_dem_upper_right) / 2.0
        d_lon = 0.0  # Geodesic spacing does not depend on longitude

        # Build a geodesic calculator from the DEM ellipsoid.
        o_ellps = o_crs.ellipsoid
        if o_ellps is not None and o_ellps.semi_major_metre and o_ellps.inverse_flattening:
            o_geod = Geod(a=o_ellps.semi_major_metre, rf=o_ellps.inverse_flattening)
        else:
            o_geod = Geod(ellps="WGS84")

        _, _, d_y_metres = o_geod.inv(d_lon, d_lat, d_lon, d_lat + d_y_cell_size)
        _, _, d_x_metres = o_geod.inv(d_lon, d_lat, d_lon + d_x_cell_size, d_lat)
        d_x_cell_size = np.fabs(d_x_metres)
        d_y_cell_size = np.fabs(d_y_metres)

    return d_x_cell_size, d_y_cell_size

def create_weightbox(tw: int, dx: float, dy: float) -> np.ndarray:
    """Inverse squared distance weights for a ``(2*tw + 1)`` square window, float32.

    ``tw`` is the window's half width in cells; the centre weight is capped at
    ``1 / 0.0001``.  Worked out with ``tw`` as int32 and ``dx`` and ``dy`` as
    float32.
    """
    return _create_weightbox(np.int32(tw), np.float32(dx), np.float32(dy))

@njit(cache=True)
def _create_weightbox(tw, dx, dy):
    """``create_weightbox``, compiled when first used rather than on import."""
    n = 2*tw + 1
    indices = np.arange(n)
    # Squared distance (m^2) of each column and row from the centre cell
    X = ((indices - tw) * dx) ** 2
    Y = ((indices - tw) * dy) ** 2

    WeightBox = X[None, :] + Y[:, None]
    # Avoid very small values (to prevent division by zero)
    WeightBox = np.clip(WeightBox, 0.0001, None)
    WeightBox = 1.0 / WeightBox

    return (WeightBox).astype(np.float32)

def _at_stream_cells(raster, RR, CC):
    """A raster's values at the stream cells ``(RR, CC)`` of the padded grid; None for None."""
    if raster is None or isinstance(raster, StreamCellValues):
        return None if raster is None else raster.values
    return raster[RR - 1, CC - 1]

def CreateSimpleKernelFloodMap(config: Config, RR, CC, T_Rast, W_Rast, S_Rast, E,
                         nrows, ncols, TW_m, dx, dy,
                         WeightBox, TW_for_WeightBox_ElipseMask, flood_vdt_cells, V_Rast=None, channel=None):
    """Kernel-weighted mapper: ``(flood, depth, slope)`` for one flow event.

    Each stream cell spreads its WSE to the cells up to one top width away,
    weighted by inverse squared distance, and a cell floods where the weighted
    WSE is above the DEM.  ``depth`` is None unless a depth, WSE or velocity
    raster is written, and ``slope`` is None unless a velocity raster is.
    The slope averages the stream cells' slopes ``S_Rast`` with the same
    weights, or, given the valley slopes ``V_Rast``, theirs outside the
    ``channel`` (see ``channel_cells``).
    """
    WSE_array, Total_Weight, Slope_Times_Weight = spread_stream_cells(
        RR,
        CC,
        _at_stream_cells(T_Rast, RR, CC),
        _at_stream_cells(W_Rast, RR, CC),
        _at_stream_cells(S_Rast, RR, CC),
        E,
        nrows,
        ncols,
        TW_m,
        dx,
        dy,
        config.localfloodoption,
        WeightBox,
        TW_for_WeightBox_ElipseMask,
        config.topwidthplausiblelimit,
        config.set_depth,
        _at_stream_cells(V_Rast, RR, CC),
        channel,
    )

    with_depth = bool(config.outdep or config.outwse or S_Rast is not None)
    with_slope = S_Rast is not None
    Flooded_array = _kernel_maps(WSE_array, E, Total_Weight, Slope_Times_Weight, with_depth, with_slope)
    del Total_Weight

    # Also make sure all the Cells that have Stream are counted as flooded.
    if flood_vdt_cells:
        Flooded_array[RR - 1, CC - 1] = 1

    # The depth and slope are written over the sums they came from
    return (Flooded_array, WSE_array[1:-1, 1:-1] if with_depth else None,
            Slope_Times_Weight[1:-1, 1:-1] if with_slope else None)

@njit(cache=True, parallel=True, error_model='numpy')
def _kernel_maps(WSE_array, E, Total_Weight, Slope_Times_Weight, with_depth, with_slope):
    """The kernel-weighted mapper's flood, unpadded, from its padded sums (see ``create_kernel_weighted_spread_map``):
    a cell floods where the weighted WSE is above a valid DEM.  With ``with_depth``, ``WSE_array`` gets the depth
    there, and with ``with_slope`` ``Slope_Times_Weight`` the weighted slope (at least 0.0002), each NaN elsewhere.
    Cells no stream cell reached have no weight, hence 0 / 0."""
    nrows, ncols = E.shape[0] - 2, E.shape[1] - 2
    flooded = np.empty((nrows, ncols), dtype=np.uint8)
    nan = np.float32(np.nan)
    for r in prange(nrows):
        for c in range(ncols):
            water, ground = WSE_array[r + 1, c + 1], E[r + 1, c + 1]
            wet = water > ground > -9998.0
            flooded[r, c] = 1 if wet else 0
            if with_depth:
                WSE_array[r + 1, c + 1] = water - ground if wet else nan
            if with_slope:
                value = Slope_Times_Weight[r + 1, c + 1] / Total_Weight[r + 1, c + 1] if wet else nan
                Slope_Times_Weight[r + 1, c + 1] = np.float32(0.0002) if value <= 0 else value

    return flooded

def _make_fldpln_flood_map(
        config: Config,
        COMID_Unique_Flow,
        S_Rast,
        fill_below_dem,
        stream_info,
        flow_dir,
        fldpln_library,
        V_Rast=None,
        sources=None,
        ):
    """FLDPLN mapper: ``(flood, depth, None)`` for one flow event.

    The VDT gives each stream cell's WSE for the event and the floodplain
    library spreads it (see ``_make_fldpln_flood_map``); neither the DEM nor
    the filled DEM is needed.  ``depth`` is None unless a depth, WSE or
    velocity raster is written.  When ``S_Rast`` is given, the stream cells'
    slopes are written into it, and their valley slopes into ``V_Rast``; when
    ``fill_below_dem`` is, how far the filled DEM sits below the DEM at each
    library cell (for ``water_surface``), unless ``sources``, the library's
    stream pixels and fills (``library_sources``), are given: it was written
    as they were read.
    """
    vdt_df = calculate_interpolated_vdt(
        config.print_vdt_database,
        COMID_Unique_Flow,
        config.tw_multfact
    )
    if S_Rast is not None:
        S_Rast[vdt_df['Row'].to_numpy(), vdt_df['Col'].to_numpy()] = vdt_df['Slope'].to_numpy()
    if V_Rast is not None:
        V_Rast[vdt_df['Row'].to_numpy(), vdt_df['Col'].to_numpy()] = vdt_df['Valley_Slope'].to_numpy()

    depth = make_fldpln_flood_map(
        vdt_df,
        flow_dir,
        fldpln_library,
        stream_info,
        fill_below_dem=fill_below_dem,
        min_depth=config.fldpln_min_flood_depth,
        sources=sources,
    )

    # 1 where the depth is above 0, else 0: the comparison's bools are those bytes
    Flood_array = np.greater(depth, 0).view(np.uint8)
    # Depths are only used to write depth, WSE or velocity rasters
    Depth_array = depth if config.outdep or config.outwse or config.outvel else None

    return Flood_array, Depth_array, None  # Slopes are spread after all events are mapped

def Curve2Flood(config: Config, E, B, RR, CC, nrows, ncols, dx, dy, COMID_Unique,
                COMID_Unique_Flow, WeightBox,
                TW_for_WeightBox_ElipseMask,
                quiet, flood_vdt_cells, T_Rast, W_Rast, S_Rast,
                flowdir,
                fill_below_dem,
                stream_info,
                fldpln_library,
                V_Rast=None,
                channel=None,
                sources=None,):
    """Map one flow event with the configured mapper.

    Returns ``(flood, depth, slope)`` on the DEM grid; ``depth`` and ``slope``
    are None when no output needs them.  ``E`` and ``B`` (DEM and stream ids)
    are padded by one cell, and ``RR``/``CC`` index their stream cells.  The
    FLDPLN mapper reads neither.  ``V_Rast`` gets the stream cells' valley
    slopes, and the kernel-weighted mapper's slope outside the ``channel``
    is theirs (see ``channel_cells``).  ``sources`` are the FLDPLN library's
    stream pixels and fills (``library_sources``).
    """
    if config.mapper == FLDPLN:
        return _make_fldpln_flood_map(
            config,
            COMID_Unique_Flow,
            S_Rast,
            fill_below_dem,
            stream_info,
            flowdir,
            fldpln_library,
            V_Rast,
            sources,
        )

    if config.mapper == MULTI_POINT:
        # Imported here: it alone needs pandas and SciPy
        from curve2flood.spreaders.multi_point import multi_point_interpolation
        # It reads its rating curves itself, at its points
        Flood_array, Depth_array, Slope_array, stats_message = multi_point_interpolation(
            E=E,
            B=B,
            RR=RR,
            CC=CC,
            COMID_Unique_Flow=COMID_Unique_Flow,
            CurveParamFileName=config.print_curve_file,
            VDTDatabaseFileName=config.print_vdt_database,
            dx=dx,
            dy=dy,
            TopWidthPlausibleLimit=config.topwidthplausiblelimit,
            Set_Depth=config.set_depth,
            flood_vdt_cells=flood_vdt_cells,
            mapper_options=config.multi_point_options,
            with_depth=bool(config.outdep or config.outwse or config.outvel),
            with_slope=bool(config.outvel),
        )
        LOG.info(stats_message)
        return Flood_array, Depth_array, Slope_array

    (_, _, TopWidthMax, T_Rast, W_Rast, S_Rast) = Calculate_Depth_TopWidth_TWMax_Velocity(
         config, COMID_Unique_Flow, COMID_Unique,
         T_Rast, W_Rast, S_Rast, quiet, V_Rast)

    LOG.info('Creating Rough Flood Map Data...')

    return CreateSimpleKernelFloodMap(config, RR, CC, T_Rast, W_Rast, S_Rast,
                                      E, nrows, ncols,
                                      TopWidthMax, dx, dy,
                                      WeightBox,
                                      TW_for_WeightBox_ElipseMask,
                                      flood_vdt_cells,
                                      V_Rast,
                                      channel)

def _check_dem_units(dem_projection: str):
    """Raise a ValueError unless the DEM's coordinate system is in metres or degrees."""
    try:
        raster_crs = CRS.from_wkt(dem_projection)
    except Exception as ex:
        raise ValueError(f'Unable to parse the DEM coordinate system: {ex}') from ex

    axis_units = {(axis.unit_name or '').strip().lower() for axis in raster_crs.axis_info if axis is not None}
    axis_units.discard('')
    if not axis_units:
        raise ValueError('Unable to determine the units of the DEM coordinate system.')

    invalid_units = sorted(axis_units - {'meter', 'meters', 'metre', 'metres', 'degree', 'degrees'})
    if invalid_units:
        raise ValueError(f'The DEM coordinate system must be in metres or degrees, not {", ".join(invalid_units)}.')

def Set_Stream_Locations(nrows: int, ncols: int, infilename: str, one_based: bool = False):
    """Stream raster built from the ``Row``, ``Col`` and ``COMID`` columns of a VDT or curve file.

    Cells without a stream are -9999.  The raster is int32 when every COMID
    fits, and int64 otherwise (see ``id_dtype``).  With ``one_based`` the
    rows and columns count from 1.  A cell outside the ``nrows`` by
    ``ncols`` grid raises a ValueError: the table is another DEM's.
    """
    rows, cols, ids = _stream_table(nrows, ncols, infilename, one_based)
    S = np.full((nrows, ncols), -9999, dtype=id_dtype(ids))
    S[rows, cols] = ids

    return S

def _stream_table(nrows: int, ncols: int, infilename: str, one_based: bool = False):
    """The ``Row``, ``Col`` (int64, from 0) and ``COMID`` columns of a VDT or curve file (``Set_Stream_Locations``)."""
    df = _read_table(infilename, ['Row', 'Col', 'COMID'])
    rows = df['Row'].to_numpy().astype(np.int64) - int(one_based)
    cols = df['Col'].to_numpy().astype(np.int64) - int(one_based)
    check_cells_inside(rows, cols, nrows, ncols, infilename)
    return rows, cols, df['COMID'].to_numpy()

def _stream_cells(nrows: int, ncols: int, infilename: str, one_based: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The cells of ``Set_Stream_Locations``' raster with an id above 0, without the raster: their flat indices,
    ``row * ncols + col`` (int64), ascending, and their ids, int32 when every id of the table fits (see
    ``id_dtype``).  A cell listed twice keeps its last row's id, as the raster does."""
    rows, cols, ids = _stream_table(nrows, ncols, infilename, one_based)
    ids = ids.astype(id_dtype(ids), copy=False)
    cells = rows * ncols + cols
    order = np.argsort(cells, kind='stable')
    cells, ids = cells[order], ids[order]
    last = np.ones(cells.size, dtype=np.bool_)
    last[:-1] = cells[1:] != cells[:-1]
    last &= ids > 0
    return cells[last], ids[last]

def _read_table(path: str, columns: list[str]) -> pl.DataFrame:
    """Some columns of a CSV or Parquet table."""
    if is_parquet(path):
        return pl.read_parquet(path, columns=columns)
    
    return pl.read_csv(path, columns=columns)

def check_cells_inside(rows: np.ndarray, cols: np.ndarray, nrows: int, ncols: int, table: str):
    """Raise a ValueError if any of a table's cells ``(rows, cols)`` lies outside the ``nrows`` by ``ncols`` grid."""
    outside = (rows < 0) | (rows >= nrows) | (cols < 0) | (cols >= ncols)
    if outside.any():
        first = int(np.flatnonzero(outside)[0])
        raise ValueError(f"{int(outside.sum())} of the {rows.size} cells of {table} lie outside the DEM's {nrows} by "
                         f"{ncols} cells, the first at row {rows[first]}, column {cols[first]}: is it another DEM's, "
                         f"or do its rows and columns count from 1?")

@njit(cache=True, parallel=True)
def _positive_cells(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``np.nonzero(values > 0)``, row by row in parallel."""
    nrows, ncols = values.shape
    counts = np.zeros(nrows + 1, dtype=np.int64)
    for r in prange(nrows):
        n = 0
        for c in range(ncols):
            if values[r, c] > 0:
                n += 1
        counts[r + 1] = n

    starts = np.cumsum(counts)
    rows = np.empty(starts[-1], dtype=np.int64)
    cols = np.empty(starts[-1], dtype=np.int64)
    for r in prange(nrows):
        k = starts[r]
        for c in range(ncols):
            if values[r, c] > 0:
                rows[k] = r
                cols[k] = c
                k += 1

    return rows, cols

def check_flow_file(FlowFileName: str) -> int:
    """The number of flow events in a flow file, after checking it has at least one and at most
    ``MAX_FLOW_EVENTS``, and no negative flow.  Raises a ValueError otherwise."""
    table = _scan_flows(FlowFileName)
    events = table.collect_schema().names()[1:]
    if not events:
        raise ValueError(f"The flow file {FlowFileName} has no flow events: a column of flows for each after its "
                         f"COMID column.")
    
    if len(events) > MAX_FLOW_EVENTS:
        raise ValueError(f"The flow file {FlowFileName} has {len(events)} flow events; a run maps at most "
                         f"{MAX_FLOW_EVENTS}. Split the events between runs.")
    
    lowest = table.select(pl.col(events).cast(pl.Float64, strict=False).min()).collect().row(0)
    negative = [name for name, value in zip(events, lowest) if value is not None and value < 0]
    if negative:
        raise ValueError(f"The flow file {FlowFileName} has negative flows in event(s) {', '.join(negative)}.")
    
    return len(events)

def _warn_reaches_without_flows(FlowFileName: str, stream_ids: np.ndarray):
    """Say which reaches with stream cells have no flow in the flow file, and how many of its reaches have no
    stream cell: the ones without a flow don't flood."""
    flow_ids = _scan_flows(FlowFileName).select(pl.first()).collect().to_series().to_numpy()
    no_flow = np.setdiff1d(stream_ids, flow_ids)
    if no_flow.size:
        LOG.warning(f"{no_flow.size} of the {stream_ids.size} reaches with stream cells have no flow in "
                    f"{FlowFileName} (such as {', '.join(map(str, no_flow[:5]))}): they don't flood.")
    no_cells = np.setdiff1d(flow_ids, stream_ids)
    if no_cells.size:
        LOG.info(f"{no_cells.size} of the {np.unique(flow_ids).size} reaches of {FlowFileName} have no stream cell "
                 f"here.")

def FindFlowRateForEachCOMID_Ensemble(FlowFileName: str, flow_event_num: int) -> dict:
    """Return ``{COMID: flow}`` for one flow event.

    The flow file (CSV or Parquet) has the COMID in its first column and one
    column of flows per event.  Missing flows are treated as zero.
    """
    table = _scan_flows(FlowFileName)
    names = table.collect_schema().names()
    comid, event = names[0], names[flow_event_num + 1]
    flow_df = table.select(comid, pl.col(event).cast(pl.Float64)).collect()
    flows = flow_df[event]
    if flows.is_null().any() or flows.is_nan().any():
        LOG.warning(f"Warning: NaN values found in flow data for event {flow_event_num}. These will be treated as zero flow.")
        flows = flows.fill_nan(0.0).fill_null(0.0)

    # A COMID listed twice keeps its last flow
    return dict(zip(flow_df[comid].to_list(), flows.to_list()))

def _scan_table(path: str) -> pl.LazyFrame:
    """A CSV or Parquet table, lazily."""
    return pl.scan_parquet(path) if is_parquet(path) else pl.scan_csv(path)

def _scan_flows(path: str) -> pl.LazyFrame:
    """A flow file, lazily: a CSV's column types from all of its rows, as pandas read it, so that a flow with
    decimals after whole ones is a float."""
    return pl.scan_parquet(path) if is_parquet(path) else pl.scan_csv(path, infer_schema_length=None)

def has_valley_slopes(VDTDatabaseFileName: str | None) -> bool:
    """Whether a VDT database gives its stream cells' valley slopes (``Valley_Slope``, ARC's with Valley_Sections)."""
    return bool(VDTDatabaseFileName) and "Valley_Slope" in _scan_table(VDTDatabaseFileName).collect_schema().names()

def channel_cells(VDTDatabaseFileName: str, nrows: int, ncols: int, dx: float, dy: float) -> np.ndarray:
    """The channels: the cells within half their reach's baseflow top width of a stream cell of the VDT database, and
    the stream cells themselves.

    The water in a channel runs along it, at the stream's slope, but over the
    floodplain of a meandering river it runs straight down the valley, at the
    valley's slope: the stream's times its sinuosity.  So outside the channels
    the velocity takes the valley's slope, where the VDT gives it.

    A stream cell's baseflow top width is the VDT's at the first flow at or
    above its ``QBaseflow``, as in ``Find_TopWidth_at_Baseflow_when_using_VDT``,
    and a reach's is the median of its cells': on tight synthetic meanders, a
    few cross sections' rating curves top their banks below that flow, and
    their own top widths, up to 700 m, made channels six times the channel's
    area of the floodplain.
    """
    vdt = _scan_table(VDTDatabaseFileName).select(
        'COMID', 'Row', 'Col', 'QBaseflow', cs.starts_with('q_'), cs.starts_with('t_')).collect()
    flows = vdt.select(cs.starts_with('q_')).to_numpy().astype(np.float32, copy=False)
    top_widths = vdt.select(cs.starts_with('t_')).to_numpy().astype(np.float32, copy=False)
    baseflow = vdt['QBaseflow'].to_numpy().astype(np.float32, copy=False)
    first = np.minimum((flows < baseflow[:, None]).sum(axis=1), flows.shape[1] - 1)
    cell_width = top_widths[np.arange(len(vdt)), first].astype(np.float64)
    half_width = pl.DataFrame({'COMID': vdt['COMID'], 'width': cell_width}).select(
        pl.col('width').median().over('COMID') / 2.0).to_series().to_numpy()
    channel = np.zeros((nrows, ncols), dtype=np.bool_)
    _paint_channel(vdt['Row'].to_numpy().astype(np.int64), vdt['Col'].to_numpy().astype(np.int64), half_width,
                   float(dx), float(dy), channel)
    return channel

@njit(cache=True)
def _paint_channel(rows, cols, half_widths, dx, dy, channel):
    """Mark ``channel`` within ``half_widths[i]`` metres of each cell ``(rows[i], cols[i])``, and at the cell itself."""
    nrows, ncols = channel.shape
    for i in range(rows.size):
        r, c, h = rows[i], cols[i], half_widths[i]
        channel[r, c] = True
        if not h > 0.0:
            continue

        reach_rows, reach_cols = int(h // dy), int(h // dx)
        for a in range(max(r - reach_rows, 0), min(r + reach_rows + 1, nrows)):
            for b in range(max(c - reach_cols, 0), min(c + reach_cols + 1, ncols)):
                if ((a - r) * dy) ** 2 + ((b - c) * dx) ** 2 <= h * h:
                    channel[a, b] = True

def _vdt_cells(VDTDatabaseFileName: str, ncols: int) -> np.ndarray:
    """The flat indices, ``row * ncols + col``, of a VDT database's cells, ascending."""
    df = _read_table(VDTDatabaseFileName, ['Row', 'Col'])
    return np.unique(df['Row'].to_numpy().astype(np.int64) * ncols + df['Col'].to_numpy().astype(np.int64))

def spread_flood(
        config: Config,
        E: np.ndarray | None,
        nrows: int,
        ncols: int,
        WeightBox: np.ndarray,
        TW_for_WeightBox_ElipseMask: int,
        dx: float,
        dy: float,
        dem_projection: str,
        quiet: bool = False,
        flood_vdt_cells: bool = False,
        num_flows: int | None = None):
    """Map every flow event and write the flood, depth, WSE and velocity rasters.

    Returns the flood raster: 0 is dry, and 1-100 the percentage of flow events
    that flooded the cell, rounded down (100 with a single event).  ``E`` is
    the DEM padded by one cell; the FLDPLN mapper doesn't need it, and reads
    the DEM for ``OutWSE`` a strip at a time when it is None.  ``num_flows``
    is the flow file's number of events, counted when None.
    """
    set_depth = config.uses_set_depth
    _check_dem_units(dem_projection)

    if config.mapper == FLDPLN:
        FlowDir = read_array(config.flow_direction_file, np.uint8)
        _check_shape(FlowDir, (nrows, ncols), "Flow direction raster")

        stream_info = _read_table(config.stream_info_file, ['stream_id', 'start_pixel', 'length'])

        # A Parquet library is read by its path, a part at a time (see make_fldpln_flood_map)
        if is_parquet(config.fldpln_library):
            fldpln_library = config.fldpln_library
        else:
            fldpln_library = pl.scan_csv(config.fldpln_library)
        # How far the filled DEM sits below the DEM, from the library, for the water surface (see water_surface)
        fill_below_dem = np.full((nrows, ncols), np.nan, np.float32) if config.outwse else None
        # The library's stream pixels and fills are the same for every event: read once, writing fill_below_dem
        sources = library_sources(fldpln_library, nrows * ncols, fill_below_dem)
    else:
        FlowDir = None
        fill_below_dem = None
        stream_info = None
        fldpln_library = None
        sources = None

    LOG.info("Executing flood mapping logic...")

    # Stream cells come from the stream raster with Set_Depth, otherwise from the VDT or curve file, whose rows and
    # columns the multi-point mapper may count from 1
    one_based = config.mapper == MULTI_POINT and config.mpi_one_based_vdt_rc
    # Where the stream cells are, row by row, and their ids, int32 when they all fit (see id_dtype).  After this
    # only whether a cell is a stream cell is needed (S), but for the multi-point mapper's padded stream ids.
    if set_depth:
        S, _, _ = Read_Raster_GDAL(config.stream_file)
        _check_shape(S, (nrows, ncols), "Stream raster")
        ids = as_ids(S)
        rows, cols = _positive_cells(ids)
        stream_ids = ids[rows, cols]
        del ids
        S = S > 0
    else:
        cells, stream_ids = _stream_cells(nrows, ncols, config.print_vdt_database or config.print_curve_file,
                                          one_based)
        rows, cols = np.divmod(cells, ncols)
        S = np.zeros((nrows, ncols), dtype=np.bool_)
        S.reshape(-1)[cells] = True
        del cells

    # As indices of the rasters padded by one cell
    RR, CC = rows + 1, cols + 1
    COMID_Unique = np.unique(stream_ids).astype(np.int64)  # Always sorted
    B = None
    if config.mapper == MULTI_POINT:
        # It reads them only at the stream cells (see StreamCellValues)
        B = StreamCellValues(RR * (ncols + 2) + CC, ncols + 2, 0, dtype=stream_ids.dtype)
        B.values[:] = stream_ids

    del stream_ids, rows, cols

    if set_depth:
        # Every stream cell gets the same depth, so there is a single event
        num_flows = 1
        COMID_Unique_Flow = {}
    else:
        LOG.info('Opening and Reading ' + config.comid_flow_file)
        if num_flows is None:
            num_flows = check_flow_file(config.comid_flow_file)
        _warn_reaches_without_flows(config.comid_flow_file, COMID_Unique)

    LOG.info('Evaluating ' + str(num_flows) + ' Flow Events')

    # Create initial rasters once, outside the loop.  The kernel-weighted mapper reads them only at the stream
    # cells, and the FLDPLN mapper writes its slopes only at the VDT's cells, so they keep only those cells' values.
    # The multi-point mapper reads its rating curves at its own points.
    cells = None
    if config.mapper == KERNEL_WEIGHTED:
        cells = (RR - 1) * ncols + (CC - 1)
    elif config.mapper == FLDPLN and config.outvel:
        cells = _vdt_cells(config.print_vdt_database, ncols)

    def raster(value):
        if cells is not None:
            return StreamCellValues(cells, ncols, value)
        return np.full((nrows, ncols), value, np.float32)

    T_Rast = W_Rast = S_Rast = V_Rast = channel = None
    if config.mapper == KERNEL_WEIGHTED:
        T_Rast = raster(-1.0)
        W_Rast = raster(np.nan)

    if config.outvel and config.mapper != MULTI_POINT:
        # Stream-cell slopes, only needed for velocity
        S_Rast = raster(np.nan)
        # With the VDT's valley slopes (ARC's Valley_Sections), the velocity outside the channels takes the valley's
        # slope rather than the stream's
        if has_valley_slopes(config.print_vdt_database):
            V_Rast = raster(np.nan)
            channel = channel_cells(config.print_vdt_database, nrows, ncols, dx, dy)

    # How many events flooded each cell: uint16 holds MAX_FLOW_EVENTS
    Flood_Ensemble = np.zeros((nrows, ncols), dtype=np.uint16 if num_flows > 1 else np.uint8)
    # The largest depth and slope over the events so far (see accumulate_positive_max)
    Depth_Max = None
    Slope_Max = None
    for flow_event_num in range(num_flows):
        LOG.info('Working on Flow Event ' + str(flow_event_num))
        # clear out last events values
        if T_Rast is not None:
            T_Rast[:] = -1.0
        if W_Rast is not None:
            W_Rast[:] = np.nan

        if not set_depth:
            COMID_Unique_Flow = FindFlowRateForEachCOMID_Ensemble(config.comid_flow_file, flow_event_num)
        Flood_array_this_flow, Depth_array, Slope_array = Curve2Flood(
            config, E, B, RR, CC, nrows, ncols, dx, dy, COMID_Unique,
            COMID_Unique_Flow, WeightBox, TW_for_WeightBox_ElipseMask,
            quiet, flood_vdt_cells, T_Rast, W_Rast, S_Rast, FlowDir,
            fill_below_dem, stream_info, fldpln_library, V_Rast, channel, sources
            )
        Flood_array_this_flow = remove_cells_not_connected(Flood_array_this_flow, S)
        Flood_Ensemble += Flood_array_this_flow
        # Neither is read again
        Depth_Max = accumulate_positive_max(Depth_Max, Depth_array, in_place=True)
        Slope_Max = accumulate_positive_max(Slope_Max, Slope_array, in_place=True)
        del Flood_array_this_flow, Depth_array, Slope_array

    # clear some memory by deleting large arrays that are no longer needed
    del T_Rast
    del W_Rast
    del FlowDir

    return (_percent_of_events(Flood_Ensemble, num_flows), S, fill_below_dem, channel, S_Rast, V_Rast, Depth_Max,
            Slope_Max)

def _percent_of_events(counts: np.ndarray, num_flows: int) -> np.ndarray:
    """The percentage of the ``num_flows`` flow events that flooded each cell, from how many did (``counts``, uint8
    or uint16, worked in), rounded down, as uint8.  ``MAX_FLOW_EVENTS`` times 100 fits uint16."""
    counts *= 100
    if num_flows > 1:
        counts //= num_flows
        
    return counts.astype(np.uint8, copy=False)

def run_grid(config: Config) -> Grid:
    """The grid a run maps on: the DEM's, or for an FLDPLN flood map its library's (see ``library_metadata``) or,
    for a library that doesn't say, its flow directions'.  The rasters given must all share it."""
    if not (config.mapper == FLDPLN and config.needs_flood_map):
        return Grid.of(config.dem_file)

    flow_directions = Grid.of(config.flow_direction_file)
    metadata = library_metadata(config.fldpln_library)
    if metadata is None:
        grid = flow_directions
    else:
        grid = Grid(tuple(metadata["geotransform"]), metadata["crs"], metadata["nrows"], metadata["ncols"])
        grid.check_matches(flow_directions, "The FLDPLN library's grid", "Flow_Direction_File")
    if config.dem_file:
        grid.check_matches(Grid.of(config.dem_file), "The FLDPLN grid", "DEM_File")
    return grid

def Curve2Flood_MainFunction(input_file: str | os.PathLike | Config = None,
                             args: dict | None = None,
                             quiet: bool = False,
                             flood_vdt_cells: bool = True,
                             bathymetry_creation_options: list[str] | None = None):
    """Run Curve2Flood: map the flood events and write the requested outputs.

    Before anything is read, every input file must exist and every output's
    folder must (see ``Config.check_paths``), and the flow file must have at
    most ``MAX_FLOW_EVENTS`` events and no negative flow.

    Parameters
    ----------
    input_file : str, os.PathLike or Config, optional
        Path to a YAML or ``key value`` text input file, or the settings
        already read into a `Config`.  Used instead of ``args`` when given.
    args : dict, optional
        The same settings as a dict.  Paths may be strings or ``Path`` objects.
    quiet : bool
        Skip the per-reach warnings about top widths above
        ``TopWidthPlausibleLimit``.
    flood_vdt_cells : bool
        Always mark the stream cells as flooded (kernel-weighted and
        multi-point mappers).
    bathymetry_creation_options : list[str], optional
        GDAL creation options for the bathymetry raster.
    """
    if isinstance(input_file, Config):
        config = input_file
    elif input_file:
        config = Config.from_file(input_file)
    elif args:
        config = Config.from_mapping(args)
    else:
        raise ValueError("Provide an input file, a dict of settings or a Config.")

    # Everything the run reads and writes, before any of it
    config.check_paths()
    num_flows = None
    if config.needs_flood_map and not config.uses_set_depth:
        num_flows = check_flow_file(config.comid_flow_file)

    grid = run_grid(config)
    dem_geotransform, dem_projection = grid.geotransform, grid.projection
    nrows, ncols = grid.nrows, grid.ncols
    yll = dem_geotransform[3] - nrows * abs(dem_geotransform[5])
    yur = dem_geotransform[3]

    # Only the curve-based mappers need the whole DEM at once: the FLDPLN mapper's WSE raster and the bathymetry
    # read it a strip of rows at a time
    bathymetry = config.makes_bathymetry
    E = None
    if config.needs_flood_map and config.mapper != FLDPLN:
        LOG.info('Opening ' + config.dem_file)
        # The DEM padded by one cell of -9999, so neighbour lookups need no bounds checks
        E = np.full((nrows+2, ncols+2), -9999.0, dtype=np.float32)
        read_into(open_raster(config.dem_file), E[1:-1, 1:-1])

    # Cell size from the geotransform; the vector magnitudes also handle rotated grids
    dem_cell_size_x = np.hypot(dem_geotransform[1], dem_geotransform[2])
    dem_cell_size_y = np.hypot(dem_geotransform[4], dem_geotransform[5])
    dx, dy = convert_cell_size(dem_cell_size_x, dem_cell_size_y, yll, yur, dem_projection)
    LOG.info('Cellsize X = ' + str(dx))
    LOG.info('Cellsize Y = ' + str(dy))

    TW_for_WeightBox_ElipseMask = int( max( np.round(config.topwidthplausiblelimit/dx,0), np.round(config.topwidthplausiblelimit/dy,0) ) )  # Half width of the weight box, in cells
    # Only the kernel-weighted mapper spreads with it
    WeightBox = None
    if config.needs_flood_map and config.mapper == KERNEL_WEIGHTED:
        LOG.info('Creating the Weight Box')
        WeightBox = create_weightbox(TW_for_WeightBox_ElipseMask, dx, dy)

    if config.needs_flood_map:
        flood_ensemble, S, fill_below_dem, channel, S_Rast, V_Rast, Depth_Max, Slope_Max = spread_flood(
            config, E, nrows, ncols, WeightBox, TW_for_WeightBox_ElipseMask, dx, dy, dem_projection, quiet=quiet,
            flood_vdt_cells=flood_vdt_cells, num_flows=num_flows)
        main_flood_ouputs(flood_ensemble, fill_below_dem, S, S_Rast, V_Rast, Depth_Max, Slope_Max, channel, config, E, nrows, ncols, dem_projection, dem_geotransform)

    if bathymetry:
        create_bathymetry(config, E, nrows, ncols, dem_geotransform, dem_projection, bathymetry_creation_options)

    LOG.info("Flood mapping completed.")
