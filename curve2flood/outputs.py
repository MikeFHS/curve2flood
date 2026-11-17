import numpy as np
import pyarrow as pa
from numba import njit, prange
from osgeo import gdal, ogr, osr

from curve2flood._log import LOG
from curve2flood._patches import patches_holding
from curve2flood._raster import (
    _STRIP_ROWS,
    Read_Raster_GDAL,
    _check_shape,
    _read_rows,
    open_raster,
    write_threads,
)
from curve2flood.config import FLDPLN, Config

# Rows of _smooth_3x3's band
_SMOOTH_BAND = 64

# Cell kinds of _matched_to_flood, as bits: a cell may be both
_SOURCE, _TARGET = 1, 2

# Target rows each thread works through in turn, keeping its place in each column's sources
_NEAREST_CHUNK_ROWS = 32

# Manning's n of land cover the LU_Manning_n table doesn't list, no-data among it
DEFAULT_MANNING_N = 0.035

def main_flood_ouputs(
        Flood_Ensemble: np.ndarray,
        fill_below_dem: np.ndarray | None,
        S: np.ndarray | None,
        S_Rast: np.ndarray | None,
        V_Rast: np.ndarray | None,
        Depth_Max: np.ndarray,
        Slope_Max: np.ndarray | None,
        channel: np.ndarray | None,
        config: Config,
        E: np.ndarray | None,
        nrows: int,
        ncols: int,
        dem_projection: str,
        dem_geotransform: tuple):
    """Write the flood, depth, WSE and velocity rasters.

    ``Flood_Ensemble`` is the flood raster (uint8): 0 is dry, and 1-100 the
    percentage of flow events that flooded the cell (see ``spread_flood``);
    it is returned with the water land cover, stream and water mask cells
    added as the settings ask.  ``E`` is the DEM padded by one cell; the
    FLDPLN mapper doesn't need it, and reads the DEM for ``OutWSE`` a strip
    at a time when it is None.
    """
    OutVEL = config.outvel
    Flood_WaterLC_and_STRM_Cells = config.flood_waterlc_and_strm_cells
    OutDEP = config.outdep
    OutWSE = config.outwse
    Flood_File = config.outfld

    if Flood_WaterLC_and_STRM_Cells or OutVEL:
        LC_array, _, _ = Read_Raster_GDAL(config.lu_raster_sameres)
        _check_shape(LC_array, (nrows, ncols), "Land cover raster")

    if Flood_WaterLC_and_STRM_Cells:
        LOG.info('Flooding the Water-Related Land Cover and STRM cells')
        Flood_Ensemble = Flood_WaterLC_and_STRM_Cells_in_Flood_Map(Flood_Ensemble, S, LC_array, config.land_watervalue)

    if config.flood_bathy_mask:
        LOG.info('Flooding the bathymetry water mask cells')
        mask, _, _ = Read_Raster_GDAL(config.bathywatermask)
        _check_shape(mask, (nrows, ncols), "Bathymetry water mask")
        Flood_Ensemble = Flood_Bathy_Mask_in_Flood_Map(Flood_Ensemble, mask)
        del mask

    # Remove crop circles and other disconnected cells
    Flood_Ensemble = remove_cells_not_connected(Flood_Ensemble, S)

    if Flood_File:
        LOG.info('Creating Ensemble Flood Map...' + str(Flood_File))
        _write_strips(Flood_File, ncols, nrows, dem_geotransform, dem_projection, config.compression, None,
                      lambda r0, r1: Flood_Ensemble[r0:r1], gdal.GDT_Byte, tiled=False)


    if OutDEP or OutWSE or OutVEL:
        Depth_Array = create_depth(finish_positive_max(Depth_Max), Flood_Ensemble, in_place=True)
        del Depth_Max

        if OutDEP:
            _write_strips(OutDEP, ncols, nrows, dem_geotransform, dem_projection, config.compression, -9999.0,
                          lambda r0, r1: np.where(np.isnan(Depth_Array[r0:r1]), -9999.0,
                                                  Depth_Array[r0:r1]).astype(np.float32, copy=False))

    if OutWSE:
        if E is not None:
            dem_rows = lambda r0, r1: E[1 + r0:1 + r1, 1:-1]
        else:
            # The DEM wasn't read whole (see Curve2Flood_MainFunction): its rows as they are written, as float32
            dem_ds = open_raster(config.dem_file)  # kept, or its band would outlive it
            dem_band = dem_ds.GetRasterBand(1)
            dem_rows = lambda r0, r1: _read_rows(dem_band, r0, r1 - r0, ncols)
            
        fill_rows = fill_below_dem
        _write_strips(OutWSE, ncols, nrows, dem_geotransform, dem_projection, config.compression, np.nan,
                      lambda r0, r1: water_surface(Depth_Array[r0:r1], dem_rows(r0, r1),
                                                   None if fill_rows is None else fill_rows[r0:r1]))
    del fill_below_dem

    # The FLDPLN mapper gives no slopes: each flooded cell takes its nearest
    # stream cell's slope, averaged over its 3x3 neighbourhood, and outside
    # the channels its valley slope, if there are valley slopes
    if config.mapper == FLDPLN and S_Rast is not None:
        Slope_array = spread_stream_slopes(S_Rast.raster(nrows), Flood_Ensemble, in_place=True)
        S_Rast = None
        if V_Rast is not None:
            valley = spread_stream_slopes(V_Rast.raster(nrows), Flood_Ensemble, in_place=True)
            V_Rast = None
            np.copyto(Slope_array, valley, where=~channel)
            del valley
        Slope_Max = accumulate_positive_max(None, Slope_array, in_place=True)
        del Slope_array
    del S_Rast, V_Rast, channel

    if OutVEL:
        create_velocity(config, OutVEL, Depth_Array, LC_array, finish_positive_max(Slope_Max), dem_geotransform,
                        dem_projection, ncols, nrows, Flood_Ensemble)

    if config.gpkg_file:
        flood_gdf = Write_Output_Raster_As_GeoDataFrame(Flood_Ensemble, ncols, nrows, dem_geotransform, dem_projection, gdal.GDT_Byte)
        # The driver is inferred from the file extension
        flood_gdf.to_file(config.gpkg_file)

    return Flood_Ensemble



def Flood_WaterLC_and_STRM_Cells_in_Flood_Map(Flood_Ensemble, S, LC_array, watervalue):
    """Mark stream cells and water land-cover cells as flooded (100) in ``Flood_Ensemble``."""
    # One boolean mask: int64 copies of three rasters made this the memory peak of a whole run
    F = (S > 0) | (LC_array == watervalue)
    Flood_Ensemble[F] = 100

    return Flood_Ensemble

def Flood_Bathy_Mask_in_Flood_Map(Flood_Ensemble, mask):
    """Mark the water mask's cells (above 0) as flooded (100) in ``Flood_Ensemble``."""
    Flood_Ensemble[mask > 0] = 100

    return Flood_Ensemble


def remove_cells_not_connected(flood_array: np.ndarray, streams_array: np.ndarray) -> np.ndarray:
    """Remove flooded patches that do not touch a stream.

    A patch is a group of cells with ``flood_array`` not 0 connected through
    any of their 8 neighbours, as ``scipy.ndimage.label`` finds them.  Patches
    containing a cell with ``streams_array > 0`` are kept; the rest are set to
    0, as ``flood_array`` times a mask of the kept cells gives it.
    """
    # A bool mask keeps the flood's type
    return flood_array * patches_holding(flood_array, streams_array, True)

def accumulate_positive_max(maximum: np.ndarray | None, array: np.ndarray | None,
                            in_place: bool = False) -> np.ndarray | None:
    """Raise ``maximum`` (float32, -inf where nothing is positive yet) to ``array``'s positive values, in place.

    Starts a new maximum when ``maximum`` is None: with ``in_place``,
    ``array`` itself (float32) becomes it.  An ``array`` of None changes
    nothing.  Each flow event's raster is folded in as it is mapped, so the
    events' rasters are never all held at once.
    """
    if array is None:
        return maximum
    
    array = np.asarray(array, dtype=np.float32)
    if maximum is None:
        maximum = array if in_place else np.empty(array.shape, dtype=np.float32)
        _positive_max(maximum, array, True)
        return maximum
    
    _positive_max(maximum, array, False)
    return maximum

@njit(cache=True, parallel=True)
def _positive_max(maximum: np.ndarray, array: np.ndarray, start: bool):
    """``maximum`` raised to ``array``'s positive values, or with ``start`` set to them (-inf elsewhere); ``maximum``
    may be ``array``.  Neither has a NaN where it matters: a NaN of ``array`` isn't positive."""
    minus_inf = np.float32(-np.inf)
    for r in prange(array.shape[0]):
        for c in range(array.shape[1]):
            value = array[r, c]
            positive = value if value > 0 else minus_inf
            if start or positive > maximum[r, c]:
                maximum[r, c] = positive

def finish_positive_max(maximum: np.ndarray) -> np.ndarray:
    """``accumulate_positive_max``'s result with NaN, in place, where no value was positive."""
    maximum[maximum == -np.inf] = np.nan
    return maximum

def create_depth(max_depth: np.ndarray, Flood_Ensemble: np.ndarray, in_place: bool = False):
    """The maximum depth over the flow events (``accumulate_positive_max``), matched to the flood extent (see
    ``Flood_Flooded_Cells_in_Map``), in ``max_depth`` itself with ``in_place``."""
    return Flood_Flooded_Cells_in_Map(max_depth, Flood_Ensemble, eps=0.01, in_place=in_place)


def water_surface(depth: np.ndarray, dem: np.ndarray, fill_below_dem: np.ndarray | None = None) -> np.ndarray:
    """The water surface elevation where the depth is above 0 and the DEM has a value, NaN elsewhere.

    The depth is measured from the DEM, except with the FLDPLN mapper, which measures it from the lower of the DEM
    and the filled DEM: where bathymetry raises the DEM above the filled DEM, the water was routed across the filled
    surface and its depth taken from there (see ``_make_fldpln_flood_map``), and the water surface must be too, or it
    would stand above its neighbours by the raise.  ``fill_below_dem`` is how far the filled DEM sits below the DEM
    there (0, or negative where raised; NaN where the library doesn't say), as the mapper writes it from the
    library's fill.  Only the flooded cells are worked out.
    """
    wse = np.full(depth.shape, np.nan, dtype=np.float32)
    flooded = np.nonzero((depth > 0) & (dem > -9998.0))
    ground = dem[flooded]
    if fill_below_dem is not None:
        below = fill_below_dem[flooded]
        ground = np.where(np.isnan(below), ground, ground + below)

    wse[flooded] = depth[flooded] + ground
    return wse

def create_velocity(config: Config, OutVEL, Depth_Array, LC_array, Slope_Array,
                           geotransform, projection, ncols, nrows,
                           Flood_Ensemble):
    """Write the velocity raster from Manning's equation, ``V = D^(2/3) S^(1/2) / n``.

    Each cell is treated as a wide rectangular channel: ``D`` is its flood
    depth, ``S`` the largest positive slope over the flow events
    (``Slope_Array``, NaN where there is none: the stream's, or outside the
    channels the valley's, see ``channel_cells``) and ``n`` comes from its
    land cover.  Flooded cells without a value take their nearest neighbour's.
    The velocity is worked out in ``Slope_Array`` (float32) itself, a strip
    of rows at a time.
    """
    lookup = _manning_lookup(config.lu_manning_n)
    unlisted = set()
    VEL_Array = Slope_Array
    for r0 in range(0, nrows, _STRIP_ROWS):
        r1 = min(r0 + _STRIP_ROWS, nrows)
        strip = _manning_n(LC_array[r0:r1], lookup, unlisted)
        np.divide(1, strip, out=strip)
        strip *= Depth_Array[r0:r1] ** (2/3)
        strip *= Slope_Array[r0:r1] ** (1/2)
        VEL_Array[r0:r1] = strip

    _warn_unlisted_land_cover(unlisted, config.lu_manning_n)

    out_band_data = Flood_Flooded_Cells_in_Map(VEL_Array, Flood_Ensemble, eps=0.01, in_place=True)
    del VEL_Array

    _write_strips(OutVEL, ncols, nrows, geotransform, projection, config.compression, np.nan,
                  lambda r0, r1: out_band_data[r0:r1])



def _write_strips(path: str, ncols: int, nrows: int, geotransform: tuple, projection: str, compression: str,
                  nodata: float | None, rows, data_type: int = gdal.GDT_Float32, tiled: bool = True):
    """Write a GeoTIFF, tiled unless not ``tiled``, a strip of ``_STRIP_ROWS`` rows at a time: ``rows(r0, r1)``
    gives rows ``r0`` to ``r1``, so that no raster of the whole need be made.  Each strip is flushed from GDAL's
    block cache as it is written, which would otherwise hold it until the file is closed."""
    options = [f"COMPRESS={compression}", "PREDICTOR=2"] + (["TILED=YES"] if tiled else []) + [write_threads()]
    ds: gdal.Dataset = gdal.GetDriverByName("GTiff").Create(path, ncols, nrows, 1, data_type, options=options)
    if ds is None:
        raise RuntimeError(f"Failed to create output raster: {path}")
    
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)
    band: gdal.Band = ds.GetRasterBand(1)
    for r0 in range(0, nrows, _STRIP_ROWS):
        r1 = min(r0 + _STRIP_ROWS, nrows)
        band.WriteArray(rows(r0, r1), 0, r0)
        band.FlushCache()

    if nodata is not None:
        band.SetNoDataValue(nodata)

    band.FlushCache()
    ds.FlushCache()

def _manning_lookup(s_manning_path: str) -> np.ndarray:
    """The ``LU_Manning_n`` table as a float32 lookup array: Manning's n by land-cover code, NaN for a code that
    isn't in it or has no positive n.  The table is tab separated, with the land-cover code in the first column and
    Manning's n in the third."""
    import pandas as pd  # Imported here: only velocity needs it

    df = pd.read_csv(s_manning_path, sep='\t')
    codes = pd.to_numeric(df.iloc[:, 0], errors='coerce').to_numpy(np.float64)
    n = pd.to_numeric(df.iloc[:, 2], errors='coerce').to_numpy(np.float64)
    usable = np.isfinite(codes) & (codes >= 0) & (codes == np.round(codes)) & np.isfinite(n) & (n > 0)
    if not usable.all():
        LOG.warning(f"{int((~usable).sum())} rows of {s_manning_path} have no land cover code (a whole number, 0 or "
                    f"more) or no positive Manning's n; their land cover takes n = {DEFAULT_MANNING_N}.")
        
    codes = codes[usable].astype(np.int64)
    lookup_array = np.full(codes.max() + 1 if codes.size else 0, np.nan, dtype=np.float32)
    lookup_array[codes] = n[usable]
    return lookup_array

def _manning_n(land_cover: np.ndarray, lookup: np.ndarray, unlisted: set) -> np.ndarray:
    """Each cell's Manning's n (float32) from its land cover code and ``lookup`` (``_manning_lookup``), or
    ``DEFAULT_MANNING_N``; the codes the table doesn't give are added to ``unlisted``."""
    land_cover = np.asarray(land_cover)
    n = np.full(land_cover.shape, DEFAULT_MANNING_N, dtype=np.float32)
    if land_cover.dtype.kind == 'f':
        inside = np.isfinite(land_cover) & (land_cover >= 0) & (land_cover < lookup.size)
        codes = land_cover[inside].astype(np.int64)
    else:
        inside = (land_cover >= 0) & (land_cover < lookup.size)
        codes = land_cover[inside]

    values = lookup[codes]
    listed = np.isfinite(values)
    n[inside] = np.where(listed, values, np.float32(DEFAULT_MANNING_N))
    if not listed.all():
        unlisted.update(np.unique(codes[~listed]).tolist())

    if not inside.all():
        outside = land_cover[~inside]
        if outside.dtype.kind == 'f' and np.isnan(outside).any():
            unlisted.add('NaN')
            outside = outside[~np.isnan(outside)]

        unlisted.update(np.unique(outside).tolist())

    return n

def _warn_unlisted_land_cover(unlisted: set, s_manning_path: str):
    """Warn that the land cover codes ``unlisted`` take ``DEFAULT_MANNING_N``, not being in the table."""
    if unlisted:
        codes = sorted(unlisted, key=lambda code: (isinstance(code, str), 0 if isinstance(code, str) else code))
        shown = ', '.join(f'{code:g}' if isinstance(code, float) else str(code) for code in codes[:10])
        LOG.warning(f"{len(codes)} land cover codes aren't in {s_manning_path} ({shown}"
                    f"{', ...' if len(codes) > 10 else ''}): their cells take Manning's n = {DEFAULT_MANNING_N}.")



def spread_stream_slopes(slopes: np.ndarray, Flood_Ensemble: np.ndarray, in_place: bool = False) -> np.ndarray:
    """The FLDPLN mapper's slopes for velocity: each flooded cell takes its nearest stream cell's (``slopes``, NaN off
    the stream; see ``Flood_Flooded_Cells_in_Map``), averaged over the flooded cells of its 3x3 neighbourhood, which a
    dry cell beside the flood gets too.  NaN elsewhere.  With ``in_place``, in ``slopes`` itself (float32).
    """
    Slope_array = Flood_Flooded_Cells_in_Map(slopes, np.asarray(Flood_Ensemble, dtype=np.uint8), eps=0.0002,
                                             in_place=in_place)
    uniform_smoothing(_positive_or_nan(Slope_array), Slope_array)
    return Slope_array

@njit(cache=True, parallel=True)
def _positive_or_nan(values: np.ndarray) -> np.ndarray:
    """Where ``values`` is above 0, and NaN, in place, everywhere else."""
    positive = np.empty(values.shape, dtype=np.bool_)
    for r in prange(values.shape[0]):
        for c in range(values.shape[1]):
            positive[r, c] = values[r, c] > 0
            if not positive[r, c]:
                values[r, c] = np.nan

    return positive

@njit(cache=True, inline="always")
def _masked(mask, values, kept_w, kept_m, r0, row, c):
    """The masked value (0 off the mask) and the mask (1 or 0) at ``(row, c)``, as floats, as they were before the
    band from ``r0`` was smoothed: rows above it from ``kept_w``/``kept_m``, which hold its two rows above."""
    if row < r0:
        return kept_w[row - r0 + 2, c], kept_m[row - r0 + 2, c]
    
    if mask[row, c]:
        return np.float64(values[row, c]), 1.0
    
    return 0.0, 0.0

@njit(cache=True, parallel=True)
def _smooth_3x3(mask: np.ndarray, values: np.ndarray, three):
    """``uniform_smoothing``: SciPy's ``uniform_filter1d`` down the columns and then along the rows, a band of rows
    at a time, in place.

    The filter keeps a running sum along each line, in float64, adding the
    value entering the window less the one leaving it, and writes its mean
    in the array's own type; the same arithmetic in the same order gives the
    same values.  ``three`` is 3 in ``values``' type.
    """
    nrows, ncols = values.shape
    band = _SMOOTH_BAND
    # The running sums down each column, of the masked values and of the mask
    run_w = np.zeros(ncols)
    run_m = np.zeros(ncols)
    # The masked values and mask of the two rows above the band, as they were before the band above was smoothed
    kept_w = np.zeros((2, ncols))
    kept_m = np.zeros((2, ncols))
    down_w = np.empty((band, ncols), dtype=values.dtype)
    down_m = np.empty((band, ncols), dtype=np.float32)
    block = 256
    for r0 in range(0, nrows, band):
        r1 = min(r0 + band, nrows)
        # Down the columns, the ends repeated
        for b in prange((ncols + block - 1) // block):
            for c in range(b * block, min(b * block + block, ncols)):
                tw = run_w[c]
                tm = run_m[c]
                for r in range(r0, r1):
                    if r == 0:
                        w0, m0 = _masked(mask, values, kept_w, kept_m, r0, 0, c)
                        w1, m1 = _masked(mask, values, kept_w, kept_m, r0, min(1, nrows - 1), c)
                        tw = 0.0 + w0
                        tw += w0
                        tw += w1
                        tm = 0.0 + m0
                        tm += m0
                        tm += m1
                    else:
                        wa, ma = _masked(mask, values, kept_w, kept_m, r0, min(r + 1, nrows - 1), c)
                        wb, mb = _masked(mask, values, kept_w, kept_m, r0, max(r - 2, 0), c)
                        tw += wa - wb
                        tm += ma - mb
                    down_w[r - r0, c] = tw / 3.0
                    down_m[r - r0, c] = tm / 3.0
                run_w[c] = tw
                run_m[c] = tm

            # The band's last two rows, before they are smoothed, for the band below
            for c in range(b * block, min(b * block + block, ncols)):
                w0, m0 = _masked(mask, values, kept_w, kept_m, r0, max(r1 - 2, 0), c)
                w1, m1 = _masked(mask, values, kept_w, kept_m, r0, r1 - 1, c)
                kept_w[0, c], kept_w[1, c] = w0, w1
                kept_m[0, c], kept_m[1, c] = m0, m1

        # Along the rows, and the quotient of the two
        for r in prange(r0, r1):
            line_w = down_w[r - r0]
            line_m = down_m[r - r0]
            mean_w = np.empty(1, dtype=values.dtype)
            mean_m = np.empty(1, dtype=np.float32)
            tw = 0.0
            tm = 0.0
            for c in range(ncols):
                # In float64, as the filter's line buffers are (numba's float() of a float32 is a float32)
                if c == 0:
                    tw = 0.0 + np.float64(line_w[0])
                    tw += np.float64(line_w[0])
                    tw += np.float64(line_w[min(1, ncols - 1)])
                    tm = 0.0 + np.float64(line_m[0])
                    tm += np.float64(line_m[0])
                    tm += np.float64(line_m[min(1, ncols - 1)])
                else:
                    tw += np.float64(line_w[min(c + 1, ncols - 1)]) - np.float64(line_w[max(c - 2, 0)])
                    tm += np.float64(line_m[min(c + 1, ncols - 1)]) - np.float64(line_m[max(c - 2, 0)])
                mean_w[0] = tw / 3.0
                mean_m[0] = tm / 3.0
                count = mean_m[0] * np.float32(3)
                if count > 0:
                    values[r, c] = (mean_w[0] * three) / count

def uniform_smoothing(mask: np.ndarray, values: np.ndarray):
    """Replace ``values`` (float32 or float64), in place, by their 3x3 mean over the cells where ``mask`` is True.

    Cells with no masked cell in their 3x3 window keep their value.  The
    result is that of the masked values' and the mask's
    ``scipy.ndimage.uniform_filter`` (``mode='nearest'``), each times 3,
    divided, to the last bit (see ``_smooth_3x3``).
    """
    _smooth_3x3(np.asarray(mask, dtype=np.bool_), values, values.dtype.type(3))

    

def Write_Output_Raster_As_GeoDataFrame(raster_data, ncols, nrows, dem_geotransform, dem_projection, s_output_type):
    """Polygonise a raster and return the polygons of its positive cells as a GeoDataFrame."""
    # Imported here: only the GeoPackage needs them
    import geopandas as gpd
    import shapely

    driver = gdal.GetDriverByName('MEM')
    raster_ds = driver.Create('', xsize=ncols, ysize=nrows, bands=1, eType=s_output_type)
    raster_ds.SetGeoTransform(dem_geotransform)
    raster_ds.SetProjection(dem_projection)
    raster_ds.GetRasterBand(1).WriteArray(raster_data)

    # Set NoData value to NaN, and auto-generate a mask band that respects it
    nodata_value = np.nan
    raster_ds.GetRasterBand(1).SetNoDataValue(nodata_value)
    mask_band = raster_ds.GetRasterBand(1).GetMaskBand()

    # Create an in-memory vector layer for the polygonized data
    memory_driver = (ogr.GetDriverByName('MEM') or ogr.GetDriverByName('Memory'))
    vector_ds = memory_driver.CreateDataSource('')
    srs = osr.SpatialReference()
    srs.ImportFromWkt(dem_projection)
    layer = vector_ds.CreateLayer('polygons', srs=srs)

    field = ogr.FieldDefn("Value", ogr.OFTInteger)
    layer.CreateField(field)

    gdal.Polygonize(raster_ds.GetRasterBand(1), mask_band, layer, 0, [], callback=None)

    if hasattr(layer, "__arrow_c_stream__"):
        # Every polygon at once, as WKB (GDAL 3.8 and later)
        table = pa.RecordBatchReader.from_stream(layer).read_all()
        polygons = shapely.from_wkb(table.column("wkb_geometry").to_numpy(zero_copy_only=False))
        values = table.column("Value").to_numpy().astype(np.int64)
    else:
        features = list(layer)
        polygons = shapely.from_wkb([bytes(feature.GetGeometryRef().ExportToWkb()) for feature in features])
        values = np.array([feature.GetField("Value") for feature in features], dtype=np.int64)

    flood_gdf = gpd.GeoDataFrame({'Value': values, 'geometry': polygons})
    flood_gdf.set_crs(dem_projection, inplace=True)

    # filter to only the flooded area
    flood_gdf = flood_gdf[flood_gdf['Value']>0]

    raster_ds = None
    vector_ds = None

    return flood_gdf

@njit(cache=True, parallel=True)
def _flooded_kind(values: np.ndarray, flood: np.ndarray) -> np.ndarray:
    """``_SOURCE`` where a cell floods with a value, ``_TARGET`` where it floods without one (NaN), else 0."""
    kind = np.zeros(values.shape, dtype=np.uint8)
    for r in prange(values.shape[0]):
        for c in range(values.shape[1]):
            if flood[r, c] > 0:
                kind[r, c] = _TARGET if np.isnan(values[r, c]) else _SOURCE

    return kind

@njit(cache=True, parallel=True)
def _matched_to_flood(values: np.ndarray, flood: np.ndarray, kind: np.ndarray, eps: np.float32, filled: np.ndarray,
                      in_place: bool):
    """``filled`` (float32, ``values`` itself ``in_place``) set to ``values``, but each target (in ``kind``) to the
    value of its nearest source, a target without any ``eps``, and NaN where ``flood`` is 0 or less.  In place, the
    sources are never written: they keep their values.

    The nearest source is the one SciPy's ``distance_transform_edt`` (Maurer
    et al. 2003) finds, ties and all: each cell's nearest source along its
    column, then, along each row, the nearest of those.  But only the rows
    with a target are worked out, in parallel, from each column's list of
    sources, where the transform keeps two ``int32`` of every cell."""
    nrows, ncols = values.shape
    target_rows, _ = _target_rows(kind)
    col_starts, col_rows = _sources_by_column(kind)
    with_sources = col_rows.size > 0
    # The targets, a chunk of their rows a thread
    if with_sources:
        for chunk in prange((target_rows.size + _NEAREST_CHUNK_ROWS - 1) // _NEAREST_CHUNK_ROWS):
            k0 = chunk * _NEAREST_CHUNK_ROWS
            at = np.empty(ncols, dtype=np.int64)
            near = np.empty(ncols, dtype=np.int64)
            g = np.empty(ncols, dtype=np.int64)
            _first_at_or_below(target_rows[k0], col_starts, col_rows, at)
            for k in range(k0, min(k0 + _NEAREST_CHUNK_ROWS, target_rows.size)):
                r = target_rows[k]
                top = _row_sources(r, col_starts, col_rows, at, near, g)
                j = 0
                for i in range(ncols):
                    j = _next_nearest(r, i, j, top, near, g)
                    if kind[r, i] & _TARGET:
                        filled[r, i] = np.float32(values[near[g[j]], g[j]])

    # Every other cell
    for r in prange(nrows):
        for c in range(ncols):
            if kind[r, c] & _TARGET:
                if not with_sources:
                    filled[r, c] = eps
                continue
            
            if flood[r, c] <= 0:
                filled[r, c] = np.float32(np.nan)
            elif not in_place:
                filled[r, c] = np.float32(values[r, c])

    return filled

def Flood_Flooded_Cells_in_Map(Array_Ensemble: np.ndarray, Flood_Ensemble: np.ndarray, eps=0.01,
                               in_place: bool = False):
    """Match a value raster (depth, WSE, velocity, slope) to the flood extent.

    Flooded cells that are NaN take the value of the nearest flooded cell that
    has one (see ``_matched_to_flood``); flooded cells with no such cell get
    ``eps``.  Cells outside the flood become NaN.  Returns a new float32 array,
    or with ``in_place`` ``Array_Ensemble`` itself, which must be float32.
    """
    if in_place and Array_Ensemble.dtype != np.float32:
        raise TypeError(f"Only a float32 raster is matched to the flood in place, not {Array_Ensemble.dtype}.")
    kind = _flooded_kind(Array_Ensemble, Flood_Ensemble)
    filled = Array_Ensemble if in_place else np.empty(Array_Ensemble.shape, dtype=np.float32)
    return _matched_to_flood(Array_Ensemble, Flood_Ensemble, kind, np.float32(eps), filled, in_place)

@njit(cache=True, parallel=True)
def _target_rows(kind: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rows with a target, and where each one's targets start among all the targets, row by row (one more)."""
    nrows, ncols = kind.shape
    per_row = np.zeros(nrows, dtype=np.int64)
    for r in prange(nrows):
        n = 0
        for c in range(ncols):
            if kind[r, c] & _TARGET:
                n += 1
        per_row[r] = n

    rows = np.flatnonzero(per_row)
    starts = np.zeros(rows.size + 1, dtype=np.int64)
    starts[1:] = np.cumsum(per_row[rows])
    return rows, starts

@njit(cache=True, parallel=True)
def _sources_by_column(kind: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each column's source rows, ascending: column ``c``'s are ``rows[starts[c]:starts[c + 1]]``."""
    nrows, ncols = kind.shape
    block = 64
    counts = np.zeros(ncols + 1, dtype=np.int64)
    for b in prange((ncols + block - 1) // block):
        for r in range(nrows):
            for c in range(b * block, min(b * block + block, ncols)):
                if kind[r, c] & _SOURCE:
                    counts[c + 1] += 1

    starts = np.cumsum(counts)
    rows = np.empty(starts[-1], dtype=np.int32)
    for b in prange((ncols + block - 1) // block):
        c0 = b * block
        c1 = min(c0 + block, ncols)
        at = starts[c0:c1].copy()
        for r in range(nrows):
            for c in range(c0, c1):
                if kind[r, c] & _SOURCE:
                    rows[at[c - c0]] = r
                    at[c - c0] += 1

    return starts, rows

@njit(cache=True, inline="always")
def _row_sources(r: int, col_starts: np.ndarray, col_rows: np.ndarray, at: np.ndarray, near: np.ndarray,
                 g: np.ndarray) -> int:
    """Row ``r``'s candidate sources: in each column the nearest source, the upper of two equally near, in ``near``
    (-1 in a column without one), and then those that are someone's nearest along the row, a stack in ``g``; returns
    its top (-1 for none).  ``at[c]`` is where column ``c``'s first source at or below the previous row was, and is
    moved on to ``r``'s.  The arithmetic is the distance transform's (see ``_matched_to_flood``)."""
    ncols = near.size
    for c in range(ncols):
        end = col_starts[c + 1]
        i = at[c]
        while i < end and col_rows[i] < r:
            i += 1
        at[c] = i
        up = col_rows[i - 1] if i > col_starts[c] else -1
        down = col_rows[i] if i < end else -1
        near[c] = down if down >= 0 and (up < 0 or down - r < r - up) else up

    top = -1
    for i in range(ncols):
        if near[i] < 0:
            continue
        fd = float(i)
        tw = float(near[i] - r)
        w = tw * tw
        while top >= 1:
            i1 = g[top]
            i2 = g[top - 1]
            f1 = float(i1)
            a = f1 - float(i2)
            b = fd - f1
            c = a + b
            tu = float(near[i2] - r)
            tv = float(near[i1] - r)
            u = tu * tu
            v = tv * tv
            if c * v - b * u - a * w - a * b * c <= 0.0:
                break
            top -= 1
        top += 1
        g[top] = i

    return top

@njit(cache=True, inline="always")
def _next_nearest(r: int, i: int, j: int, top: int, near: np.ndarray, g: np.ndarray) -> int:
    """Where along ``g`` (``_row_sources``) the nearest source of cell ``(r, i)`` is, from the previous cell's ``j``."""
    t0 = float(near[g[j]] - r)
    t1 = float(g[j] - i)
    d1 = t0 * t0 + t1 * t1
    while j < top:
        t0 = float(near[g[j + 1]] - r)
        t1 = float(g[j + 1] - i)
        d2 = t0 * t0 + t1 * t1
        if d1 <= d2:
            break
        d1 = d2
        j += 1

    return j

@njit(cache=True)
def _first_at_or_below(r: int, col_starts: np.ndarray, col_rows: np.ndarray, at: np.ndarray):
    """``at[c]``: where column ``c``'s first source at or below row ``r`` is among its sources."""
    for c in range(at.size):
        lo, hi = col_starts[c], col_starts[c + 1]
        at[c] = lo + np.searchsorted(col_rows[lo:hi], r)
