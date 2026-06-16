
#This code looks at a DEM raster to find the dimensions, then writes a script to create a STRM raster.
# built-in imports
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# third-party imports
try:
    import gdal 
    import osr 
    import ogr
    import gdal_array
    #from gdalconst import GA_ReadOnly
except: 
    from osgeo import gdal
    from osgeo import osr
    from osgeo import ogr
    from osgeo import gdal_array
    #from osgeo.gdalconst import GA_ReadOnly

from numba import njit, prange
from numba.core import types
from numba.typed import Dict
    
import yaml
import numpy as np
import pandas as pd
from pyproj import CRS, Geod
import geopandas as gpd
# from scipy.interpolate import interp1d
from scipy.ndimage import label, generate_binary_structure, distance_transform_edt
from scipy.spatial import cKDTree
from shapely.geometry import Point, shape
from curve2flood import LOG

gdal.UseExceptions()

COMID_FLOW_DICT_TYPE = dict[np.int32, np.float32]
ID_SYNONYMS = ["COMID", "RIVID", "river_id", "LINKNO"]
GeoTransform = tuple[float, float, float, float, float, float]

def _normalize_geotransform(transform) -> GeoTransform:
    """
    Return a GDAL geotransform tuple.

    Accepts native GDAL tuples and Affine-like objects that expose `to_gdal()`.
    """
    if hasattr(transform, "to_gdal"):
        values = tuple(float(v) for v in transform.to_gdal())
    else:
        values = tuple(float(v) for v in transform)
    if len(values) != 6:
        raise ValueError(f"Expected a 6-element geotransform, received {len(values)} values.")
    return values

def _numpy_dtype_from_output_type(output_type) -> np.dtype:
    if isinstance(output_type, int):
        np_type = gdal_array.GDALTypeCodeToNumericTypeCode(output_type)
        if np_type is None:
            raise TypeError(f"Unsupported GDAL output type: {output_type}")
        return np.dtype(np_type)
    return np.dtype(output_type)

def _gdal_dtype_from_output_type(output_type) -> int:
    if isinstance(output_type, int):
        return int(output_type)
    np_dtype = np.dtype(output_type)
    if np_dtype == np.dtype(bool):
        return int(gdal.GDT_Byte)
    gdal_type = gdal_array.NumericTypeCodeToGDALTypeCode(np_dtype)
    if gdal_type is None:
        raise TypeError(f"Unsupported NumPy output type: {output_type}")
    return int(gdal_type)

def rasterize_shapes_gdal(
    shapes,
    out_shape: tuple[int, int],
    transform,
    fill=0,
    dtype=np.uint8,
    all_touched: bool = False,
    projection_wkt: str | None = None,
) -> np.ndarray:
    """
    Rasterize shapely geometries to a NumPy array using GDAL.
    """
    geotransform = _normalize_geotransform(transform)
    np_dtype = _numpy_dtype_from_output_type(dtype)
    gdal_dtype = _gdal_dtype_from_output_type(dtype)

    raster_driver = gdal.GetDriverByName("MEM")
    raster_ds = raster_driver.Create("", xsize=int(out_shape[1]), ysize=int(out_shape[0]), bands=1, eType=gdal_dtype)
    raster_ds.SetGeoTransform(geotransform)
    if projection_wkt:
        raster_ds.SetProjection(str(projection_wkt))

    band = raster_ds.GetRasterBand(1)
    band.WriteArray(np.full(out_shape, fill, dtype=np_dtype))

    vector_driver = ogr.GetDriverByName("MEM") or ogr.GetDriverByName("Memory")
    vector_ds = vector_driver.CreateDataSource("")
    layer_srs = None
    if projection_wkt:
        layer_srs = osr.SpatialReference()
        layer_srs.ImportFromWkt(str(projection_wkt))
    layer = vector_ds.CreateLayer("shapes", srs=layer_srs, geom_type=ogr.wkbUnknown)

    field_name = "burn"
    layer.CreateField(ogr.FieldDefn(field_name, ogr.OFTReal))
    layer_defn = layer.GetLayerDefn()

    feature_count = 0
    for geom, value in shapes:
        if geom is None or value is None:
            continue
        if hasattr(geom, "is_empty") and geom.is_empty:
            continue
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            continue
        shapely_geom = geom if hasattr(geom, "wkb") else shape(geom)
        ogr_geom = ogr.CreateGeometryFromWkb(shapely_geom.wkb)
        feature = ogr.Feature(layer_defn)
        feature.SetField(field_name, float(value))
        feature.SetGeometry(ogr_geom)
        layer.CreateFeature(feature)
        feature = None
        feature_count += 1

    if feature_count > 0:
        options = [f"ATTRIBUTE={field_name}"]
        if all_touched:
            options.append("ALL_TOUCHED=TRUE")
        err = gdal.RasterizeLayer(raster_ds, [1], layer, options=options)
        if err != 0:
            raise RuntimeError(f"GDAL rasterization failed with error code {err}.")

    array = band.ReadAsArray()
    vector_ds = None
    raster_ds = None
    return np.asarray(array)

def _parse_optional_bool(value, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "y")

def _parse_optional_float(value, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    return float(value)

def _parse_optional_int(value, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    return int(value)

def find_id_column(columns, context: str) -> str:
    """Find a stream identifier column using the FHS synonym set."""
    col_map = {str(c).lower(): c for c in columns}
    for candidate in ID_SYNONYMS:
        key = candidate.lower()
        if key in col_map:
            return col_map[key]
    raise ValueError(
        f"Could not find an ID column in {context}. Expected one of: {ID_SYNONYMS}. "
        f"Available columns: {list(columns)}"
    )

def get_curve_columns(columns) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Return sorted q_, v_, t_, and wse_ column groups from an FHS/VDT table.
    """
    q_cols = sorted([c for c in columns if str(c).startswith("q_")], key=lambda x: int(str(x).split("_")[1]))
    v_cols = sorted([c for c in columns if str(c).startswith("v_")], key=lambda x: int(str(x).split("_")[1]))
    t_cols = sorted([c for c in columns if str(c).startswith("t_")], key=lambda x: int(str(x).split("_")[1]))
    wse_cols = sorted([c for c in columns if str(c).startswith("wse_")], key=lambda x: int(str(x).split("_")[1]))
    if not len(q_cols) == len(v_cols) == len(t_cols) == len(wse_cols):
        raise ValueError("Mismatch in number of q_, v_, t_, and wse_ columns.")
    if len(q_cols) == 0:
        raise ValueError("No q_, v_, t_, wse_ columns found in the VDT file.")
    return q_cols, v_cols, t_cols, wse_cols

def interp_row_clamped(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    """
    FHS-style 1D interpolation with endpoint clamping.
    """
    n = xp.size
    if n == 0:
        return np.nan
    if n == 1:
        return float(fp[0])
    if x <= xp[0]:
        return float(fp[0])
    if x >= xp[-1]:
        return float(fp[-1])
    idx = np.searchsorted(xp, x)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]
    if x1 == x0:
        return float(y0)
    w = (x - x0) / (x1 - x0)
    return float(y0 + w * (y1 - y0))

def clean_and_interp_one_row(target_q: float, q_row: np.ndarray, v_row: np.ndarray, t_row: np.ndarray, wse_row: np.ndarray) -> tuple[float, float, float]:
    """
    Clean one VDT row and interpolate velocity, top width, and WSE at `target_q`.
    This is a direct port of the FHS row-interpolation logic.
    """
    valid = (
        np.isfinite(q_row) &
        np.isfinite(v_row) &
        np.isfinite(t_row) &
        np.isfinite(wse_row) &
        (q_row > 0.0) &
        (v_row >= 0.0) &
        (t_row > 0.0) &
        (wse_row > -90.0)
    )
    if not np.any(valid):
        return np.nan, np.nan, np.nan

    q = q_row[valid]
    v = v_row[valid]
    t = t_row[valid]
    wse = wse_row[valid]

    order = np.argsort(q)
    q = q[order]
    v = v[order]
    t = t[order]
    wse = wse[order]

    q_unique, unique_idx = np.unique(q, return_index=True)
    v = v[unique_idx]
    t = t[unique_idx]
    wse = wse[unique_idx]

    out_v = interp_row_clamped(target_q, q_unique, v)
    out_t = interp_row_clamped(target_q, q_unique, t)
    out_wse = interp_row_clamped(target_q, q_unique, wse)
    return out_v, out_t, out_wse

def read_manning_table(s_manning_path: str, da_input_mannings: np.ndarray):
    """
    Reads the Manning's n information from the input file

    Parameters
    ----------
    s_manning_path: str
        Path to the Manning's n input table
    da_input_mannings: ndarray
        Array holding the mannings estimates

    Returns
    -------
    da_input_mannings: ndarray
        Array holding the mannings estimates

    """

    # Open and read the input file
    df = pd.read_csv(s_manning_path, sep='\t')

    # Create a lookup array for the Manning's n values
    # This is the fastest way to reclassify the values in the input array
    idx = df.iloc[:, 0].astype(int).values
    lookup_array = np.zeros(idx.max() + 1)
    lookup_array[idx] = df.iloc[:, 2].values
    da_input_mannings = lookup_array[da_input_mannings.astype(int)]
    # Return to the calling function
    return da_input_mannings

from scipy import ndimage as ndi

def create_velocity(OutVEL, Depth_Array, LU_Manning_n, LC_array, Slope_array_list,
                           geotransform, projection, ncols, nrows,
                           Flood_Ensemble, S):
    """
    """

    # find the maximum slope across ensembles, ignoring NaNs when calculating an average with NaNs mixed in
    # Stack them into one 3D array
    Slope_Array = create_positive_max_array(Slope_array_list)

    # Read Manning's n raster ---
    da_input_mannings = read_manning_table(LU_Manning_n, LC_array).astype(np.float32)

    # use Manning's solutions that assumes each pixel is a rectangular channel
    VEL_Array = (1/(da_input_mannings))*((Depth_Array)**(2/3))*((Slope_Array)**(1/2))

    VEL_Array = Flood_Flooded_Cells_in_Map(VEL_Array, Flood_Ensemble, eps=0.01)

    nodata_value = np.nan
    out_band_data = np.where(np.isnan(VEL_Array), nodata_value, VEL_Array).astype(np.float32)

    driver = gdal.GetDriverByName("GTiff")
    ds: gdal.Dataset = driver.Create(
        OutVEL, ncols, nrows, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "PREDICTOR=2", "TILED=YES"]
    )
    if ds is None:
        raise RuntimeError(f"Failed to create output raster: {OutVEL}")

    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)

    band = ds.GetRasterBand(1)
    band.WriteArray(out_band_data)
    band.SetNoDataValue(nodata_value)
    band.FlushCache()
    ds.FlushCache()

    band = None
    ds = None

    return

def convert_cell_size(
    d_dem_cell_size_x: float,
    d_dem_cell_size_y: float,
    d_dem_lower_left: float,
    d_dem_upper_right: float,
    s_dem_projection: str
):
    """
    Converts DEM cell size to x/y resolution in meters.

    For geographic rasters (degrees), this uses pyproj geodesic distances
    on the DEM ellipsoid. For projected rasters, it returns the original
    map-unit cell size for x and y.

    Parameters
    ----------
    d_dem_cell_size_x: float
        DEM x cell size (degrees for geographic rasters; map units otherwise)
    d_dem_cell_size_y: float
        DEM y cell size (degrees for geographic rasters; map units otherwise)
    d_dem_lower_left: float
        Lower-left y value (latitude for geographic rasters)
    d_dem_upper_right: float
        Upper-right y value (latitude for geographic rasters)
    s_dem_projection: str
        DEM projection WKT/CRS definition

    Returns
    -------
    d_x_cell_size: float
        Resolution of the cells in x direction (meters for geographic rasters)
    d_y_cell_size: float
        Resolution of the cells in y direction (meters for geographic rasters)
    d_projection_conversion_factor: float
        Mean meters-per-degree factor used for conversion

    """

    # Default output for projected/non-geographic rasters
    d_dem_cell_size_x = np.fabs(d_dem_cell_size_x)
    d_dem_cell_size_y = np.fabs(d_dem_cell_size_y)
    d_x_cell_size = d_dem_cell_size_x
    d_y_cell_size = d_dem_cell_size_y
    d_projection_conversion_factor = 1

    # Parse DEM CRS and use geodesic conversion for geographic grids.
    try:
        o_crs = CRS.from_user_input(s_dem_projection)
    except Exception as e:
        raise ValueError("Unable to parse DEM projection for cell-size conversion.") from e

    if o_crs.is_geographic:
        d_lat = (d_dem_lower_left + d_dem_upper_right) / 2.0
        d_lon = 0.0  # Geodesic spacing at a reference longitude

        # Build a geodesic calculator from the DEM ellipsoid.
        o_ellps = o_crs.ellipsoid
        if o_ellps is not None and o_ellps.semi_major_metre and o_ellps.inverse_flattening:
            o_geod = Geod(a=o_ellps.semi_major_metre, rf=o_ellps.inverse_flattening)
        else:
            o_geod = Geod(ellps="WGS84")

        # North-south cell spacing (meters)
        _, _, d_y_cell_size = o_geod.inv(d_lon, d_lat, d_lon, d_lat + d_dem_cell_size_y)
        # East-west cell spacing (meters) at midpoint latitude
        _, _, d_x_cell_size = o_geod.inv(d_lon, d_lat, d_lon + d_dem_cell_size_x, d_lat)

        d_x_cell_size = np.fabs(d_x_cell_size)
        d_y_cell_size = np.fabs(d_y_cell_size)
        d_projection_conversion_factor = 0.5 * (
            (d_x_cell_size / max(d_dem_cell_size_x, 1e-12))
            + (d_y_cell_size / max(d_dem_cell_size_y, 1e-12))
        )
    # if the raster is projected, we assume the cell size is already in meters and use it directly
    elif o_crs.is_projected:
        # For projected rasters, x/y map units are already meters based on CRS checks in main().
        d_x_cell_size = d_dem_cell_size_x
        d_y_cell_size = d_dem_cell_size_y
        d_projection_conversion_factor = 1.0


    # Return to the calling function
    return d_x_cell_size, d_y_cell_size, d_projection_conversion_factor

def FindFlowRateForEachCOMID_Ensemble(FlowFileName: str, flow_event_num: int) -> dict:  
    if FlowFileName.endswith('.parquet'):
        flow_df = pd.read_parquet(FlowFileName, engine='fastparquet')
    else:
        flow_df = pd.read_csv(FlowFileName, usecols=[0, flow_event_num + 1])

    # If any of the flow values are nan, send a little warning
    if flow_df.iloc[:, 1].isna().any():
        LOG.warning(f"Warning: NaN values found in flow data for event {flow_event_num}. These will be treated as zero flow.")
        flow_df.iloc[:, 1] = flow_df.iloc[:, 1].fillna(0)

    comid_q_dict = flow_df.set_index(flow_df.columns[0])[flow_df.columns[1]].to_dict()

    return comid_q_dict


def filter_outliers(group):
    """
    A function to filter outliers based on mean and standard deviation for each COMID group
    """
    # Calculate mean and standard deviation for TopWidth
    topwidth_mean = group['TopWidth'].mean()
    topwidth_std = group['TopWidth'].std()
    lower_bound_tw = topwidth_mean - 2 * topwidth_std
    upper_bound_tw = topwidth_mean + 2 * topwidth_std

    # Filter TopWidth outliers
    group = group[(group['TopWidth'] >= lower_bound_tw) & (group['TopWidth'] <= upper_bound_tw)]

    # Calculate mean and standard deviation for WSE
    wse_mean = group['WSE'].mean()
    wse_std = group['WSE'].std()
    lower_bound_wse = wse_mean - 2 * wse_std
    upper_bound_wse = wse_mean + 2 * wse_std

    # Filter WSE outliers
    group = group[(group['WSE'] >= lower_bound_wse) & (group['WSE'] <= upper_bound_wse)]

    # Calculate mean and standard deviation for Velocity
    wse_mean = group['Velocity'].mean()
    wse_std = group['Velocity'].std()
    lower_bound_wse = wse_mean - 2 * wse_std
    upper_bound_wse = wse_mean + 2 * wse_std

    # Filter WSE outliers
    group = group[(group['Velocity'] >= lower_bound_wse) & (group['Velocity'] <= upper_bound_wse)]

    return group

def compute_tw_multfact_scale(flow: np.ndarray,
                              qbase: np.ndarray,
                              k: float = 1,
                              min_s: float = 0.1,
                              max_s: float = 3) -> np.ndarray:
    flow = flow.astype(np.float32, copy=False)
    qbase = qbase.astype(np.float32, copy=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(qbase > 0, flow / qbase, 1.0)
        scale = np.power(ratio, k)
    scale = np.clip(scale, min_s, max_s)
    scale = np.where(np.isfinite(scale), scale, 1.0)
    return scale.astype(np.float32)

def Calculate_TW_D_V_ForEachCOMID_CurveFile(CurveParamFileName: str, COMID_Unique_Flow: dict, COMID_Unique, T_Rast, W_Rast, S_Rast, TW_MultFact, dx=None, dy=None):

    LOG.debug('\nOpening and Reading ' + CurveParamFileName)

    # read the curve data in as a Pandas dataframe
    if CurveParamFileName.endswith('.parquet'):
        curve_df = pd.read_parquet(CurveParamFileName, engine='fastparquet')
    else:
        curve_df = pd.read_csv(CurveParamFileName)

    # Add COMID flow information
    comid_flow_df = pd.DataFrame(COMID_Unique_Flow.items(), columns=['COMID', 'Flow'])

    # merging the curve and streamflow data together
    curve_df = curve_df.merge(comid_flow_df, on="COMID", how="left")

    # calculating depth and top-width with the COMID's discharge and the curve parameters
    curve_df['Depth'] = curve_df['depth_a']*curve_df['Flow']**curve_df['depth_b']
    curve_df['TopWidth'] = curve_df['tw_a']*curve_df['Flow']**curve_df['tw_b']
    curve_df['Velocity'] = curve_df['vel_a']*curve_df['Flow']**curve_df['vel_b']
    curve_df = curve_df[curve_df['Depth']>0]
    curve_df = curve_df[curve_df['TopWidth']>0]
    curve_df['WSE'] = curve_df['Depth'] + curve_df['BaseElev']

    # Apply the outlier filtering function to each COMID group
    curve_df = curve_df.groupby('COMID', group_keys=False)[curve_df.columns].apply(filter_outliers)

    if 'QBaseflow' in curve_df.columns:
        tw_scale = compute_tw_multfact_scale(
            curve_df['Flow'].values.astype(np.float32),
            curve_df['QBaseflow'].values.astype(np.float32),
        )
    else:
        tw_scale = np.ones(len(curve_df), dtype=np.float32)

    curve_df['TopWidth'] = curve_df['TopWidth'].astype(np.float32) * (TW_MultFact * tw_scale)

    # Fill in the T_Rast and W_Rast
    for index, row in curve_df.iterrows():
        T_Rast[int(row['Row']), int(row['Col'])] = row['TopWidth']
        W_Rast[int(row['Row']), int(row['Col'])] = row['Depth'] + row['BaseElev']
        if S_Rast is not None:
            S_Rast[int(row['Row']), int(row['Col'])] = row['Slope']

    # Calculate median values by COMID
    median_values = curve_df.groupby('COMID').agg({
        'TopWidth': 'median',
        'Depth': 'median',
        'WSE': 'median',
        'Velocity': 'median',
        'Row': 'first',
        'Col': 'first'
    })

    wse_stats = curve_df.groupby('COMID')['WSE'].agg(['mean', 'std'])
    
    # Map results back to the unique COMID list
    comid_result_df = pd.DataFrame({'COMID': COMID_Unique})
    comid_result_df = comid_result_df.merge(median_values, on='COMID', how='left').fillna(0)
    comid_wse_stats = comid_result_df[['COMID']].merge(wse_stats, on='COMID', how='left').fillna(0)

    # Create dicts
    comid_result_df['COMID'] = comid_result_df['COMID'].astype(np.int32)
    comid_result_df['TopWidth'] = comid_result_df['TopWidth'].astype(np.float32)
    comid_result_df['Depth'] = comid_result_df['Depth'].astype(np.float32)
    comid_result_df['Velocity'] = comid_result_df['Velocity'].astype(np.float32)
    
    # Create dicts
    COMID_Unique_TW = comid_result_df.set_index('COMID')['TopWidth'].to_dict()
    COMID_Unique_Depth = comid_result_df.set_index('COMID')['Depth'].to_dict()

    # Get the maximum TopWidth for all COMIDs
    TopWidthMax = comid_result_df['TopWidth'].max()

    return (
        COMID_Unique_TW,
        COMID_Unique_Depth,
        TopWidthMax,
        T_Rast,
        W_Rast,
        S_Rast,
    )

# @njit(cache=True)
@njit("float32(float32, float32[:], float32[:])", cache=True)
def Find_TopWidth_at_Baseflow_when_using_VDT(QB, flow_values, top_width_values):
    """
    Find the TopWidth corresponding to the baseflow (QB).
    
    Args:
        QB (float): Baseflow discharge value.
        flow_values (list-like): Array or list of flow values (q_1, q_2, ..., q_n).
        top_width_values (list-like): Array or list of TopWidth values (t_1, t_2, ..., t_n).

    Returns:
        float: TopWidth corresponding to QB.
    """
    # for i in range(len(flow_values)):
    #     if QB <= flow_values[i]:
    #         return top_width_values[i]
    # # If QB is larger than all flow values, return the last TopWidth
    # return top_width_values[-1]
    idx = np.searchsorted(flow_values, QB, side="left")
    
    if idx >= len(flow_values):
        # QB is larger than all flow values
        return top_width_values[-1]
    else:
        return top_width_values[idx]

# @njit(cache=True)
@njit("float32(float32[:], float32[:], float32)", cache=True)
def interp1d_numba(x: np.ndarray, y: np.ndarray, xi: float | int) -> float:
    """
    Linearly interpolates/extrapolates a single value xi based on 1D arrays x and y.
    Equivalent to calling: `interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate', assume_sorted=True)(xi)`
    Parameters:
    - x: 1D array (assumed sorted)
    - y: 1D array of same length as x
    - xi: scalar input to interpolate
    Returns:
    - yi: interpolated or extrapolated value
    """
    n = len(x)

    if xi <= x[0]:
        # extrapolate left
        return y[0] + (xi - x[0]) * (y[1] - y[0]) / (x[1] - x[0]) if (x[1] - x[0]) != 0 else y[0]
    elif xi >= x[n - 1]:
        # extrapolate right
        return y[n - 2] + (xi - x[n - 2]) * (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]) if (x[n - 1] - x[n - 2]) != 0 else y[n - 1]

    # binary search for correct interval
    low = 0
    high = n - 1
    while high - low > 1:
        mid = (high + low) // 2
        if x[mid] > xi:
            high = mid
        else:
            low = mid

    # linear interpolation
    x0 = x[low]
    x1 = x[high]
    y0 = y[low]
    y1 = y[high]

    return y0 + (xi - x0) * (y1 - y0) / (x1 - x0) if (x1 - x0) != 0 else y0

@njit("Tuple((float32[:], float32[:], float32[:], float32[:]))(float32[:], float32[:], float32[:,:], float32[:, :], float32[:], float32[:, :], float32[:, :], float32[:], float32[:])",
      cache=True)
def vdt_interpolate(flow: np.ndarray,
                    qb: np.ndarray, 
                    flow_values: np.ndarray, 
                    top_width_values: np.ndarray,
                    elev_values: np.ndarray,
                    wse_values: np.ndarray,
                    vel_values: np.ndarray,
                    e_dem: np.ndarray,
                    tw_mult_fact: np.ndarray) -> tuple[np.ndarray, ...]:
    top_width = np.empty_like(flow)
    depth = np.empty_like(flow)
    wse = np.empty_like(flow)
    baseflow_tw = np.empty_like(flow)
    vel = np.empty_like(flow)

    # Loop through each row in the DataFrame, interpolate as needed
    for i in range(len(flow)):
        if flow[i] <= qb[i]:
            # Below baseflow
            top_width[i] = Find_TopWidth_at_Baseflow_when_using_VDT(qb[i], flow_values[i], top_width_values[i])
            depth[i] = 0.001
            wse[i] = elev_values[i]
            vel[i] = vel_values[i][-1]
        elif flow[i] >= flow_values[i][-1]:
            # Above the maximum flow value
            top_width[i] = top_width_values[i][-1]
            wse[i] = wse_values[i][-1]
            depth[i] = wse[i] - e_dem[i]
            vel[i] = vel_values[i][-1]
        else:
            # Interpolate
            wse[i] = interp1d_numba(flow_values[i], wse_values[i], flow[i])
            top_width[i] = interp1d_numba(flow_values[i], top_width_values[i], flow[i])
            wse[i] = max(wse[i], e_dem[i])
            depth[i] = max(wse[i] - e_dem[i], 0.001)
            vel[i] = interp1d_numba(flow_values[i], vel_values[i], flow[i])

        baseflow_tw[i] = Find_TopWidth_at_Baseflow_when_using_VDT(qb[i], flow_values[i], top_width_values[i])

    # Ensure TopWidth respects baseflow and scale
    top_width = np.maximum(top_width, baseflow_tw) * tw_mult_fact
    
    return top_width, depth, wse, vel

def Calculate_TW_D_V_ForEachCOMID_VDTDatabase(E_DEM, VDTDatabaseFileName: str, COMID_Unique_Flow: dict, COMID_Unique, T_Rast, W_Rast, S_Rast, TW_MultFact, fast_vdt: bool, dx, dy):    

    LOG.debug('\nOpening and Reading ' + VDTDatabaseFileName)
    
    # Read the VDT Database into a DataFrame
    if VDTDatabaseFileName.endswith('.parquet'):
        vdt_df = pd.read_parquet(VDTDatabaseFileName, engine='fastparquet')
    else:
        vdt_df = pd.read_csv(VDTDatabaseFileName)
        
    if vdt_df.empty:
        raise ValueError("The VDT Database file is empty or could not be read properly.")
    
    # Add COMID flow information
    comid_flow_df = pd.DataFrame(COMID_Unique_Flow.items(), columns=['COMID', 'Flow'])
    vdt_df = vdt_df.merge(comid_flow_df, on='COMID', how='inner').copy()

    # Ensure row and col are integers
    vdt_df['Row'] = vdt_df['Row'].astype(int)
    vdt_df['Col'] = vdt_df['Col'].astype(int)
    
    # Extract the column indices for interpolation
    flow_cols = [list(vdt_df.columns).index(col) for col in vdt_df.columns if col.startswith('q_')]
    top_width_cols = [list(vdt_df.columns).index(col) for col in vdt_df.columns if col.startswith('t_')]
    wse_cols = [list(vdt_df.columns).index(col) for col in vdt_df.columns if col.startswith('wse_')]
    vel_cols = [list(vdt_df.columns).index(col) for col in vdt_df.columns if col.startswith('v_')]
    
    # Extract flow, baseflow, elevation, and Slope values
    flow = vdt_df['Flow'].values.astype(np.float32)
    qb = vdt_df['QBaseflow'].values.astype(np.float32)
    e_dem = E_DEM[vdt_df['Row'].values + 1, vdt_df['Col'].values + 1]

    # Extract flow, TopWidth, and WSE values for interpolation
    flow_values = vdt_df.iloc[:, flow_cols].values.astype(np.float32)
    top_width_values = vdt_df.iloc[:, top_width_cols].values.astype(np.float32)
    wse_values = vdt_df.iloc[:, wse_cols].values.astype(np.float32)
    vel_values = vdt_df.iloc[:, vel_cols].values.astype(np.float32)
    elev_values = vdt_df['Elev'].values.astype(np.float32)

    tw_scale = compute_tw_multfact_scale(flow, qb)
    tw_mult_fact = (TW_MultFact * tw_scale).astype(np.float32)
    top_width, depth, wse, velocity = vdt_interpolate(flow, qb, flow_values, top_width_values, elev_values, wse_values, vel_values, e_dem, tw_mult_fact)

    # Rebuild once before adding derived columns so pandas does not keep
    # appending blocks onto a highly fragmented wide frame.
    vdt_df = vdt_df.assign(
        TopWidth=top_width,
        Depth=depth,
        WSE=wse,
        Velocity=velocity,
    )

    # Drop rows with NaN values introduced during outlier removal
    vdt_df = vdt_df.dropna(subset=['TopWidth', 'Depth', 'WSE', 'Velocity']).copy()

    # Round the interpolated TopWidth, WSE, and Velocity to 2 decimal places
    vdt_df = vdt_df.assign(
        TopWidth=vdt_df['TopWidth'].round(2),
        WSE=vdt_df['WSE'].round(2),
        Velocity=vdt_df['Velocity'].round(2),
    )

    # Apply the outlier filtering function to each COMID group
    cols = ['TopWidth', 'WSE', 'Velocity']
    for col in cols:
        q01 = vdt_df.groupby('COMID')[col].transform(lambda x: x.quantile(0.01))
        q99 = vdt_df.groupby('COMID')[col].transform(lambda x: x.quantile(0.99))

        vdt_df = vdt_df[
            (vdt_df[col] >= q01) &
            (vdt_df[col] <= q99)
        ]
    
    # Fill T_Rast, W_Rast, and S_Rast
    T_Rast[vdt_df['Row'], vdt_df['Col']] = vdt_df['TopWidth']
    W_Rast[vdt_df['Row'], vdt_df['Col']] = vdt_df['WSE']
    if S_Rast is not None:
        S_Rast[vdt_df['Row'], vdt_df['Col']] = vdt_df['Slope']    
    
    # Calculate median values by COMID
    median_values = vdt_df.groupby('COMID').agg({
        'TopWidth': 'median',
        'Depth': 'median',
        'WSE': 'median',
        'Velocity': 'median',
    })

    wse_stats = vdt_df.groupby('COMID')['WSE'].agg(['mean', 'std'])
    
    # Map results back to the unique COMID list
    comid_result_df = pd.DataFrame({'COMID': COMID_Unique})
    comid_result_df = comid_result_df.merge(median_values, on='COMID', how='left').fillna(0)
    comid_wse_stats = comid_result_df[['COMID']].merge(wse_stats, on='COMID', how='left').fillna(0)
    comid_result_df['COMID'] = comid_result_df['COMID'].astype(np.int32)
    comid_result_df['TopWidth'] = comid_result_df['TopWidth'].astype(np.float32)
    comid_result_df['Depth'] = comid_result_df['Depth'].astype(np.float32)
    comid_result_df['Velocity'] = comid_result_df['Velocity'].astype(np.float32)
    
    # Create dicts
    COMID_Unique_TW = comid_result_df.set_index('COMID')['TopWidth'].to_dict()
    COMID_Unique_Depth = comid_result_df.set_index('COMID')['Depth'].to_dict()

    # Get the maximum TopWidth for all COMIDs
    TopWidthMax = comid_result_df['TopWidth'].max()

    return (
        COMID_Unique_TW,
        COMID_Unique_Depth,
        TopWidthMax,
        T_Rast,
        W_Rast,
        S_Rast
    )
  
def Get_Raster_Details(DEM_File):
    LOG.debug(DEM_File)
    gdal.Open(DEM_File, gdal.GA_ReadOnly)
    data = gdal.Open(DEM_File)
    geoTransform = data.GetGeoTransform()
    ncols = int(data.RasterXSize)
    nrows = int(data.RasterYSize)
    minx = geoTransform[0]
    dx = geoTransform[1]
    maxy = geoTransform[3]
    dy = geoTransform[5]
    maxx = minx + dx * ncols
    miny = maxy + dy * nrows
    Rast_Projection = data.GetProjectionRef()
    data = None
    return minx, miny, maxx, maxy, dx, dy, ncols, nrows, geoTransform, Rast_Projection

def Read_Raster_GDAL(InRAST_Name):
    dataset = gdal.Open(InRAST_Name, gdal.GA_ReadOnly)

    # Retrieve dimensions and geospatial metadata.
    geotransform = dataset.GetGeoTransform()
    band = dataset.GetRasterBand(1)
    RastArray = band.ReadAsArray()
    ncols = band.XSize
    nrows = band.YSize
    band = None

    # Normalize south-up rasters (pixel height > 0) to north-up arrays.
    if geotransform[5] > 0:
        LOG.warning('Raster appears south-up (positive pixel height); flipping to north-up: ' + str(InRAST_Name))
        RastArray = np.flipud(RastArray)
        geotransform = (
            geotransform[0],
            geotransform[1],
            geotransform[2],
            geotransform[3] + geotransform[5] * nrows,
            geotransform[4],
            -geotransform[5],
        )

    cellsize = geotransform[1]
    yll = geotransform[3] - nrows * np.fabs(geotransform[5])
    yur = geotransform[3]
    xll = geotransform[0]
    xur = xll + (ncols) * geotransform[1]
    lat = np.fabs((yll + yur) / 2.0)
    Rast_Projection = dataset.GetProjectionRef()
    dataset = None

    LOG.debug('Spatial Data for Raster File:')
    LOG.debug('   ncols = ' + str(ncols))
    LOG.debug('   nrows = ' + str(nrows))
    LOG.debug('   cellsize = ' + str(cellsize))
    LOG.debug('   yll = ' + str(yll))
    LOG.debug('   yur = ' + str(yur))
    LOG.debug('   xll = ' + str(xll))
    LOG.debug('   xur = ' + str(xur))
    return RastArray, ncols, nrows, cellsize, yll, yur, xll, xur, lat, geotransform, Rast_Projection

def GetListOfDEMs(inputfolder):
    DEM_Files = []
    for file in os.listdir(inputfolder):
        #if file.startswith('return_') and file.endswith('.geojson'):
        if file.endswith('.tif') or file.endswith('.img'):
            DEM_Files.append(file)
    return DEM_Files

def Write_Output_Raster(s_output_filename, raster_data, ncols, nrows, dem_geotransform, dem_projection, s_file_format, s_output_type, creation_options: list[str] = None):   
    o_driver = gdal.GetDriverByName(s_file_format)  #Typically will be a GeoTIFF "GTiff"
    #o_metadata = o_driver.GetMetadata()

    if creation_options is None:
        creation_options = ["COMPRESS=LZW", 'PREDICTOR=2']
    
    # Construct the file with the appropriate data shape
    o_output_file = o_driver.Create(s_output_filename, xsize=ncols, ysize=nrows, bands=1, eType=s_output_type, options=creation_options)    

    # Set the geotransform
    o_output_file.SetGeoTransform(dem_geotransform)
    
    # Set the spatial reference
    o_output_file.SetProjection(dem_projection)
    
    # Write the data to the file
    o_output_file.GetRasterBand(1).WriteArray(raster_data)
    
    # Once we're done, close properly the dataset
    o_output_file = None


#   Convert_GDF_to_Output_Raster(Flood_File, flood_gdf, 'Value', ncols, nrows, dem_geotransform, dem_projection, "GTiff", gdal.GDT_Int32)
def Convert_GDF_to_Output_Raster(s_output_filename, gdf, Param, ncols, nrows, dem_geotransform, dem_projection, s_file_format, s_output_type):   
    LOG.info(s_output_filename)

    # Rasterize geometries
    LOG.info('Rasterizing geometries')
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[Param]))  # Replace 'value_column' with your column name
    raster_data = rasterize_shapes_gdal(
        shapes=shapes,
        out_shape=(nrows, ncols),
        transform=dem_geotransform,
        fill=0,
        dtype=s_output_type,
        projection_wkt=dem_projection,
    )
    LOG.info('Writing output file')
    Write_Output_Raster(
        s_output_filename,
        raster_data,
        ncols,
        nrows,
        _normalize_geotransform(dem_geotransform),
        dem_projection,
        s_file_format,
        s_output_type,
    )
    return

def Write_Output_Raster_As_GeoDataFrame(raster_data, ncols, nrows, dem_geotransform, dem_projection, s_output_type):
    # Create an in-memory raster dataset
    driver = gdal.GetDriverByName('MEM')
    raster_ds = driver.Create('', xsize=ncols, ysize=nrows, bands=1, eType=s_output_type)

    # Set the geotransform and projection
    raster_ds.SetGeoTransform(dem_geotransform)
    raster_ds.SetProjection(dem_projection)

    # Write the data to the in-memory raster dataset
    raster_ds.GetRasterBand(1).WriteArray(raster_data)

    # Set NoData value to NaN
    nodata_value = np.nan
    raster_ds.GetRasterBand(1).SetNoDataValue(nodata_value)

    # Auto-generate a mask band that respects NoData
    mask_band = raster_ds.GetRasterBand(1).GetMaskBand()

    # Create an in-memory vector layer for the polygonized data
    memory_driver = (ogr.GetDriverByName('MEM') or ogr.GetDriverByName('Memory'))
    vector_ds = memory_driver.CreateDataSource('')
    srs = osr.SpatialReference()
    srs.ImportFromWkt(dem_projection)
    layer = vector_ds.CreateLayer('polygons', srs=srs)

    # Add a field to the layer
    field = ogr.FieldDefn("Value", ogr.OFTInteger)
    layer.CreateField(field)

    # Polygonize the raster and write to the vector layer
    gdal.Polygonize(raster_ds.GetRasterBand(1), mask_band, layer, 0, [], callback=None)

    # Convert the OGR layer to GeoPandas GeoDataFrame
    polygons = []
    values = []
    
    for feature in layer:
        geom = feature.GetGeometryRef()
        # Parse the JSON string to a dictionary
        geom_dict = json.loads(geom.ExportToJson())
        polygons.append(shape(geom_dict))        
        values.append(feature.GetField("Value"))

    # Create a GeoDataFrame
    flood_gdf = gpd.GeoDataFrame({'Value': values, 'geometry': polygons})

    # Set the CRS
    flood_gdf.set_crs(dem_projection, inplace=True)

    # filter to only the flooded area
    flood_gdf = flood_gdf[flood_gdf['Value']>0]

    # Clean up
    raster_ds = None
    vector_ds = None

    return flood_gdf


@njit(cache=True)
def FloodAllLocalAreas(WSE, E_Box, r_min, r_max, c_min, c_max, r_use, c_use):
    FourMatrix = np.full((3, 3), 4)
    
    # JLG commented this out because of an error but not sure the fix is correct
    # nrows_local = r_max - r_min + 2
    # ncols_local = c_max - c_min + 2
    # FloodLocal = np.zeros((nrows_local, ncols_local))
    nrows_local = np.int32(r_max - r_min + 2)
    ncols_local = np.int32(c_max - c_min + 2)
    FloodLocal = np.zeros((nrows_local,ncols_local), dtype=np.float32)
    
    FloodLocal[1:nrows_local-1,1:ncols_local-1] = np.where(E_Box<=WSE,1,0)
    
    # JLG commented this out because of an error but not sure the fix is correct
    #This is the Stream Cell.  Mark it with a 4
    # FloodLocal[(r_use-r_min+1),(c_use-c_min+1)] = 4 
    r_idx = int(r_use - r_min + 1)
    c_idx = int(c_use - c_min + 1)
    FloodLocal[r_idx, c_idx] = 4

    
    #Go through and mark all the cells that 
    for r in range((r_use-r_min+1),nrows_local-1):
        for c in range((c_use-c_min+1),ncols_local-1):
            #print(FloodLocal[r-1:r+2,c-1:c+2].shape)
            #print(FourMatrix.shape)
            #print(FloodLocal[r-1:r+2,c-1:c+2])
            if FloodLocal[r,c]>=3:
                FloodLocal[r-1:r+2,c-1:c+2] = FloodLocal[r-1:r+2,c-1:c+2] * FourMatrix
    for r in range((r_use-r_min+1), 0, -1):
        for c in range((c_use-c_min+1), 0, -1):
            if FloodLocal[r,c]>=3:
                FloodLocal[r-1:r+2,c-1:c+2] = FloodLocal[r-1:r+2,c-1:c+2] * FourMatrix
    
    for r in range(1, nrows_local-1):
        for c in range(1, ncols_local-1):
            if FloodLocal[r,c]>=3:
                FloodLocal[r-1:r+2,c-1:c+2] = FloodLocal[r-1:r+2,c-1:c+2] * FourMatrix
    
    #print(FloodLocal)
    #FloodReturn = np.where(FloodLocal[1:nrows_local-1,1:ncols_local-1]>0.0,1.0,0.0)
    #print(np.where(FloodLocal[1:nrows_local-1,1:ncols_local-1]>3.0,1.0,0.0))
    return np.where(FloodLocal[1:nrows_local-1,1:ncols_local-1]>3.0,1.0,0.0)

@njit(cache=True)
def CreateWeightAndElipseMask(TW_temp, dx, dy, TW_MultFact):
    TW = int(TW_temp)  #This is the number of cells in the top-width

    # 

    # ElipseMask = np.zeros((TW+1,int(TW*2+1),int(TW*2+1)))  #3D Array
    WeightBox = np.zeros((int(TW*2+1),int(TW*2+1)), dtype=np.float32)  #2D Array
    # ElevMask = np.ones((int(TW*2+1),int(TW*2+1)))  #2D Array    #This is set and used later, and only if limit_low_elev_flooding=True


    # for i in range(1,TW+1):
    #     TWDX = i*dx*i*dx
    #     TWDY = i*dy*i*dy
    #     for r in range(0,i+1):
    #         for c in range(0,i+1):
    #             is_elipse = (c*dx*c*dx/(TWDX)) + (r*dy*r*dy/(TWDY))   #https://www.mathopenref.com/coordgeneralellipse.html
    #             if is_elipse<=1.0:
    #                 ElipseMask[i,TW+r,TW+c] = 1.0
    #                 ElipseMask[i,TW-r,TW+c] = 1.0
    #                 ElipseMask[i,TW+r,TW-c] = 1.0
    #                 ElipseMask[i,TW-r,TW-c] = 1.0
    # print(ElipseMask[2,TW-4:TW+4+1,TW-4:TW+4+1].astype(int))
    # print(ElipseMask[10,TW-14:TW+14+1,TW-14:TW+14+1].astype(int))
    # print(ElipseMask[40,TW-44:TW+44+1,TW-44:TW+44+1].astype(int))


    # --- Vectorized WeightBox creation ---
    n = 2*TW + 1
    # Create an array of indices [0, 1, ..., n-1]
    indices = np.arange(n)
    # Compute offsets from the center (center index is TW)
    # These offsets represent the "cell distance" in each direction.
    Y = indices - TW  # shape (n,)
    X = indices - TW  # shape (n,)
    # Broadcast to compute the squared distances:
    # For every cell, z2 = (dx * (x offset))^2 + (dy * (y offset))^2.
    # We use broadcasting: the row vector (X) and column vector (Y) combine to form an (n x n) array.
    z2 = (X * dx)**2  # shape (n,)
    z2 = z2[None, :] + ((Y * dy)**2)[:, None]  # shape (n, n)
    # Avoid very small values (to prevent division by zero)
    z2 = np.where(z2 < 0.0001, 0.0001, z2)
    WeightBox = 1.0 / z2
    
    # for r in range(0,TW+1):
    #     for c in range(0,TW+1):
    #         z2 = c*dx*c*dx + r*dy*r*dy
    #         if z2<0.0001:
    #             z2=0.001
    #         WeightBox[TW+r,TW+c] = 1 / (z2)
    #         WeightBox[TW-r,TW+c] = 1 / (z2)
    #         WeightBox[TW+r,TW-c] = 1 / (z2)
    #         WeightBox[TW-r,TW-c] = 1 / (z2)
    
    # return WeightBox, ElipseMask

    return WeightBox

@njit("float32[:,:](int32, float32, float32)", cache=True)
def create_weightbox(tw: int, dx: float, dy: float):
    # tw is the number of cells in the top-width

    # --- Vectorized WeightBox creation ---
    n = 2*tw + 1
    # Create an array of indices [0, 1, ..., n-1]
    indices = np.arange(n)
    # Compute offsets from the center (center index is TW)
    # These offsets represent the "cell distance" in each direction.
    X = ((indices - tw) * dx) ** 2 # shape (n,)
    Y = ((indices - tw) * dy) ** 2 # shape (n,)
    
    # Broadcast to compute the squared distances:
    # For every cell, z2 = (dx * (x offset))^2 + (dy * (y offset))^2.
    # We use broadcasting: the row vector (X) and column vector (Y) combine to form an (n x n) array.
    WeightBox = X[None, :] + Y[:, None]  # shape (n, n)
    # Avoid very small values (to prevent division by zero)
    WeightBox = np.clip(WeightBox, 0.0001, None)
    WeightBox = 1.0 / WeightBox


    return (WeightBox).astype(np.float32)

@njit(cache=True)
def get_owner_stats_all(WSE: np.ndarray, owner: np.ndarray, nrows: int, ncols: int):
    """Compute median and std-dev of WSE values for *each* owner id.

    This avoids calling get_local_stats for every cell (O(N^3)) and
    trades memory for CPU by scanning the grid per owner.

    Returns
    -------
    owner_median, owner_std : 1D float32 arrays of length (max_owner + 1)
        owner_median[i] / owner_std[i] are NaN when that owner has <2 valid values.
    """
    # Find the maximum owner id so we can size lookup tables
    max_owner = 0
    for r in range(nrows):
        for c in range(ncols):
            oid = owner[r, c]
            if oid > max_owner:
                max_owner = oid

    # Count valid (non-NaN) values per owner
    counts = np.zeros(max_owner + 1, dtype=np.int32)
    for r in range(nrows):
        for c in range(ncols):
            oid = owner[r, c]
            if oid <= 0:
                continue
            val = WSE[r, c]
            if not np.isnan(val):
                counts[oid] += 1

    # Compute stats per owner
    owner_median = np.full(max_owner + 1, np.nan, dtype=np.float32)
    owner_std    = np.full(max_owner + 1, np.nan, dtype=np.float32)

    for oid in range(1, max_owner + 1):
        n = counts[oid]
        if n < 2:
            continue
        vals = np.empty(n, dtype=np.float32)
        idx = 0
        for r in range(nrows):
            for c in range(ncols):
                if owner[r, c] != oid:
                    continue
                val = WSE[r, c]
                if not np.isnan(val):
                    vals[idx] = val
                    idx += 1

        vals = np.sort(vals)

        # Median
        if n % 2 == 1:
            med = vals[n // 2]
        else:
            med = 0.5 * (vals[n // 2 - 1] + vals[n // 2])

        # Mean
        mean_val = 0.0
        for i in range(n):
            mean_val += vals[i]
        mean_val /= n

        # Std (population std to match previous implementation)
        sum_sq = 0.0
        for i in range(n):
            diff = vals[i] - mean_val
            sum_sq += diff * diff
        std = np.sqrt(sum_sq / n)

        owner_median[oid] = np.float32(med)
        owner_std[oid]    = np.float32(std)

    return owner_median, owner_std

@njit(cache=True)
def get_local_stats(r, c, WSE, nrows, ncols):
    """
    Calculates the median and standard deviation of valid (non-NaN) WSE values
    in the 3x3 neighborhood of (r, c).
    """
    vals = np.empty(9, dtype=np.float32)
    count = 0

    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if 0 <= rr < nrows and 0 <= cc < ncols:
                val = WSE[rr, cc]
                if not np.isnan(val):
                    vals[count] = val
                    count += 1

    if count < 2:
        return np.nan, np.nan

    valid_vals = vals[:count]
    for i in range(count):
        for j in range(0, count - i - 1):
            if valid_vals[j] > valid_vals[j + 1]:
                tmp = valid_vals[j]
                valid_vals[j] = valid_vals[j + 1]
                valid_vals[j + 1] = tmp

    if count % 2 == 1:
        median = valid_vals[count // 2]
    else:
        median = 0.5 * (valid_vals[count // 2 - 1] + valid_vals[count // 2])

    sum_sq_diff = 0.0
    mean_val = 0.0
    for i in range(count):
        mean_val += valid_vals[i]
    mean_val /= count

    for i in range(count):
        sum_sq_diff += (valid_vals[i] - mean_val) ** 2

    std = np.sqrt(sum_sq_diff / count)

    return median, std
@njit(cache=True)
def _heap_push(r_heap, c_heap, p_heap, size, r, c, p):
    i = size
    size += 1
    r_heap[i] = r
    c_heap[i] = c
    p_heap[i] = p

    while i > 0:
        parent = (i - 1) // 2
        if p_heap[parent] <= p_heap[i]:
            break
        # swap parent and child
        tmp_r = r_heap[parent]
        tmp_c = c_heap[parent]
        tmp_p = p_heap[parent]
        r_heap[parent] = r_heap[i]
        c_heap[parent] = c_heap[i]
        p_heap[parent] = p_heap[i]
        r_heap[i] = tmp_r
        c_heap[i] = tmp_c
        p_heap[i] = tmp_p
        i = parent

    return size

@njit(cache=True)
def _heap_pop(r_heap, c_heap, p_heap, size):
    r = r_heap[0]
    c = c_heap[0]
    p = p_heap[0]
    size -= 1
    if size > 0:
        r_heap[0] = r_heap[size]
        c_heap[0] = c_heap[size]
        p_heap[0] = p_heap[size]

        i = 0
        while True:
            left = 2 * i + 1
            right = left + 1
            if left >= size:
                break
            smallest = left
            if right < size and p_heap[right] < p_heap[left]:
                smallest = right
            if p_heap[i] <= p_heap[smallest]:
                break
            # swap
            tmp_r = r_heap[i]
            tmp_c = c_heap[i]
            tmp_p = p_heap[i]
            r_heap[i] = r_heap[smallest]
            c_heap[i] = c_heap[smallest]
            p_heap[i] = p_heap[smallest]
            r_heap[smallest] = tmp_r
            c_heap[smallest] = tmp_c
            p_heap[smallest] = tmp_p
            i = smallest

    return r, c, p, size

@njit(cache=True)
def fldpln(WSE_Initial, E, flowdir, stream_id, nrows, ncols, dx, dy):
    """
    This function is meant to mimic the FLDPLN model developed at the University of Kansas, translated iteratively
    using Codex-ChatGPT and this repository: https://github.com/AlabamaWaterInstitute/fldpln and this documentation: 
    https://services.kars.geoplatform.ku.edu/fldpln/AGU_2023_Operational_FIM_in_Kansas.pdf and 
    https://kuscholarworks.ku.edu/server/api/core/bitstreams/df102f13-5968-4e45-ad04-91b4d8086de0/content

    Propagate each seeded WSE upstream along reverse flow direction, then iteratively
    perform boundary spillover and upstream backfill to steady-state.
    A cell is inundated if its elevation + 0.1 m is lower than the seed WSE and it drains to
    that seed cell. If multiple seeds reach a cell, keep the maximum WSE.

    Inputs:
    - WSE_Initial: seeded WSE raster (nan/-9998 where dry); WSE for each stream cell
    - E: ground elevation raster
    - flowdir: D8 flow direction grid
    - stream_id: stream/segment ids for source labeling
    - nrows/ncols: raster dimensions
    - dx, dy: cell dimensions, in meters
    
    Notes:
    - Spillover candidates are dry boundary cells adjacent to wet cells where WSE_Out > E.
    - Candidate depth is selected from wet neighbors using a minimum required depth
      (tie-breaker: highest boundary elevation).
    - Spillover floods the candidate point and then backfills upstream (reverse flowdir)
      to the spill depth until steady-state.

    Returns WSE_Out or the WSE Array for a one set of streamflow inputs
    """
    # Initialize outputs: WSE_Out holds max WSE per cell, fsp holds source stream id, dtf holds first inundation stage.
    WSE_Out = np.full((nrows, ncols), np.nan, dtype=np.float32)

    # Preallocate BFS queue and per-seed visitation map for reverse-flow traversal.
    max_q = nrows * ncols
    q_r = np.empty(max_q, dtype=np.int32)
    q_c = np.empty(max_q, dtype=np.int32)
    visit_id = np.zeros((nrows, ncols), dtype=np.int32)
    seed_id = 0
    # Track adjacent dry neighbors that could be spilled into later.
    end_flag = np.zeros((nrows, ncols), dtype=np.uint8)


    # find all nonzero stream_id values in the stream_id array
    # Numba-safe unique nonzero stream-id collection (no boolean-mask np.unique).
    unique_stream_ids = np.empty(max_q, dtype=np.int32)
    stream_count = 0
    for rr in range(nrows):
        for cc in range(ncols):
            sid0 = stream_id[rr, cc]
            if sid0 <= 0:
                continue
            seen = False
            for k in range(stream_count):
                if unique_stream_ids[k] == sid0:
                    seen = True
                    break
            if not seen:
                unique_stream_ids[stream_count] = sid0
                stream_count += 1
    unique_stream_ids = unique_stream_ids[:stream_count]

    # Build a catalog of stream cells for each unique stream_id.
    # Catalog is keyed by index in unique_stream_ids:
    # [catalog_start[i], catalog_start[i+1]) gives positions in catalog_r/catalog_c.
    stream_cell_counts = np.zeros(stream_count, dtype=np.int32)
    for rr in range(nrows):
        for cc in range(ncols):
            sid0 = stream_id[rr, cc]
            if sid0 <= 0:
                continue
            for i in range(stream_count):
                if unique_stream_ids[i] == sid0:
                    stream_cell_counts[i] += 1
                    break

    catalog_start = np.zeros(stream_count + 1, dtype=np.int32)
    for i in range(stream_count):
        catalog_start[i + 1] = catalog_start[i] + stream_cell_counts[i]

    total_stream_cells = catalog_start[stream_count]
    catalog_r = np.empty(total_stream_cells, dtype=np.int32)
    catalog_c = np.empty(total_stream_cells, dtype=np.int32)
    write_ptr = np.empty(stream_count, dtype=np.int32)
    for i in range(stream_count):
        write_ptr[i] = catalog_start[i]

    for rr in range(nrows):
        for cc in range(ncols):
            sid0 = stream_id[rr, cc]
            if sid0 <= 0:
                continue
            for i in range(stream_count):
                if unique_stream_ids[i] == sid0:
                    p = write_ptr[i]
                    catalog_r[p] = rr
                    catalog_c[p] = cc
                    write_ptr[i] = p + 1
                    break
    
    # Per-stream workspace so BFS can expand from all seeds of a stream
    # before any cross-stream merge.
    stream_wse = np.full((nrows, ncols), np.nan, dtype=np.float32)

    # begin loop over stream ids in order of average WSE (lowest first)
    for stream_pos in range(unique_stream_ids.shape[0]):

        # create an empty array for this segments WSE and that will be blended at the end
        WSE_Out_stream = np.full((nrows, ncols), np.nan, dtype=np.float32)

        # reset per-stream workspace
        for rr in range(nrows):
            for cc in range(ncols):
                stream_wse[rr, cc] = np.nan

        sid = unique_stream_ids[stream_pos]
        sid_idx = -1
        for i in range(stream_count):
            if unique_stream_ids[i] == sid:
                sid_idx = i
                break
        if sid_idx < 0:
            continue
        
        # These are the start and end locations of the stream segment
        p0 = catalog_start[sid_idx]
        p1 = catalog_start[sid_idx + 1]

        # Build per-stream snapped seed catalog first (before excl path logic).
        # Each valid WSE_Initial stream cell is snapped to the nearest cell
        # with FAC >= flowacc_stream_threshold.
        seg_len = p1 - p0
        seed_r = np.empty(seg_len, dtype=np.int32)
        seed_c = np.empty(seg_len, dtype=np.int32)
        seed_wse = np.empty(seg_len, dtype=np.float32)
        seed_count = 0

        for p in range(p0, p1):
            sr0 = catalog_r[p]
            sc0 = catalog_c[p]
            wse0 = WSE_Initial[sr0, sc0]
            if np.isnan(wse0) or wse0 <= -9998.0 or wse0 <= E[sr0, sc0]:
                continue
            seed_r[seed_count] = sr0
            seed_c[seed_count] = sc0
            seed_wse[seed_count] = wse0
            seed_count += 1

        # Process the seed catalog for this stream.
        for i in range(seed_count):
            sr = seed_r[i]
            sc = seed_c[i]
            wse = seed_wse[i]
            depth = wse - E[sr, sc]

            if np.isnan(stream_wse[sr, sc]) or wse > stream_wse[sr, sc]:
                stream_wse[sr, sc] = wse

            # Start a new seed traversal for this stream cell.
            seed_id += 1
            q_read = 0
            q_write = 0
            q_r[q_write] = sr
            q_c[q_write] = sc
            q_write += 1
            visit_id[sr, sc] = seed_id

            # Breadth-first search (BFS) over reverse-flow neighbors.
            while q_read < q_write:
                r = q_r[q_read]
                c = q_c[q_read]
                q_read += 1

                # Whitebox D8 pointer encoding:
                # 1=NE, 2=E, 4=SE, 8=S, 16=SW, 32=W, 64=NW, 128=N
                if r > 0 and flowdir[r - 1, c] == 8:
                    nr = r - 1
                    nc = c
                    if visit_id[nr, nc] != seed_id and E[nr, nc] > -9998.0 and (wse - E[nr, nc]) > 0.1:
                        visit_id[nr, nc] = seed_id
                        q_r[q_write] = nr
                        q_c[q_write] = nc
                        q_write += 1
                        if np.isnan(stream_wse[nr, nc]) or wse > stream_wse[nr, nc]:
                            stream_wse[nr, nc] = wse
                            end_flag[nr, nc] = 0
                if r < nrows - 1 and flowdir[r + 1, c] == 128:
                    nr = r + 1
                    nc = c
                    if visit_id[nr, nc] != seed_id and E[nr, nc] > -9998.0 and (wse - E[nr, nc]) > 0.1:
                        visit_id[nr, nc] = seed_id
                        q_r[q_write] = nr
                        q_c[q_write] = nc
                        q_write += 1
                        if np.isnan(stream_wse[nr, nc]) or wse > stream_wse[nr, nc]:
                            stream_wse[nr, nc] = wse
                            end_flag[nr, nc] = 0
                if c > 0 and flowdir[r, c - 1] == 2:
                    nr = r
                    nc = c - 1
                    if visit_id[nr, nc] != seed_id and E[nr, nc] > -9998.0 and (wse - E[nr, nc]) > 0.1:
                        visit_id[nr, nc] = seed_id
                        q_r[q_write] = nr
                        q_c[q_write] = nc
                        q_write += 1
                        if np.isnan(stream_wse[nr, nc]) or wse > stream_wse[nr, nc]:
                            stream_wse[nr, nc] = wse
                            end_flag[nr, nc] = 0
                if c < ncols - 1 and flowdir[r, c + 1] == 32:
                    nr = r
                    nc = c + 1
                    if visit_id[nr, nc] != seed_id and E[nr, nc] > -9998.0 and (wse - E[nr, nc]) > 0.1:
                        visit_id[nr, nc] = seed_id
                        q_r[q_write] = nr
                        q_c[q_write] = nc
                        q_write += 1
                        if np.isnan(stream_wse[nr, nc]) or wse > stream_wse[nr, nc]:
                            stream_wse[nr, nc] = wse
                            end_flag[nr, nc] = 0
                if r > 0 and c > 0 and flowdir[r - 1, c - 1] == 4:
                    nr = r - 1
                    nc = c - 1
                    if visit_id[nr, nc] != seed_id and E[nr, nc] > -9998.0 and (wse - E[nr, nc]) > 0.1:
                        visit_id[nr, nc] = seed_id
                        q_r[q_write] = nr
                        q_c[q_write] = nc
                        q_write += 1
                        if np.isnan(stream_wse[nr, nc]) or wse > stream_wse[nr, nc]:
                            stream_wse[nr, nc] = wse
                            end_flag[nr, nc] = 0
                if r > 0 and c < ncols - 1 and flowdir[r - 1, c + 1] == 16:
                    nr = r - 1
                    nc = c + 1
                    if visit_id[nr, nc] != seed_id and E[nr, nc] > -9998.0 and (wse - E[nr, nc]) > 0.1:
                        visit_id[nr, nc] = seed_id
                        q_r[q_write] = nr
                        q_c[q_write] = nc
                        q_write += 1
                        if np.isnan(stream_wse[nr, nc]) or wse > stream_wse[nr, nc]:
                            stream_wse[nr, nc] = wse
                            end_flag[nr, nc] = 0
                if r < nrows - 1 and c > 0 and flowdir[r + 1, c - 1] == 1:
                    nr = r + 1
                    nc = c - 1
                    if visit_id[nr, nc] != seed_id and E[nr, nc] > -9998.0 and (wse - E[nr, nc]) > 0.1:
                        visit_id[nr, nc] = seed_id
                        q_r[q_write] = nr
                        q_c[q_write] = nc
                        q_write += 1
                        if np.isnan(stream_wse[nr, nc]) or wse > stream_wse[nr, nc]:
                            stream_wse[nr, nc] = wse
                            end_flag[nr, nc] = 0
                if r < nrows - 1 and c < ncols - 1 and flowdir[r + 1, c + 1] == 64:
                    nr = r + 1
                    nc = c + 1
                    if visit_id[nr, nc] != seed_id and E[nr, nc] > -9998.0 and (wse - E[nr, nc]) > 0.1:
                        visit_id[nr, nc] = seed_id
                        q_r[q_write] = nr
                        q_c[q_write] = nc
                        q_write += 1
                        if np.isnan(stream_wse[nr, nc]) or wse > stream_wse[nr, nc]:
                            stream_wse[nr, nc] = wse
                            end_flag[nr, nc] = 0
        
        # # catalog the stream_wse minimum value that will be used to limit spillover
        # stream_wse_minimum = np.nanmin(stream_wse)
        # # catalog the stream_wse maximum value to make sure we have valid WSE values for the stream
        # stream_wse_maximum = np.nanmax(stream_wse)
        # if np.isnan(stream_wse_minimum) or np.isnan(stream_wse_maximum):
        #     continue
        # Final merge for this stream.
        for rr in range(nrows):
            for cc in range(ncols):
                sw = stream_wse[rr, cc]
                if np.isnan(sw):
                    continue
                if np.isnan(WSE_Out_stream[rr, cc]) or (sw < WSE_Out_stream[rr, cc] and WSE_Out_stream[rr, cc] > -9998.0):
                    WSE_Out_stream[rr, cc] = sw

        

        ### now perform spillover and backfill for this stream before moving on to the next stream id ###


        # Iterative spillover: boundary spill points -> upsteam and downstream spread using flowdir until steady-state.
        cellsize = np.float32(max(dx, dy))
        if cellsize <= 0.0:
            cellsize = np.float32(1.0)
        candidate_wse = np.full((nrows, ncols), np.nan, dtype=np.float32)
        spill_source_wse = np.full((nrows, ncols), np.nan, dtype=np.float32)
        spill_source_depth = np.full((nrows, ncols), np.float32(-1.0), dtype=np.float32)
        boundary_r = np.empty(max_q, dtype=np.int32)
        boundary_c = np.empty(max_q, dtype=np.int32)
        bq_r = np.empty(max_q, dtype=np.int32)
        bq_c = np.empty(max_q, dtype=np.int32)
        bq_inq = np.zeros((nrows, ncols), dtype=np.uint8)
        dry_nbr_count = np.zeros((nrows, ncols), dtype=np.int16)
        touched_r = np.empty(max_q, dtype=np.int32)
        touched_c = np.empty(max_q, dtype=np.int32)
        cand_r = np.empty(max_q, dtype=np.int32)
        cand_c = np.empty(max_q, dtype=np.int32)
        touched_count = 0
        cand_inq = np.zeros((nrows, ncols), dtype=np.uint8)
        newly_wet_depth = np.zeros((nrows, ncols), dtype=np.float32)
        new_r = np.empty(max_q, dtype=np.int32)
        new_c = np.empty(max_q, dtype=np.int32)
        new_inq = np.zeros((nrows, ncols), dtype=np.uint8)
        # Build initial boundary queue of dry cells adjacent to wet cells (WSE_Out < E).
        bq_read = 0
        bq_write = 0
        for r in range(1, nrows - 1):
            for c in range(1, ncols - 1):
                if np.isnan(WSE_Out_stream[r, c]) or E[r, c] < -9998.0 or (WSE_Out_stream[r, c] - E[r, c]) <= 0.1:
                    continue
                dcnt = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = r + dr
                        nc = c + dc
                        if np.isnan(WSE_Out_stream[nr, nc]) or (WSE_Out_stream[nr, nc] - E[nr, nc]) <=  0.1:
                            dcnt += 1
                dry_nbr_count[r, c] = dcnt
                if dcnt > 0 and bq_inq[r, c] == 0:
                    bq_r[bq_write] = r
                    bq_c[bq_write] = c
                    bq_inq[r, c] = 1
                    bq_write += 1

        spill_changed = True
        spill_count = 0
        loop_count = 0
        while spill_changed:
            loop_count += 1
            spill_changed = False
            new_count = 0
            cand_count = 0


            # Build boundary list from the queue of wet cells adjacent to dry cells.
            boundary_count = 0
            while bq_read < bq_write:
                r = bq_r[bq_read]
                c = bq_c[bq_read]
                bq_read += 1
                bq_inq[r, c] = 0
                if np.isnan(WSE_Out_stream[r, c]) or E[r, c] < -9998.0 or (WSE_Out_stream[r, c] - E[r, c]) <=  0.1:
                    continue
                if dry_nbr_count[r, c] > 0:
                    boundary_r[boundary_count] = r
                    boundary_c[boundary_count] = c
                    boundary_count += 1
                        
            # if boundary_count == 0, end the loop because we don't have spillover candidates.
            if boundary_count == 0:
                break
                
            # Identify spillover candidates by minimum depth from wet neighbors.
            for i in range(boundary_count):
                r = boundary_r[i]
                c = boundary_c[i]
                # the wet cell elevation
                obdy_fill = E[r, c]
                if obdy_fill <= -9998.0:
                    continue
                wet_wse = WSE_Out_stream[r, c]
                wet_depth = wet_wse - obdy_fill
                if wet_depth <= 0.0:
                    continue
                # Find the candidate dry cell that the wet cell should spill into.
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                                    (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr = r + dr
                    nc = c + dc
                    if nr < 0 or nr >= nrows or nc < 0 or nc >= ncols:
                        continue
                    if stream_id[nr, nc] > 0:
                        continue
                    # candidate cell's elevation
                    bdy_fill = E[nr, nc]
                    # if we've hit the boundary of the DEM, ignore this cell
                    if bdy_fill <= -9998.0:
                        continue
                    # if the cell is wet, ignore it
                    if WSE_Out_stream[nr, nc] > bdy_fill:
                        continue
                    # Preserve the wet-cell depth and reduce it only when the
                    # spill candidate is at a higher elevation.
                    candidate_depth = wet_depth
                    delta_elevation = obdy_fill - bdy_fill
                    if delta_elevation < 0.0:
                        candidate_depth = candidate_depth + delta_elevation
                    if candidate_depth <= 0.0:
                        continue
                    # Keep a sparse list of newly touched dry cells and retain
                    # the minimum passing depth from all wet neighbors.
                    if cand_inq[nr, nc] == 0:
                        cand_inq[nr, nc] = 1
                        cand_r[cand_count] = nr
                        cand_c[cand_count] = nc
                        newly_wet_depth[nr, nc] = candidate_depth
                        cand_count += 1
                    elif candidate_depth < newly_wet_depth[nr, nc]:
                        newly_wet_depth[nr, nc] = candidate_depth

            # Convert sparse candidate list to touched candidate arrays.
            touched_count = cand_count
            for i in range(cand_count):
                rr = cand_r[i]
                cc = cand_c[i]
                d = newly_wet_depth[rr, cc]
                touched_r[i] = rr
                touched_c[i] = cc
                candidate_wse[rr, cc] = E[rr, cc] + d
                # reset sparse bookkeeping for the next spill iteration
                newly_wet_depth[rr, cc] = np.float32(0.0)
                cand_inq[rr, cc] = 0

            # Sort spillover locations by increasing WSE (NaNs last) using argsort.
            cand_wse = np.empty(cand_count, dtype=np.float32)
            for i in range(cand_count):
                rr = touched_r[i]
                cc = touched_c[i]
                wse = candidate_wse[rr, cc]
                if np.isnan(wse):
                    wse = np.float32(1.0e30)
                cand_wse[i] = wse
            cand_idx = np.argsort(cand_wse).astype(np.int32)

            # Process spillover candidates and backfill immediately for each new wet cell.
            for i in range(cand_count):
                r = touched_r[cand_idx[i]]
                c = touched_c[cand_idx[i]]
                source_wse = candidate_wse[r, c]
                source_depth = candidate_wse[r, c] - E[r, c]
                if source_depth <= 0.0:
                    continue
                # Write candidate spill only if it actually improves this cell.
                wrote = False
                if np.isnan(WSE_Out_stream[r, c]):
                    WSE_Out_stream[r, c] = candidate_wse[r, c]
                    wrote = True
                elif source_wse < WSE_Out_stream[r, c] and source_wse > E[r, c]:
                    WSE_Out_stream[r, c] = candidate_wse[r, c]
                    wrote = True
                if not wrote:
                    continue
                if new_inq[r, c] == 0:
                    new_r[new_count] = r
                    new_c[new_count] = c
                    new_inq[r, c] = 1
                    new_count += 1
                spill_source_wse[r, c] = source_wse
                spill_source_depth[r, c] = source_depth
                spill_changed = True

                # Seed backfill stack with the current spill source (and any downstream updates).
                back_write = 0
                q_r[back_write] = r
                q_c[back_write] = c
                back_write += 1

                # Flow downstream (using flowdir) until encountering a wet cell, stream location, or dead end.
                rr = r
                cc = c
                # initial WSE for spillover
                source_wse = spill_source_wse[r, c]
                # intial depth for spilloever
                depth_use = spill_source_wse[r, c] - E[r, c]
                # initial elevation
                previous_elev = E[r, c]
                steps = 0
                while steps < max_q:
                    fd = flowdir[rr, cc]
                    if fd <= 0:
                        break
                    if fd == 1:
                        dr, dc = -1, 1
                    elif fd == 2:
                        dr, dc = 0, 1
                    elif fd == 4:
                        dr, dc = 1, 1
                    elif fd == 8:
                        dr, dc = 1, 0
                    elif fd == 16:
                        dr, dc = 1, -1
                    elif fd == 32:
                        dr, dc = 0, -1
                    elif fd == 64:
                        dr, dc = -1, -1
                    else:
                        dr, dc = -1, 0
                    rr = rr + dr
                    cc = cc + dc
                    if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols:
                        break
                    # if the cell is a stream cell, stop routing
                    if stream_id[rr, cc] > 0:
                        break
                    # if we are at the edge of the DEM stop routing
                    if E[rr, cc] <= -9998.0:
                        break
                    # Recompute spill depth at each downstream step from the
                    # upstream cell's current WSE and the destination elevation.
                    if (source_wse - E[rr, cc]) <= 0.1:
                        break
                    # delta_elevation is almost like a rough head loss term
                    delta_elevation = previous_elev - E[rr, cc]
                    if delta_elevation < 0.0:
                        depth_use = depth_use + delta_elevation
                    if depth_use <= 0.0:
                        break
                    new_wse = E[rr, cc] + depth_use
                    wrote = False
                    # if the cell is dry go ahead and flood it
                    if np.isnan(WSE_Out_stream[rr, cc]):
                        WSE_Out_stream[rr, cc] = new_wse
                        wrote = True
                        if new_inq[rr, cc] == 0:
                            new_r[new_count] = rr
                            new_c[new_count] = cc
                            new_inq[rr, cc] = 1
                            source_wse = new_wse
                            previous_elev = E[rr, cc]
                            new_count += 1
                        spill_source_wse[rr, cc] = new_wse
                    # if the cell is wet stop routing
                    if (not np.isnan(WSE_Out_stream[rr, cc])):
                        break
                    if not wrote:
                        break
                    spill_source_depth[rr, cc] = depth_use
                    spill_changed = True
                    q_r[back_write] = rr
                    q_c[back_write] = cc
                    back_write += 1
                    steps += 1

                    # Backfill upstream (reverse flowdir) immediately from newly flooded spillover cell.
                    while back_write > 0:
                        back_write -= 1
                        ur = q_r[back_write]
                        uc = q_c[back_write]
                        source_depth = spill_source_depth[ur, uc]
                        if source_depth <= 0.0:
                            continue
                        source_wse = E[ur, uc] + source_depth
                        # Stop upstream routing when we re-enter an already wet pixel. Replace the WSE if the backfill WSE is lower than the existing WSE.
                        if (not np.isnan(WSE_Out_stream[ur, uc])):
                            if source_wse < WSE_Out_stream[ur, uc] and source_wse > E[ur, uc]:
                                WSE_Out_stream[ur, uc] = source_wse
                            else:
                                continue

                        # 1=NE, 2=E, 4=SE, 8=S, 16=SW, 32=W, 64=NW, 128=N
                        for dr, dc, fd_in in [(-1, 0, 8), (1, 0, 128), (0, -1, 2), (0, 1, 32),
                                            (-1, -1, 4), (-1, 1, 16), (1, -1, 1), (1, 1, 64)]:
                            nr = ur + dr
                            nc = uc + dc
                            # if we we are out of bounds in the domain stop routing upstream
                            if nr < 0 or nr >= nrows or nc < 0 or nc >= ncols:
                                continue
                            # if backfill encounters a stream cell that isn't the current stream, pass it
                            if stream_id[nr, nc] > 0:
                                continue
                            # if we we are out of bounds in the domain stop routing upstream
                            if E[nr, nc] <= -9998.0:
                                continue
                            # make sure the water is routing upstream, not downstream
                            if flowdir[nr, nc] != fd_in:
                                continue
                            # if the cell is already wet, continue to the next location
                            if (not np.isnan(WSE_Out_stream[nr, nc])):
                                continue
                            new_wse = E[nr, nc] + source_depth
                            # if the cell is dry and the depth is above ground, lets make it wet now 
                            if np.isnan(WSE_Out_stream[nr, nc]) and (new_wse - E[nr, nc]) > 0.1:
                                WSE_Out_stream[nr, nc] = new_wse
                                wrote_up = True
                                if new_inq[nr, nc] == 0:
                                    new_r[new_count] = nr
                                    new_c[new_count] = nc
                                    new_inq[nr, nc] = 1
                                    new_count += 1
                            spill_source_wse[nr, nc] = new_wse
                            spill_source_depth[nr, nc] = source_depth
                            spill_changed = True
                            q_r[back_write] = nr
                            q_c[back_write] = nc
                            back_write += 1
                    



            # Update boundary queue from newly wet cells using local dry-neighbor
            # count refreshes; only the 3x3 neighborhood around each new wet cell
            # can change frontier status.
            for i in range(new_count):
                r = new_r[i]
                c = new_c[i]

                for dr0 in (-1, 0, 1):
                    for dc0 in (-1, 0, 1):
                        rr = r + dr0
                        cc = c + dc0
                        if rr < 1 or rr >= nrows - 1 or cc < 1 or cc >= ncols - 1:
                            continue
                        if np.isnan(WSE_Out_stream[rr, cc]) or WSE_Out_stream[rr, cc] <= E[rr, cc]:
                            dry_nbr_count[rr, cc] = 0
                            continue
                        dcnt = 0
                        for dr1 in (-1, 0, 1):
                            for dc1 in (-1, 0, 1):
                                if dr1 == 0 and dc1 == 0:
                                    continue
                                nr = rr + dr1
                                nc = cc + dc1
                                if np.isnan(WSE_Out_stream[nr, nc]) or WSE_Out_stream[nr, nc] <= E[nr, nc]:
                                    dcnt += 1
                        dry_nbr_count[rr, cc] = dcnt
                        if dcnt > 0 and bq_inq[rr, cc] == 0:
                            bq_r[bq_write] = rr
                            bq_c[bq_write] = cc
                            bq_inq[rr, cc] = 1
                            bq_write += 1
                new_inq[r, c] = 0


        # add the WSE_Out_stream to WSE_Out taking the minimum value when WSE_Out values are not NaNs.
        for rr in range(nrows):
            for cc in range(ncols):
                ws = WSE_Out_stream[rr, cc]
                if np.isnan(ws) or ws <= 0.0 or ws <= -9998.0:
                    continue
                if np.isnan(WSE_Out[rr, cc]):
                    WSE_Out[rr, cc] = ws
                else:
                    if ws < WSE_Out[rr, cc]:
                        WSE_Out[rr, cc] = ws

    return WSE_Out

@njit(cache=True, parallel=True)
def CreateSimpleFloodMapParallel(RR, CC, T_Rast, W_Rast, S_Rast, 
                                 E, B, flowdir, 
                                 nrows, ncols, search_dist_for_min_elev, 
                                 TopWidthMax, dx, dy, LocalFloodOption, 
                                 COMID_Unique_TW, COMID_Unique_Depth, 
                                 WeightBox, 
                                 TW_for_WeightBox_ElipseMask, TopWidthPlausibleLimit, 
                                 Set_Depth, flood_vdt_cells, OutDEP,
                                 mapper):
    """
    This function uses parallelization to quickly generate floodmaps. Small differences can occur, infrequently, due to race conditions
    when adding weighted values to the WSE_Times_Weight and Slope_Times_Weight array slices. But, because of the order of the stream cells, 
    these differences are very small and infrequent, and relatively inconsequential to the overall flood inundation results.
    """

    COMID_Averaging_Method = 0

    # these are for the weighted approach but have to be made either way to keep Numba's shape inference consistent
    WSE_Times_Weight = np.zeros((nrows+2,ncols+2), dtype=np.float32)
    # Always allocate to keep Numba's shape inference consistent.
    Slope_Times_Weight = np.zeros((nrows+2,ncols+2), dtype=np.float32)
    WSE_array = np.empty_like(WSE_Times_Weight, dtype=np.float32)

    Total_Weight = np.zeros((nrows+2,ncols+2), dtype=np.float32)
    
    if mapper == "Curve2Flood-FLDPLNpy":

        # if using Set_Depth, we want to seed all stream cells with WSE = E + Set_Depth, and then let the FLDPLN model expand from there.
        if Set_Depth > 0.0:
            W_Rast_Padded = np.full((nrows + 2, ncols + 2), np.nan, dtype=np.float32)
            for r in range(nrows):
                for c in range(ncols):
                    # B and E are already padded in this scope.
                    if B[r + 1, c + 1] > 0:
                        W_Rast_Padded[r + 1, c + 1] = E[r + 1, c + 1] + Set_Depth
                    else:
                        W_Rast_Padded[r + 1, c + 1] = np.nan

        else:
            # pad the W_Rast to make it match E and the other arrays that are all nrows+2 by ncols+2
            W_Rast_Padded = np.full((nrows + 2, ncols + 2), np.nan, dtype=np.float32)
            for r in range(nrows):
                for c in range(ncols):
                    if B[r + 1, c + 1] > 0:
                        W_Rast_Padded[r + 1, c + 1] = W_Rast[r, c]
                    else:
                        W_Rast_Padded[r + 1, c + 1] = np.nan

        # This spreads WSE outward from currently-flooded cells to dry cells as long as
        # neighboring ground elevation E can be overtopped.
        WSE_array = fldpln(
                            W_Rast_Padded, E, flowdir, B, nrows+2, ncols+2, dx, dy
        )
    else:

        #Now go through each cell
        num_nonzero = len(RR)
        number_filtered = 0
        delta_wse_threshold_m = np.float32(1.0)
        large_delta_patch_area_threshold_ratio = np.float32(0.30)
        for i in range(num_nonzero):
            r = RR[i]
            c = CC[i]
            r_use = r
            c_use = c
            E_Min = E[r,c]
            
            COMID_Value = B[r,c]
            if Set_Depth>0.0:
                WSE = float(E[r_use,c_use] + Set_Depth)
                if S_Rast is not None:
                    SLOPE = float(S_Rast[r_use,c_use])
                COMID_TW_m = TopWidthPlausibleLimit
            elif COMID_Averaging_Method!=0 or W_Rast[r-1,c-1]<0.001 or T_Rast[r-1,c-1]<0.00001:
                #Get COMID, TopWidth, and Depth Information for this cell
                COMID_Value = B[r,c]
                # keys are int32, values are float32
                if COMID_Value in COMID_Unique_TW:
                    COMID_TW_m = COMID_Unique_TW[COMID_Value]
                else:
                    COMID_TW_m = np.float32(0.0)

                if COMID_Value in COMID_Unique_Depth:
                    COMID_D = COMID_Unique_Depth[COMID_Value]
                else:
                    COMID_D = np.float32(0.0)
                WSE = float(E[r_use,c_use] + COMID_D)
                if S_Rast is not None:
                    SLOPE = float(S_Rast[r_use,c_use])
            else:
                #These are Based on the AutoRoute/ARC Results, not averaged for COMID
                WSE = np.round(W_Rast[r-1,c-1], 2)  #Have to have the '-1' because of the Row and Col being inset on the B raster.
                COMID_TW_m = T_Rast[r-1,c-1]
                if S_Rast is not None:
                    SLOPE = S_Rast[r-1,c-1]            

            if WSE < 0.001 or COMID_TW_m < 0.00001 or (WSE - E[r,c]) < 0.001:
                continue

            # give the TW for the weightbox the median if its smaller than the median.
            if COMID_TW_m > TopWidthMax:
                COMID_TW_m = TopWidthMax

            #This is how many cells we will be looking at surrounding our stream cell
            COMID_TW = int(max(np.round(COMID_TW_m / dx), np.round(COMID_TW_m / dy)))

            
            # Find minimum elevation within the search box
            if search_dist_for_min_elev >= 1:
                for rr in range(max(r - search_dist_for_min_elev, 0), min(r + search_dist_for_min_elev + 1, nrows - 1)):
                    for cc in range(max(c - search_dist_for_min_elev, 1), min(c + search_dist_for_min_elev + 1, ncols - 1)):
                        if E[rr,cc] > 0.1 and E[rr,cc] < E_Min:
                            E_Min = E[rr,cc]
                            r_use = rr
                            c_use = cc

            r_min = max(r_use - COMID_TW, 1)
            r_max = min(r_use + COMID_TW + 1, nrows + 1)
            c_min = max(c_use - COMID_TW, 1)
            c_max = min(c_use + COMID_TW + 1, ncols + 1)
            
            # This uses the weighting method from FloodSpreader to create a flood map
            # Here we use TW instead of COMID_TW.  This is because we are trying to find the center of the weight raster, which was set based on TW (not COMID_TW).  
            # COMID_TW mainly applies to the r_min, r_max, c_min, c_max
            w_r_min = TW_for_WeightBox_ElipseMask - (r_use - r_min)
            w_r_max = TW_for_WeightBox_ElipseMask + (r_max - r_use)
            w_c_min = TW_for_WeightBox_ElipseMask - (c_use - c_min)
            w_c_max = TW_for_WeightBox_ElipseMask + (c_max - c_use)

            weight_slice = WeightBox[w_r_min:w_r_max, w_c_min:w_c_max]
            boundary_has_flooded_too_much = False
            if LocalFloodOption:
                #Find what would flood local
                E_Box = E[r_min:r_max,c_min:c_max]
                FloodLocalMask = FloodAllLocalAreas(WSE, E_Box, r_min, r_max, c_min, c_max, r_use, c_use)
                WSE_Times_Weight[r_min:r_max, c_min:c_max] += (WSE * weight_slice * FloodLocalMask)
                Total_Weight[r_min:r_max,c_min:c_max] += weight_slice * FloodLocalMask
                if S_Rast is not None:
                    Slope_Times_Weight[r_min:r_max, c_min:c_max] += (SLOPE * weight_slice * FloodLocalMask)
            else:

                # This puts the weights from each cell into the composite arrays.
                cell_wse_weight = WSE * weight_slice                
                WSE_Times_Weight[r_min:r_max, c_min:c_max] += cell_wse_weight
                Total_Weight[r_min:r_max,c_min:c_max] += weight_slice

                # This makes a weighted slope raster that can be used for velocity estimates
                if S_Rast is not None:
                    Slope_Times_Weight[r_min:r_max, c_min:c_max] += (SLOPE * weight_slice)

        # These are the cells that we want to flood based on the weighted WSE being greater than the elevation, and also making sure the elevation is valid and that we have some weight there.
        valid_candidate = (
            (E > -9998.0) &
            (WSE_Times_Weight > E * Total_Weight)  # Keeps the values where WSE is greater than E, which means it would be flooded
        )

        # existing_wse = np.where(valid_existing, num / den, -1.0e30)
        WSE_array = np.where(valid_candidate, WSE_Times_Weight / Total_Weight, np.nan).astype(np.float32)


    Flooded_array = np.where((WSE_array > E) & (E > -9998.0), 1, 0).astype(np.uint8)

    # Also make sure all the Cells that have Stream are counted as flooded.
    if flood_vdt_cells:
        for i in range(len(RR)):
            Flooded_array[RR[i],CC[i]] = 1

    # Create the Depth array
    if OutDEP:
        Depth_array = np.where((WSE_array > E) & (E > -9998.0), WSE_array - E, np.nan).astype(np.float32)
    else:
        Depth_array = np.empty((3, 3), dtype=np.float32) # Dummy array if not used

    # if you want, create the slope array
    if S_Rast is not None:
        Slope_divided_by_weight = Slope_Times_Weight / Total_Weight
        Slope_array = np.where((WSE_array > E) & (E > -9998.0), Slope_divided_by_weight, np.nan).astype(np.float32)
        Slope_array = np.where((Slope_array <= 0), 0.0002, Slope_array).astype(np.float32)
        return Flooded_array[1:-1, 1:-1], Depth_array[1:-1, 1:-1], Slope_array[1:-1, 1:-1]


    return Flooded_array[1:-1, 1:-1], Depth_array[1:-1, 1:-1], None

@njit(cache=True)
def detect_blocky_contribution(
    WSE_Times_Weight,   # full (nrows+2, ncols+2) accumulator array, AFTER adding this contribution
    Total_Weight,       # full (nrows+2, ncols+2) weight array, AFTER adding this contribution
    E,                  # full (nrows+2, ncols+2) elevation array
    prev_wse_slice,     # copy of WSE_Times_Weight[r_min:r_max, c_min:c_max] BEFORE this contribution
    prev_total_slice,   # copy of Total_Weight[r_min:r_max, c_min:c_max] BEFORE this contribution
    r_min, r_max,       # row bounds (into the padded arrays)
    c_min, c_max,       # col bounds (into the padded arrays)
    max_run_threshold   # minimum straight boundary run to flag as a wall (e.g. COMID_TW // 2)
):
    """
    Detects whether a single stream cell's contribution to WSE_Times_Weight produces
    an unnaturally blocky (rectangular) flooded footprint.

    Returns a tuple of scalar metrics:
        (compactness, min_edge_frac, max_edge_frac, new_flooded_cells, max_run, is_blocky)

    All inputs use the padded (nrows+2, ncols+2) coordinate space, matching the
    convention used throughout CreateSimpleFloodMap / CreateSimpleFloodMapParallel.

    Compatible with @njit — no scipy, no dicts, no list comprehensions.
    """

    n_r = r_max - r_min
    n_c = c_max - c_min

    # ------------------------------------------------------------------ #
    # 1. Build boolean flood masks for current and previous state          #
    # ------------------------------------------------------------------ #
    new_area  = np.int32(0)
    row_min_f = np.int32(n_r)
    row_max_f = np.int32(0)
    col_min_f = np.int32(n_c)
    col_max_f = np.int32(0)

    flooded     = np.zeros((n_r, n_c), dtype=np.uint8)
    was_flooded = np.zeros((n_r, n_c), dtype=np.uint8)
    new_flood   = np.zeros((n_r, n_c), dtype=np.uint8)

    for lr in range(n_r):
        gr = r_min + lr
        for lc in range(n_c):
            gc = c_min + lc
            tw_now  = Total_Weight[gr, gc]
            tw_prev = prev_total_slice[lr, lc]
            elev    = E[gr, gc]
            if elev <= -9998.0:
                continue
            if tw_now > 0.0 and WSE_Times_Weight[gr, gc] > elev * tw_now:
                flooded[lr, lc] = np.uint8(1)
            if tw_prev > 0.0 and prev_wse_slice[lr, lc] > elev * tw_prev:
                was_flooded[lr, lc] = np.uint8(1)
            if flooded[lr, lc] and not was_flooded[lr, lc]:
                new_flood[lr, lc] = np.uint8(1)
                new_area += np.int32(1)
                if lr < row_min_f: row_min_f = np.int32(lr)
                if lr > row_max_f: row_max_f = np.int32(lr)
                if lc < col_min_f: col_min_f = np.int32(lc)
                if lc > col_max_f: col_max_f = np.int32(lc)

    if new_area == 0:
        return (np.float32(0.0), np.float32(0.0), np.float32(0.0), np.int32(0), np.int32(0), np.uint8(0))

    # ------------------------------------------------------------------ #
    # 2. Metric 1 — Compactness (bounding-box fill ratio)                 #
    # ------------------------------------------------------------------ #
    bbox_area   = (row_max_f - row_min_f + np.int32(1)) * (col_max_f - col_min_f + np.int32(1))
    compactness = np.float32(new_area) / np.float32(bbox_area)

    # ------------------------------------------------------------------ #
    # 3. Metric 2 — Per-edge flood fractions                              #
    # ------------------------------------------------------------------ #
    top_total  = np.int32(0); top_flooded  = np.int32(0)
    bot_total  = np.int32(0); bot_flooded  = np.int32(0)
    left_total = np.int32(0); left_flooded = np.int32(0)
    right_total= np.int32(0); right_flooded= np.int32(0)

    for lc in range(n_c):
        gc = c_min + lc
        if E[r_min,     gc] > -9998.0 and Total_Weight[r_min,     gc] > 0.0:
            top_total += np.int32(1)
            if flooded[0,       lc]: top_flooded += np.int32(1)
        if E[r_max - 1, gc] > -9998.0 and Total_Weight[r_max - 1, gc] > 0.0:
            bot_total += np.int32(1)
            if flooded[n_r - 1, lc]: bot_flooded += np.int32(1)

    for lr in range(1, n_r - 1):
        gr = r_min + lr
        if E[gr, c_min    ] > -9998.0 and Total_Weight[gr, c_min    ] > 0.0:
            left_total += np.int32(1)
            if flooded[lr, 0      ]: left_flooded  += np.int32(1)
        if E[gr, c_max - 1] > -9998.0 and Total_Weight[gr, c_max - 1] > 0.0:
            right_total += np.int32(1)
            if flooded[lr, n_c - 1]: right_flooded += np.int32(1)

    top_frac   = np.float32(top_flooded)  / np.float32(max(top_total,   1))
    bot_frac   = np.float32(bot_flooded)  / np.float32(max(bot_total,   1))
    left_frac  = np.float32(left_flooded) / np.float32(max(left_total,  1))
    right_frac = np.float32(right_flooded)/ np.float32(max(right_total, 1))

    min_edge_frac = min(top_frac, min(bot_frac, min(left_frac, right_frac)))
    max_edge_frac = max(top_frac, max(bot_frac, max(left_frac, right_frac)))

    edge_uniformity  = np.float32(1.0) - (max_edge_frac - min_edge_frac)


    # ------------------------------------------------------------------ #
    # 4. Metric 3 — Longest straight boundary run (straight wall check)   #
    #    Reuses new_flood. A boundary cell has at least one dry cardinal   #
    #    neighbor. Long collinear runs of boundary cells = rectangular     #
    #    stamp edge rather than terrain-following flood boundary.          #
    # ------------------------------------------------------------------ #
    boundary = np.zeros((n_r, n_c), dtype=np.uint8)
    for r in range(n_r):
        for c in range(n_c):
            if not new_flood[r, c]:
                continue
            if (r == 0        or not new_flood[r - 1, c] or
                r == n_r - 1  or not new_flood[r + 1, c] or
                c == 0        or not new_flood[r, c - 1] or
                c == n_c - 1  or not new_flood[r, c + 1]):
                boundary[r, c] = np.uint8(1)

    max_run = np.int32(0)

    for r in range(n_r):                        # horizontal runs
        run = np.int32(0)
        for c in range(n_c):
            if boundary[r, c]: run += np.int32(1)
            else:               run  = np.int32(0)
            if run > max_run:   max_run = run

    for c in range(n_c):                        # vertical runs
        run = np.int32(0)
        for r in range(n_r):
            if boundary[r, c]: run += np.int32(1)
            else:               run  = np.int32(0)
            if run > max_run:   max_run = run

    for start in range(n_r + n_c - 1):         # diagonal runs (top-left → bottom-right)
        r = start if start < n_r else 0
        c = 0     if start < n_r else start - n_r + 1
        run = np.int32(0)
        while r < n_r and c < n_c:
            if boundary[r, c]: run += np.int32(1)
            else:               run  = np.int32(0)
            if run > max_run:   max_run = run
            r += 1; c += 1

    for start in range(n_r + n_c - 1):         # anti-diagonal runs (top-right → bottom-left)
        r = start if start < n_r else 0
        c = n_c - 1 if start < n_r else n_c - 1 - (start - n_r + 1)
        run = np.int32(0)
        while r < n_r and c >= 0:
            if boundary[r, c]: run += np.int32(1)
            else:               run  = np.int32(0)
            if run > max_run:   max_run = run
            r += 1; c -= 1

    box_diagonal = np.float32(np.sqrt(np.float64(n_r * n_r + n_c * n_c)))
    max_run_fraction = np.float32(max_run) / box_diagonal 

    # ------------------------------------------------------------------ #
    # 5. Combined blockiness decision                                      #
    # ------------------------------------------------------------------ #
    compactness_weight = np.float32(0.2)
    edge_uniformity_weight = np.float32(0.3)
    max_run_weight = np.float32(0.5)
    blockiness_composite_index = compactness * compactness_weight + edge_uniformity * edge_uniformity_weight + max_run_fraction * max_run_weight
    is_blocky = np.uint8(
        blockiness_composite_index > np.float32(0.50)
    )

    return (compactness, min_edge_frac, max_edge_frac, new_area, max_run, is_blocky)

@njit(cache=True)
def remove_depth_outliers_from_slice(WSE_Times_Weight, Total_Weight, E,
                                     r_min, r_max, c_min, c_max,
                                     source_depth,
                                     n_sigma=np.float32(2.0)):
    """
    Removes outlier cells from the composite WSE and weight accumulators by
    zeroing out any cell whose depth (WSE - E) falls outside the range:
        [source_depth - n_sigma * MAD_sigma,
         source_depth + n_sigma * MAD_sigma]

    The reference depth is the stream cell's own WSE - E rather than the
    median of the contribution box. This is more physically meaningful —
    it represents what the hydraulic model actually predicted at the source,
    and cannot be contaminated by a large population of erroneous cells in
    the surrounding box the way a box-median could be.

    MAD-based sigma is still used for the spread estimate because the
    standard deviation of depths within the box remains vulnerable to
    inflation by the outliers being removed.

    Parameters
    ----------
    WSE_Times_Weight : 2D float32 array (nrows+2, ncols+2)
        Running sum of WSE * weight contributions — modified in place.
    Total_Weight : 2D float32 array (nrows+2, ncols+2)
        Running sum of weights — modified in place.
    E : 2D float32 array (nrows+2, ncols+2)
        Ground elevation raster (padded).
    r_min, r_max, c_min, c_max : int
        Row/col bounds of the contribution box in padded array coordinates.
    source_depth : float32
        Depth at the stream cell itself: WSE - E[r_use, c_use].
        Used as the center of the acceptance window.
    n_sigma : float32
        Number of MAD-sigmas from source_depth to accept. Default 2.0.
    """

    # --- Collect depths to compute MAD-based spread ---
    # We still need a spread estimate to set the window width, but the
    # CENTER of the window is now fixed at source_depth rather than
    # derived from the box contents.
    n_r = r_max - r_min
    n_c = c_max - c_min
    max_cells = n_r * n_c
    depths = np.empty(max_cells, dtype=np.float32)
    n = np.int32(0)

    for lr in range(n_r):
        gr = r_min + lr
        for lc in range(n_c):
            gc = c_min + lc
            elev = E[gr, gc]
            tw   = Total_Weight[gr, gc]
            if elev <= -9998.0 or tw <= 0.0:
                continue
            wse   = WSE_Times_Weight[gr, gc] / tw
            depth = wse - elev
            if depth <= 0.0:
                continue
            depths[n] = depth
            n += np.int32(1)

    # If too few cells to estimate spread, fall back to accepting everything —
    # better to leave a small contribution unchanged than remove valid cells
    # based on an unstable spread estimate.
    if n < 4:
        return

    # --- Compute MAD of box depths around source_depth ---
    # Absolute deviations are taken from source_depth (not box median),
    # so the spread estimate reflects how far cells stray from the
    # hydraulically-predicted source depth specifically.
    abs_devs = np.empty(n, dtype=np.float32)
    for i in range(n):
        abs_devs[i] = np.abs(depths[i] - source_depth)

    # Sort absolute deviations for their median
    for i in range(1, n):
        key = abs_devs[i]
        j = i - 1
        while j >= 0 and abs_devs[j] > key:
            abs_devs[j + 1] = abs_devs[j]
            j -= 1
        abs_devs[j + 1] = key

    if n % 2 == 1:
        mad = abs_devs[n // 2]
    else:
        mad = np.float32(0.5) * (abs_devs[n // 2 - 1] + abs_devs[n // 2])

    # Scale MAD to sigma-equivalent (1.4826 is the standard consistency
    # factor for a normal distribution)
    mad_sigma = np.float32(1.4826) * mad

    # If MAD is zero (all cells have identical depth), nothing to remove
    if mad_sigma <= 0.0:
        return

    # --- Acceptance window centered on source_depth ---
    lower = source_depth - n_sigma * mad_sigma
    upper = source_depth + n_sigma * mad_sigma

    # --- Remove cells outside the acceptance window ---
    # Zero both accumulators together — leaving residual weight without
    # its paired WSE would dilute the WSE of neighboring contributions.
    for lr in range(n_r):
        gr = r_min + lr
        for lc in range(n_c):
            gc = c_min + lc
            elev = E[gr, gc]
            tw   = Total_Weight[gr, gc]
            if elev <= -9998.0 or tw <= 0.0:
                continue
            wse   = WSE_Times_Weight[gr, gc] / tw
            depth = wse - elev
            if depth <= 0.0:
                continue
            if depth < lower or depth > upper:
                WSE_Times_Weight[gr, gc] = np.float32(0.0)
                Total_Weight[gr, gc]     = np.float32(0.0)

@njit(cache=True)
def remove_downhill_drift_contributions(
    WSE_Times_Weight, Total_Weight, E,
    r_min, r_max, c_min, c_max,
    r_use, c_use,
    source_wse,
    max_depth_multiplier=np.float32(10.0)
):
    """
    Removes cells from the WSE accumulator arrays whose depth exceeds a
    simple multiple of the stream cell's own depth.

    The artifact occurs when a stream cell's WSE drifts downhill into
    topographic depressions far below the stream, producing unrealistically
    deep flooding. The fix is straightforward: no cell in the contribution
    box should be deeper than max_depth_multiplier times the depth at the
    stream cell itself. If it is, the weighted average at that cell has
    been pulled into an unrealistic depression and the contribution is removed.

    Parameters
    ----------
    WSE_Times_Weight : 2D float32 array (nrows+2, ncols+2)
        Running WSE * weight accumulator — modified in place.
    Total_Weight : 2D float32 array (nrows+2, ncols+2)
        Running weight accumulator — modified in place.
    E : 2D float32 array (nrows+2, ncols+2)
        Ground elevation raster (padded).
    r_min, r_max, c_min, c_max : int
        Row/col bounds of the contribution box in padded array coordinates.
    r_use, c_use : int
        Row/col of the stream cell in padded array coordinates.
    source_wse : float32
        WSE at the stream cell.
    max_depth_multiplier : float32
        Maximum allowed depth at any cell in the box, expressed as a
        multiple of source_depth. Default 10.0 — a cell deeper than
        10x the stream cell depth is almost certainly a drift artifact.
        Tunable: lower values are stricter; higher values more permissive
        for streams with significant surrounding relief.
    """

    # Depth at the stream cell — the physical reference for this contribution.
    # All cells in the box are judged relative to this value.
    source_depth = source_wse - E[r_use, c_use]

    # Nothing to filter if the stream cell itself has negligible depth
    if source_depth <= np.float32(0.01):
        return

    # Any cell deeper than this is a drift artifact
    max_allowed_depth = source_depth * max_depth_multiplier

    for lr in range(r_max - r_min):
        gr = r_min + lr
        for lc in range(c_max - c_min):
            gc = c_min + lc

            elev = E[gr, gc]
            tw   = Total_Weight[gr, gc]

            if elev <= -9998.0 or tw <= 0.0:
                continue

            # Recover weighted-average WSE and compute depth at this cell
            depth = (WSE_Times_Weight[gr, gc] / tw) - elev

            if depth <= 0.0:
                continue

            # If depth exceeds the allowed multiple of source depth,
            # this cell has received unrealistic drift — remove it from
            # both accumulators so it does not appear in the final WSE_array
            if depth > max_allowed_depth:
                WSE_Times_Weight[gr, gc] = np.float32(0.0)
                Total_Weight[gr, gc]     = np.float32(0.0)

@njit(cache=True)
def CreateSimpleFloodMap(RR, CC, T_Rast, W_Rast, S_Rast, E, B, 
                         flowdir, nrows, ncols, sd, TW_m, dx, dy, 
                         LocalFloodOption, 
                         COMID_Unique_TW: COMID_FLOW_DICT_TYPE,
                         COMID_Unique_Depth: COMID_FLOW_DICT_TYPE,
                         WeightBox, TW_for_WeightBox_ElipseMask, 
                         TopWidthPlausibleLimit, Set_Depth, flood_vdt_cells, OutDEP,
                         mapper, OutWSE):
       
    COMID_Averaging_Method = 0

    # these are for the weighted approach but have to be made either way to keep Numba's shape inference consistent
    WSE_Times_Weight = np.zeros((nrows+2,ncols+2), dtype=np.float32)
    # Always allocate to keep Numba's shape inference consistent.
    Slope_Times_Weight = np.zeros((nrows+2,ncols+2), dtype=np.float32)
    WSE_array = np.empty_like(WSE_Times_Weight, dtype=np.float32)

    Total_Weight = np.zeros((nrows+2,ncols+2), dtype=np.float32)

    if mapper == "Curve2Flood-Kernel Weighted":
        # this is the original curve2flood flood mapping method
        WSE_array, WSE_Times_Weight, Total_Weight, Slope_Times_Weight = create_kernel_weighted_spread_map(
            RR,
            CC,
            T_Rast,
            W_Rast,
            S_Rast,
            E,
            B,
            nrows,
            ncols,
            sd,
            TW_m,
            dx,
            dy,
            LocalFloodOption,
            COMID_Unique_TW,
            COMID_Unique_Depth,
            WeightBox,
            TW_for_WeightBox_ElipseMask,
            TopWidthPlausibleLimit,
            Set_Depth,
            COMID_Averaging_Method,
        )
    
    elif mapper == "Curve2Flood-FLDPLNpy":

        # if using Set_Depth, we want to seed all stream cells with WSE = E + Set_Depth, and then let the FLDPLN model expand from there.
        if Set_Depth > 0.0:
            W_Rast_Padded = np.full((nrows + 2, ncols + 2), np.nan, dtype=np.float32)
            for r in range(nrows):
                for c in range(ncols):
                    # B and E are already padded in this scope.
                    if B[r + 1, c + 1] > 0:
                        W_Rast_Padded[r + 1, c + 1] = E[r + 1, c + 1] + Set_Depth
                    else:
                        W_Rast_Padded[r + 1, c + 1] = np.nan

        else:
            # pad the W_Rast to make it match E and the other arrays that are all nrows+2 by ncols+2
            W_Rast_Padded = np.full((nrows + 2, ncols + 2), np.nan, dtype=np.float32)
            for r in range(nrows):
                for c in range(ncols):
                    if B[r + 1, c + 1] > 0:
                        W_Rast_Padded[r + 1, c + 1] = W_Rast[r, c]
                    else:
                        W_Rast_Padded[r + 1, c + 1] = np.nan

        # This spreads WSE outward from currently-flooded cells to dry cells as long as
        # neighboring ground elevation E can be overtopped.
        WSE_array = fldpln(
                            W_Rast_Padded, E, flowdir, B, nrows+2, ncols+2, dx, dy
        )

    # Do not flood cells where WSE is below E and E/WSE are nan values
    Flooded_array = np.where((WSE_array > E) & (E > -9998.0), 1, 0).astype(np.uint8)

    # Also make sure all the Cells that have Stream are counted as flooded.
    if flood_vdt_cells:
        for i in range(len(RR)):
            Flooded_array[RR[i],CC[i]] = 1

    # Create the Depth array
    if OutDEP or S_Rast is not None or OutWSE:
        Depth_array = np.where((WSE_array > E) & (E > -9998.0), WSE_array - E, np.nan).astype(np.float32)
    else:
        Depth_array = np.empty((3, 3), dtype=np.float32) # Dummy array if not used

    # if you want, create the slope array
    if S_Rast is not None:
        Slope_divided_by_weight = Slope_Times_Weight / Total_Weight
        Slope_array = np.where((WSE_array > E) & (E > -9998.0), Slope_divided_by_weight, np.nan).astype(np.float32)
        Slope_array = np.where((Slope_array <= 0), 0.0002, Slope_array).astype(np.float32)
        return Flooded_array[1:-1, 1:-1], Depth_array[1:-1, 1:-1], Slope_array[1:-1, 1:-1]


    return Flooded_array[1:-1, 1:-1], Depth_array[1:-1, 1:-1], None

@njit(cache=True)
def create_kernel_weighted_spread_map(
    RR,
    CC,
    T_Rast,
    W_Rast,
    S_Rast,
    E,
    B,
    nrows,
    ncols,
    sd,
    TW_m,
    dx,
    dy,
    LocalFloodOption,
    COMID_Unique_TW: COMID_FLOW_DICT_TYPE,
    COMID_Unique_Depth: COMID_FLOW_DICT_TYPE,
    WeightBox,
    TW_for_WeightBox_ElipseMask,
    TopWidthPlausibleLimit,
    Set_Depth,
    COMID_Averaging_Method,
):
    WSE_Times_Weight = np.zeros((nrows + 2, ncols + 2), dtype=np.float32)
    Slope_Times_Weight = np.zeros((nrows + 2, ncols + 2), dtype=np.float32)
    Total_Weight = np.zeros((nrows + 2, ncols + 2), dtype=np.float32)

    #Now go through each cell
    num_nonzero = len(RR)
    number_filtered = 0
    delta_wse_threshold_m = np.float32(1.0)
    large_delta_patch_area_threshold_ratio = np.float32(0.30)
    for i in range(num_nonzero):
        r = RR[i]
        c = CC[i]
        r_use = r
        c_use = c
        E_Min = E[r,c]
        
        COMID_Value = B[r,c]
        if Set_Depth>0.0:
            WSE = float(E[r_use,c_use] + Set_Depth)
            if S_Rast is not None:
                SLOPE = float(S_Rast[r_use,c_use])
            COMID_TW_m = TopWidthPlausibleLimit
        elif COMID_Averaging_Method!=0:
            #Get COMID, TopWidth, and Depth Information for this cell
            COMID_Value = B[r,c]
            # keys are int32, values are float32
            if COMID_Value in COMID_Unique_TW:
                COMID_TW_m = COMID_Unique_TW[COMID_Value]
            else:
                COMID_TW_m = np.float32(0.0)

            if COMID_Value in COMID_Unique_Depth:
                COMID_D = COMID_Unique_Depth[COMID_Value]
            else:
                COMID_D = np.float32(0.0)
            WSE = float(E[r_use,c_use] + COMID_D)
            if S_Rast is not None:
                SLOPE = float(S_Rast[r_use,c_use])
        else:
            #These are Based on the AutoRoute/ARC Results, not averaged for COMID
            WSE = np.round(W_Rast[r-1,c-1], 2)  #Have to have the '-1' because of the Row and Col being inset on the B raster.
            COMID_TW_m = T_Rast[r-1,c-1]
            if S_Rast is not None:
                SLOPE = S_Rast[r-1,c-1]

        if COMID_TW_m < 0.00001 or (WSE - E[r,c]) < 0.001:
            continue

        # give the TW for the weightbox the median if its smaller than the median.
        if COMID_TW_m > TW_m:
            COMID_TW_m = TW_m

        #This is how many cells we will be looking at surrounding our stream cell
        COMID_TW = int(max(np.round(COMID_TW_m / dx), np.round(COMID_TW_m / dy)))

        
        # Find minimum elevation within the search box
        if sd >= 1:
            for rr in range(max(r - sd, 0), min(r + sd + 1, nrows - 1)):
                for cc in range(max(c - sd, 1), min(c + sd + 1, ncols - 1)):
                    if E[rr,cc] > 0.1 and E[rr,cc] < E_Min:
                        E_Min = E[rr,cc]
                        r_use = rr
                        c_use = cc

        r_min = max(r_use - COMID_TW, 1)
        r_max = min(r_use + COMID_TW + 1, nrows + 1)
        c_min = max(c_use - COMID_TW, 1)
        c_max = min(c_use + COMID_TW + 1, ncols + 1)
        
        # This uses the weighting method from FloodSpreader to create a flood map
        # Here we use TW instead of COMID_TW.  This is because we are trying to find the center of the weight raster, which was set based on TW (not COMID_TW).  
        # COMID_TW mainly applies to the r_min, r_max, c_min, c_max
        w_r_min = TW_for_WeightBox_ElipseMask - (r_use - r_min)
        w_r_max = TW_for_WeightBox_ElipseMask + (r_max - r_use)
        w_c_min = TW_for_WeightBox_ElipseMask - (c_use - c_min)
        w_c_max = TW_for_WeightBox_ElipseMask + (c_max - c_use)

        weight_slice = WeightBox[w_r_min:w_r_max, w_c_min:w_c_max]
        boundary_has_flooded_too_much = False
        if LocalFloodOption:
            #Find what would flood local
            E_Box = E[r_min:r_max,c_min:c_max]
            FloodLocalMask = FloodAllLocalAreas(WSE, E_Box, r_min, r_max, c_min, c_max, r_use, c_use)
            WSE_Times_Weight[r_min:r_max, c_min:c_max] += (WSE * weight_slice * FloodLocalMask)
            Total_Weight[r_min:r_max,c_min:c_max] += weight_slice * FloodLocalMask
            if S_Rast is not None:
                Slope_Times_Weight[r_min:r_max, c_min:c_max] += (SLOPE * weight_slice * FloodLocalMask)
        else:

            # This puts the weights from each cell into the composite arrays.
            cell_wse_weight = WSE * weight_slice                
            WSE_Times_Weight[r_min:r_max, c_min:c_max] += cell_wse_weight
            Total_Weight[r_min:r_max,c_min:c_max] += weight_slice

            # This makes a weighted slope raster that can be used for velocity estimates
            if S_Rast is not None:
                Slope_Times_Weight[r_min:r_max, c_min:c_max] += (SLOPE * weight_slice)

    # These are the cells that we want to flood based on the weighted WSE being greater than the elevation, and also making sure the elevation is valid and that we have some weight there.
    valid_candidate = (
        (E > -9998.0) &
        (WSE_Times_Weight > E * Total_Weight)  # Keeps the values where WSE is greater than E, which means it would be flooded
    )

    WSE_array = np.where(valid_candidate, WSE_Times_Weight / Total_Weight, np.nan).astype(np.float32)
    return WSE_array, WSE_Times_Weight, Total_Weight, Slope_Times_Weight

def _grid_xy_from_rc(rows: np.ndarray, cols: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert zero-based raster row/column indices to cell-center coordinates on a
    local projected grid measured in meters.

    The x axis increases to the right. The y axis is negative downward so the
    generated GDAL geotransform matches raster row indexing.
    """
    x = (cols.astype(np.float32) + np.float32(0.5)) * np.float32(dx)
    y = -((rows.astype(np.float32) + np.float32(0.5)) * np.float32(dy))
    return x.astype(np.float32), y.astype(np.float32)

def build_fhs_points_from_curve2flood_inputs(
    RR: np.ndarray,
    CC: np.ndarray,
    T_Rast: np.ndarray,
    W_Rast: np.ndarray,
    S_Rast: np.ndarray | None,
    E: np.ndarray,
    B: np.ndarray,
    COMID_Unique_TW: dict[int, float],
    COMID_Unique_Depth: dict[int, float],
    TopWidthPlausibleLimit: float,
    Set_Depth: float,
    dx: float,
    dy: float,
    COMID_Averaging_Method: int = 0,
    extra_points: dict | None = None,
) -> dict[str, np.ndarray]:
    """
    Build an HWM-style point set from the stream cells already available inside
    Curve2Flood.

    This is the array-based bridge between Curve2Flood's centerline/VDT inputs
    and the FHS flood-mapping engine. By default it emits one point per stream
    cell. Optional `extra_points` can be supplied to append any off-center
    points, such as cross-section extension points generated elsewhere.

    Parameters
    ----------
    RR, CC : ndarray
        Padded stream-cell row/column indices, as used throughout Curve2Flood.
    T_Rast, W_Rast, S_Rast : ndarray
        AutoRoute/ARC top-width, WSE, and slope rasters on the unpadded DEM
        grid. `S_Rast` may be None.
    E, B : ndarray
        Padded DEM and stream-ID rasters.
    COMID_Unique_TW, COMID_Unique_Depth : dict
        Per-COMID fallback top width and depth values.
    TopWidthPlausibleLimit : float
        Maximum plausible top width in meters, used when Set_Depth is active.
    Set_Depth : float
        Fixed depth override. When positive, WSE is computed as E + Set_Depth.
    dx, dy : float
        Cell spacing in meters.
    COMID_Averaging_Method : int, default 0
        Non-zero forces use of COMID-mean depth/top-width rather than cellwise
        ARC rasters.
    extra_points : dict, optional
        Optional additional HWM-style points. Supported keys are:
        `rows`, `cols`, `x`, `y`, `wse`, `topwidth`, `dem`, `ids`, `types`,
        and `slope`.

    Returns
    -------
    dict
        Dictionary containing point arrays: `rows`, `cols`, `x`, `y`, `wse`,
        `topwidth`, `dem`, `ids`, `types`, and `slope`.

    Notes
    -----
    This adapter does not generate the cross-section expansion points from the
    original FHS script on its own because Curve2Flood does not currently carry
    XS angle metadata through this code path. To reproduce the exact FHS point
    cloud, pass those extra points through `extra_points`.
    """
    rows = []
    cols = []
    wse_vals = []
    topwidth_vals = []
    dem_vals = []
    id_vals = []
    slope_vals = []

    for i in range(len(RR)):
        r = int(RR[i])
        c = int(CC[i])
        comid_value = int(B[r, c])
        point_dem = float(E[r, c])

        # Match Curve2Flood's existing precedence: fixed-depth override first,
        # then COMID fallback hydraulics, then cellwise raster values.
        if Set_Depth > 0.0:
            point_wse = point_dem + float(Set_Depth)
            point_topwidth = float(TopWidthPlausibleLimit)
            point_slope = float(S_Rast[r - 1, c - 1]) if S_Rast is not None else np.nan
        elif COMID_Averaging_Method != 0 or W_Rast[r - 1, c - 1] < 0.001 or T_Rast[r - 1, c - 1] < 0.00001:
            point_topwidth = float(COMID_Unique_TW.get(comid_value, 0.0))
            point_depth = float(COMID_Unique_Depth.get(comid_value, 0.0))
            point_wse = point_dem + point_depth
            point_slope = float(S_Rast[r - 1, c - 1]) if S_Rast is not None else np.nan
        else:
            point_wse = float(np.round(W_Rast[r - 1, c - 1], 2))
            point_topwidth = float(T_Rast[r - 1, c - 1])
            point_slope = float(S_Rast[r - 1, c - 1]) if S_Rast is not None else np.nan

        if (not np.isfinite(point_wse)) or point_wse < 0.001:
            continue
        if (not np.isfinite(point_topwidth)) or point_topwidth < 0.00001:
            continue
        if (not np.isfinite(point_dem)) or point_dem <= -9998.0:
            continue
        if point_wse - point_dem < 0.001:
            continue

        rows.append(r - 1)
        cols.append(c - 1)
        wse_vals.append(point_wse)
        topwidth_vals.append(point_topwidth)
        dem_vals.append(point_dem)
        id_vals.append(comid_value)
        slope_vals.append(point_slope)

    point_rows = np.asarray(rows, dtype=np.int32)
    point_cols = np.asarray(cols, dtype=np.int32)
    point_x, point_y = _grid_xy_from_rc(point_rows, point_cols, dx, dy)

    point_data = {
        "rows": point_rows,
        "cols": point_cols,
        "x": point_x,
        "y": point_y,
        "wse": np.asarray(wse_vals, dtype=np.float32),
        "topwidth": np.asarray(topwidth_vals, dtype=np.float32),
        "dem": np.asarray(dem_vals, dtype=np.float32),
        "ids": np.asarray(id_vals, dtype=np.int32),
        "types": np.full(len(point_rows), "Centerline", dtype=object),
        "slope": np.asarray(slope_vals, dtype=np.float32),
    }

    if extra_points is None:
        return point_data

    if "wse" not in extra_points:
        raise ValueError("extra_points must include a 'wse' array.")

    extra_wse = np.asarray(extra_points["wse"], dtype=np.float32)
    extra_count = len(extra_wse)
    if extra_count == 0:
        return point_data

    if "x" in extra_points and "y" in extra_points:
        extra_x = np.asarray(extra_points["x"], dtype=np.float32)
        extra_y = np.asarray(extra_points["y"], dtype=np.float32)
        if len(extra_x) != extra_count or len(extra_y) != extra_count:
            raise ValueError("extra_points x/y arrays must match the length of extra_points['wse'].")
        if "rows" in extra_points and "cols" in extra_points:
            extra_rows = np.asarray(extra_points["rows"], dtype=np.int32)
            extra_cols = np.asarray(extra_points["cols"], dtype=np.int32)
        else:
            extra_rows = np.full(extra_count, -1, dtype=np.int32)
            extra_cols = np.full(extra_count, -1, dtype=np.int32)
    elif "rows" in extra_points and "cols" in extra_points:
        extra_rows = np.asarray(extra_points["rows"], dtype=np.int32)
        extra_cols = np.asarray(extra_points["cols"], dtype=np.int32)
        if len(extra_rows) != extra_count or len(extra_cols) != extra_count:
            raise ValueError("extra_points rows/cols arrays must match the length of extra_points['wse'].")
        extra_x, extra_y = _grid_xy_from_rc(extra_rows, extra_cols, dx, dy)
    else:
        raise ValueError("extra_points must provide either x/y arrays or rows/cols arrays.")

    extra_topwidth = np.asarray(
        extra_points.get("topwidth", np.full(extra_count, TopWidthPlausibleLimit, dtype=np.float32)),
        dtype=np.float32,
    )
    extra_dem = np.asarray(extra_points.get("dem", np.full(extra_count, np.nan, dtype=np.float32)), dtype=np.float32)
    extra_ids = np.asarray(extra_points.get("ids", np.full(extra_count, -1, dtype=np.int32)), dtype=np.int32)
    extra_types = np.asarray(extra_points.get("types", np.full(extra_count, "XS_Extension", dtype=object)), dtype=object)
    extra_slope = np.asarray(extra_points.get("slope", np.full(extra_count, np.nan, dtype=np.float32)), dtype=np.float32)

    for key, arr in {
        "topwidth": extra_topwidth,
        "dem": extra_dem,
        "ids": extra_ids,
        "types": extra_types,
        "slope": extra_slope,
    }.items():
        if len(arr) != extra_count:
            raise ValueError(f"extra_points['{key}'] must match the length of extra_points['wse'].")

    for key, arr in {
        "rows": extra_rows,
        "cols": extra_cols,
        "x": extra_x,
        "y": extra_y,
        "wse": extra_wse,
        "topwidth": extra_topwidth,
        "dem": extra_dem,
        "ids": extra_ids,
        "types": extra_types,
        "slope": extra_slope,
    }.items():
        point_data[key] = np.concatenate((point_data[key], arr))

    return point_data

def build_variable_buffer_masks_from_points(
    x: np.ndarray,
    y: np.ndarray,
    topwidth: np.ndarray,
    dem_shape: tuple[int, int],
    transform: GeoTransform,
    fixed_corridor_buffer_m: float,
    fixed_anchor_buffer_m: float,
    use_topwidth_buffers: bool = True,
    corridor_topwidth_factor: float = 1.5,
    anchor_topwidth_factor: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Array-based version of the FHS point buffering step.

    Each point produces two raster masks:
    - `corridor`: the area eligible for IDW interpolation
    - `anchor`: the seed region used for hydraulic connectivity propagation
    """
    if len(x) == 0:
        empty = np.zeros(dem_shape, dtype=bool)
        return empty, empty

    # Rasterize per-point buffers so interpolation is constrained to the stream
    # corridor and connectivity starts from an anchor region near the points.
    points = [Point(float(px), float(py)) for px, py in zip(x, y)]
    if use_topwidth_buffers:
        corridor_shapes = (
            (geom.buffer(float(max(width, 0.0) * corridor_topwidth_factor)), 1)
            for geom, width in zip(points, topwidth)
        )
        anchor_shapes = (
            (geom.buffer(float(max(width, 0.0) * anchor_topwidth_factor)), 1)
            for geom, width in zip(points, topwidth)
        )
    else:
        corridor_shapes = ((geom.buffer(float(fixed_corridor_buffer_m)), 1) for geom in points)
        anchor_shapes = ((geom.buffer(float(fixed_anchor_buffer_m)), 1) for geom in points)

    corridor = rasterize_shapes_gdal(
        corridor_shapes,
        out_shape=dem_shape,
        transform=transform,
        dtype=np.uint8,
        all_touched=False,
    ).astype(bool)
    anchor = rasterize_shapes_gdal(
        anchor_shapes,
        out_shape=dem_shape,
        transform=transform,
        dtype=np.uint8,
        all_touched=False,
    ).astype(bool)
    return corridor, anchor

def build_target_coordinates(xs: np.ndarray, ys: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.where(mask)
    return xs[cols].astype(np.float32), ys[rows].astype(np.float32), rows.astype(np.int32), cols.astype(np.int32)

def precompute_idw_neighbors(
    x: np.ndarray,
    y: np.ndarray,
    point_max_distance: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    out_shape: tuple[int, int],
    target_rows: np.ndarray,
    target_cols: np.ndarray,
    k: int,
) -> dict[str, np.ndarray | tuple[int, int]]:
    """Precompute KD-tree neighbors for repeated IDW evaluations."""
    # Query the nearest points once, then record which of those neighbors are
    # actually usable after applying each point's interpolation radius.
    pts = np.column_stack((x, y))
    tree = cKDTree(pts)
    xy = np.column_stack((target_x, target_y))
    global_max_dist = float(np.nanmax(point_max_distance))
    dist, idx = tree.query(xy, k=k, distance_upper_bound=global_max_dist)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    valid_idx = idx < len(x)
    valid_dist = np.isfinite(dist) & valid_idx
    neighbor_radii = np.full(dist.shape, np.nan, dtype=np.float32)
    neighbor_radii[valid_idx] = point_max_distance[idx[valid_idx]]
    usable = valid_dist & (dist <= neighbor_radii)
    return {
        "dist": dist.astype(np.float32),
        "idx": idx.astype(np.int32),
        "usable": usable,
        "rows": target_rows.astype(np.int32),
        "cols": target_cols.astype(np.int32),
        "shape": out_shape,
    }

def idw_from_precomputed(z: np.ndarray, valid_points: np.ndarray, pre: dict, power: float) -> np.ndarray:
    """
    Evaluate IDW interpolation from a precomputed neighbor table.

    This is a direct port of the FHS interpolation routine, including the
    `len(z)` sentinel handling used by `cKDTree.query(..., distance_upper_bound=...)`.
    """
    dist = pre["dist"]
    idx = pre["idx"]
    usable = pre["usable"].copy()
    rows = pre["rows"]
    cols = pre["cols"]
    out_shape = pre["shape"]

    mask_idx = idx < len(z)
    valid_neighbor = np.zeros(idx.shape, dtype=bool)
    valid_neighbor[mask_idx] = valid_points[idx[mask_idx]]
    usable &= valid_neighbor

    # Use standard inverse-distance weights except when a target lands exactly
    # on a point, in which case that point gets all the weight.
    weights = np.zeros(dist.shape, dtype=np.float32)
    positive = usable & (dist > 0)
    weights[positive] = 1.0 / np.power(dist[positive], power)

    exact = usable & (dist == 0)
    exact_rows = np.where(np.any(exact, axis=1))[0]
    for r in exact_rows:
        weights[r, :] = 0.0
        first_exact = np.where(exact[r])[0][0]
        weights[r, first_exact] = 1.0

    z_safe = np.zeros(len(z) + 1, dtype=np.float32)
    z_safe[:len(z)] = z

    weight_sum = weights.sum(axis=1)
    good = weight_sum > 0
    vals = np.full(len(weight_sum), np.nan, dtype=np.float32)
    if np.any(good):
        vals[good] = (
            np.sum(weights[good] * z_safe[idx[good]], axis=1) / weight_sum[good]
        ).astype(np.float32)

    out = np.full(out_shape, np.nan, dtype=np.float32)
    out[rows, cols] = vals
    return out

def smooth_nan(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smooth while preserving NaN masks."""
    # Smooth the values and the valid-data mask separately so NaN gaps do not
    # behave like zero-valued water surface cells.
    valid = np.isfinite(arr)
    arr_filled = np.where(valid, arr, 0.0).astype(np.float32)
    num = ndi.gaussian_filter(arr_filled, sigma)
    den = ndi.gaussian_filter(valid.astype(np.float32), sigma)
    out = np.full_like(arr, np.nan, dtype=np.float32)
    mask = den > 1.0e-6
    out[mask] = num[mask] / den[mask]
    return out

def summarize_sanity_filter(prefix: str, valid_before: np.ndarray, valid_after: np.ndarray, point_types: np.ndarray | None = None) -> str:
    """Build the same human-readable WSE sanity-filter summary used by FHS."""
    dropped = valid_before & ~valid_after
    total_before = int(np.count_nonzero(valid_before))
    total_after = int(np.count_nonzero(valid_after))
    total_dropped = int(np.count_nonzero(dropped))
    msg = f"{prefix}: kept {total_after} of {total_before} scenario points after WSE sanity filter"
    if total_dropped > 0 and point_types is not None:
        unique, counts = np.unique(point_types[dropped].astype(str), return_counts=True)
        parts = [f"{u}={int(c)}" for u, c in zip(unique, counts)]
        msg += f"; dropped {total_dropped} ({', '.join(parts)})"
    elif total_dropped > 0:
        msg += f"; dropped {total_dropped}"
    return msg

def compute_fhs_flood_from_wse(
    wse: np.ndarray,
    dem: np.ndarray,
    corridor: np.ndarray,
    anchor: np.ndarray,
    connectivity: int,
    smooth_sigma_pixels: float,
    permanent_water_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert an interpolated WSE surface to a flood mask using the original FHS
    corridor/anchor/connectivity rules.
    """
    if smooth_sigma_pixels > 0:
        wse = smooth_nan(wse, smooth_sigma_pixels)

    # A cell is initially eligible when the interpolated WSE overtops the DEM
    # inside the corridor. Anchor cells then enforce connected flood regions.
    valid_wse = np.isfinite(wse)
    flood = valid_wse & corridor & (dem < wse)
    flood |= anchor

    structure = ndi.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    flood = ndi.binary_propagation(anchor, mask=flood, structure=structure)

    dry_cells = dem >= wse
    wse = wse.copy()
    wse[dry_cells] = np.nan
    flood[dry_cells] = False

    if permanent_water_mask is not None:
        flood |= permanent_water_mask

    return flood.astype(np.uint8), wse.astype(np.float32)

def create_fhs_flood_map_from_points(
    dem: np.ndarray,
    point_wse: np.ndarray,
    point_rows: np.ndarray | None = None,
    point_cols: np.ndarray | None = None,
    point_x: np.ndarray | None = None,
    point_y: np.ndarray | None = None,
    point_topwidth: np.ndarray | None = None,
    point_ids: np.ndarray | None = None,
    point_dem: np.ndarray | None = None,
    point_types: np.ndarray | None = None,
    point_slope: np.ndarray | None = None,
    dx: float = 1.0,
    dy: float | None = None,
    prefix: str = "default",
    corridor_buffer_m: float = 500.0,
    anchor_buffer_m: float = 5.0,
    use_topwidth_buffers: bool = True,
    corridor_topwidth_factor: float = 1.5,
    anchor_topwidth_factor: float = 0.5,
    connectivity: int = 4,
    k: int = 20,
    power: float = 2.0,
    max_distance_m: float = 1000.0,
    use_topwidth_max_distance: bool = True,
    maxdist_topwidth_factor: float = 1.5,
    min_topwidth_m: float = 5.0,
    max_topwidth_m: float = 500.0,
    fallback_topwidth_m: float = 20.0,
    smooth_sigma_pixels: float = 0.5,
    apply_wse_sanity_filter: bool = True,
    wse_sanity_tolerance_m: float = 0.0,
    permanent_water_mask: np.ndarray | None = None,
    fast_mode: bool = False,
    precomputed_neighbors: dict | None = None,
    precomputed_corridor_mask: np.ndarray | None = None,
    precomputed_anchor_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray | str | bool]:
    """
    Port of the FHS `create_flood_map_from_hwms` single-scenario flood engine.

    Parameters
    ----------
    dem : ndarray
        Unpadded DEM array for the working grid.
    point_wse : ndarray
        Water-surface elevations for one scenario.
    point_rows, point_cols : ndarray, optional
        Zero-based raster indices for the points. Required when `point_x` and
        `point_y` are not provided and also used to sample DEM elevations when
        `point_dem` is omitted.
    point_x, point_y : ndarray, optional
        Point coordinates on a local projected grid measured in meters.
    point_topwidth : ndarray, optional
        Per-point top width values in meters. If omitted, a constant fallback is
        used.
    point_ids : ndarray, optional
        Point IDs used for COMID-wise median interpolation distances.
    point_dem : ndarray, optional
        DEM elevation at the point locations. If omitted and rows/cols are
        available, values are sampled from `dem`.
    point_types : ndarray, optional
        Optional labels used in the sanity-filter summary.
    point_slope : ndarray, optional
        Optional slope values interpolated with the same IDW neighbors so the
        caller can still build velocity products.
    dx, dy : float
        Cell spacing in meters.
    prefix : str, default "default"
        Scenario label used in the returned summary string.
    corridor_buffer_m, anchor_buffer_m : float
        Fixed buffer distances used when top-width buffers are disabled.
    use_topwidth_buffers : bool
        If True, corridor/anchor radii are derived from point top widths.
    corridor_topwidth_factor, anchor_topwidth_factor : float
        Multipliers converting top width to corridor/anchor radii.
    connectivity : int
        Either 4 or 8. Controls raster connectivity during flood propagation.
    k, power : int, float
        IDW nearest-neighbor count and inverse-distance power.
    max_distance_m : float
        Fixed interpolation radius in meters.
    use_topwidth_max_distance : bool
        If True and `point_ids` are available, per-point max distance is the
        COMID-wise median top width times `maxdist_topwidth_factor`.
    min_topwidth_m, max_topwidth_m, fallback_topwidth_m : float
        Top-width clipping and fallback settings.
    smooth_sigma_pixels : float
        Gaussian smoothing applied to the interpolated WSE grid.
    apply_wse_sanity_filter : bool
        If True, drop points whose WSE falls below the DEM at the point.
    wse_sanity_tolerance_m : float
        Non-negative tolerance added before the WSE-vs-DEM sanity comparison.
    permanent_water_mask : ndarray, optional
        Boolean mask of always-wet cells.
    fast_mode : bool
        Mirrors the FHS fast-mode overrides by disabling top-width-based buffers
        and top-width-based max interpolation distance.
    precomputed_neighbors, precomputed_corridor_mask, precomputed_anchor_mask : optional
        Optional reusable state for repeated scenario evaluation. If supplied,
        they are used exactly as in the original FHS fast worker.

    Returns
    -------
    dict
        Keys: `flood`, `wse`, `depth`, `slope`, `stats_message`,
        `valid_point_mask`, and `has_valid_points`.

    Notes
    -----
    This function ports the flood-raster generation part of
    `FHS_FloodMapper_AllInOne.py`. It intentionally does not reimplement the
    original CLI, raster reprojection, or VDT-to-HWM point generation wrapper.
    Those are better handled outside this low-level helper.
    """
    dem = np.asarray(dem, dtype=np.float32)
    if dem.ndim != 2:
        raise ValueError("dem must be a 2D array.")

    # Normalize the caller inputs into one consistent in-memory point
    # representation before any buffering or interpolation begins.
    dy = dx if dy is None else dy
    point_wse = np.asarray(point_wse, dtype=np.float32)
    n_points = len(point_wse)

    if point_x is None or point_y is None:
        if point_rows is None or point_cols is None:
            raise ValueError("Provide either point_x/point_y or point_rows/point_cols.")
        point_rows = np.asarray(point_rows, dtype=np.int32)
        point_cols = np.asarray(point_cols, dtype=np.int32)
        if len(point_rows) != n_points or len(point_cols) != n_points:
            raise ValueError("point_rows and point_cols must match point_wse length.")
        point_x, point_y = _grid_xy_from_rc(point_rows, point_cols, dx, dy)
    else:
        point_x = np.asarray(point_x, dtype=np.float32)
        point_y = np.asarray(point_y, dtype=np.float32)
        if len(point_x) != n_points or len(point_y) != n_points:
            raise ValueError("point_x and point_y must match point_wse length.")
        if point_rows is None or point_cols is None:
            point_rows = np.full(n_points, -1, dtype=np.int32)
            point_cols = np.full(n_points, -1, dtype=np.int32)
        else:
            point_rows = np.asarray(point_rows, dtype=np.int32)
            point_cols = np.asarray(point_cols, dtype=np.int32)

    if point_topwidth is None:
        point_topwidth = np.full(n_points, fallback_topwidth_m, dtype=np.float32)
    else:
        point_topwidth = np.asarray(point_topwidth, dtype=np.float32)
        if len(point_topwidth) != n_points:
            raise ValueError("point_topwidth must match point_wse length.")
        point_topwidth = np.where(np.isfinite(point_topwidth), point_topwidth, fallback_topwidth_m)
    point_topwidth = np.clip(point_topwidth, min_topwidth_m, max_topwidth_m).astype(np.float32)

    if point_ids is not None:
        point_ids = np.asarray(point_ids)
        if len(point_ids) != n_points:
            raise ValueError("point_ids must match point_wse length.")

    if point_types is None:
        point_types = np.full(n_points, "Unknown", dtype=object)
    else:
        point_types = np.asarray(point_types, dtype=object)
        if len(point_types) != n_points:
            raise ValueError("point_types must match point_wse length.")

    if point_slope is not None:
        point_slope = np.asarray(point_slope, dtype=np.float32)
        if len(point_slope) != n_points:
            raise ValueError("point_slope must match point_wse length.")

    if point_dem is None:
        point_dem = np.full(n_points, np.nan, dtype=np.float32)
        inside = (
            (point_rows >= 0) & (point_rows < dem.shape[0]) &
            (point_cols >= 0) & (point_cols < dem.shape[1])
        )
        point_dem[inside] = dem[point_rows[inside], point_cols[inside]]
    else:
        point_dem = np.asarray(point_dem, dtype=np.float32)
        if len(point_dem) != n_points:
            raise ValueError("point_dem must match point_wse length.")

    valid = np.isfinite(point_wse)
    valid_before = valid.copy()
    if apply_wse_sanity_filter:
        valid &= np.isfinite(point_dem)
        valid &= point_wse + wse_sanity_tolerance_m >= point_dem

    stats_message = summarize_sanity_filter(prefix=prefix, valid_before=valid_before, valid_after=valid, point_types=point_types)

    if fast_mode:
        use_topwidth_buffers = False
        use_topwidth_max_distance = False

    nrows, ncols = dem.shape
    transform = (0.0, float(dx), 0.0, 0.0, 0.0, -float(dy))
    xs = (np.arange(ncols, dtype=np.float32) + np.float32(0.5)) * np.float32(dx)
    ys = -((np.arange(nrows, dtype=np.float32) + np.float32(0.5)) * np.float32(dy))

    # Build or reuse the masks that define where IDW can populate WSE values
    # and where hydraulic connectivity is allowed to start.
    if precomputed_corridor_mask is not None and precomputed_anchor_mask is not None:
        corridor = np.asarray(precomputed_corridor_mask, dtype=bool)
        anchor = np.asarray(precomputed_anchor_mask, dtype=bool)
    else:
        x_valid = point_x[valid]
        y_valid = point_y[valid]
        tw_valid = point_topwidth[valid]
        corridor, anchor = build_variable_buffer_masks_from_points(
            x=x_valid,
            y=y_valid,
            topwidth=tw_valid,
            dem_shape=dem.shape,
            transform=transform,
            fixed_corridor_buffer_m=corridor_buffer_m,
            fixed_anchor_buffer_m=anchor_buffer_m,
            use_topwidth_buffers=use_topwidth_buffers,
            corridor_topwidth_factor=corridor_topwidth_factor,
            anchor_topwidth_factor=anchor_topwidth_factor,
        )

    if not np.any(valid):
        flood = np.zeros(dem.shape, dtype=np.uint8)
        wse = np.full(dem.shape, np.nan, dtype=np.float32)
        depth = np.full(dem.shape, np.nan, dtype=np.float32)
        slope = np.full(dem.shape, np.nan, dtype=np.float32) if point_slope is not None else None
        if permanent_water_mask is not None:
            flood = np.where(permanent_water_mask, 1, flood).astype(np.uint8)
        return {
            "flood": flood,
            "wse": wse,
            "depth": depth,
            "slope": slope,
            "stats_message": stats_message,
            "valid_point_mask": valid,
            "has_valid_points": False,
        }

    if precomputed_neighbors is not None:
        z_all = np.where(valid, point_wse.astype(np.float32), 0.0).astype(np.float32)
        wse = idw_from_precomputed(z_all, valid, precomputed_neighbors, power)
        if point_slope is not None:
            slope_seed = np.where(valid, point_slope.astype(np.float32), 0.0).astype(np.float32)
            slope_interp = idw_from_precomputed(slope_seed, valid, precomputed_neighbors, power)
        else:
            slope_interp = None
    else:
        x_valid = point_x[valid]
        y_valid = point_y[valid]
        z_valid = point_wse[valid].astype(np.float32)
        tw_valid = point_topwidth[valid].astype(np.float32)

        # FHS can scale the interpolation reach by COMID using the median top
        # width, allowing larger rivers to influence a wider neighborhood.
        if point_ids is not None and use_topwidth_max_distance:
            tmp_df = pd.DataFrame({"id_col": point_ids[valid], "tw_based_dist": maxdist_topwidth_factor * tw_valid})
            grouped = tmp_df.groupby("id_col")["tw_based_dist"].median()
            point_max_distance = tmp_df["id_col"].map(grouped).to_numpy(dtype=np.float32, copy=True)
            bad = ~np.isfinite(point_max_distance)
            point_max_distance[bad] = max_distance_m
        else:
            point_max_distance = np.full(len(z_valid), max_distance_m, dtype=np.float32)

        # Interpolate only within the buffered corridor; cells outside it stay
        # NaN unless they are later forced wet by another rule.
        interp_mask = corridor & np.isfinite(dem)
        target_x, target_y, target_rows, target_cols = build_target_coordinates(xs, ys, interp_mask)
        if len(target_x) == 0:
            wse = np.full(dem.shape, np.nan, dtype=np.float32)
            slope_interp = np.full(dem.shape, np.nan, dtype=np.float32) if point_slope is not None else None
        else:
            pre = precompute_idw_neighbors(
                x=x_valid,
                y=y_valid,
                point_max_distance=point_max_distance,
                target_x=target_x,
                target_y=target_y,
                out_shape=dem.shape,
                target_rows=target_rows,
                target_cols=target_cols,
                k=k,
            )
            valid_points = np.ones(len(z_valid), dtype=bool)
            wse = idw_from_precomputed(z_valid, valid_points, pre, power)
            if point_slope is not None:
                slope_interp = idw_from_precomputed(point_slope[valid].astype(np.float32), valid_points, pre, power)
            else:
                slope_interp = None

    flood, wse = compute_fhs_flood_from_wse(
        wse=wse,
        dem=dem,
        corridor=corridor,
        anchor=anchor,
        connectivity=connectivity,
        smooth_sigma_pixels=smooth_sigma_pixels,
        permanent_water_mask=permanent_water_mask,
    )
    depth = np.where(np.isfinite(wse), np.maximum(wse - dem, 0.0), np.nan).astype(np.float32)

    if slope_interp is not None:
        slope = np.where(flood > 0, slope_interp, np.nan).astype(np.float32)
        slope = np.where(slope <= 0.0, 0.0002, slope).astype(np.float32)
    else:
        slope = None

    return {
        "flood": flood,
        "wse": wse,
        "depth": depth,
        "slope": slope,
        "stats_message": stats_message,
        "valid_point_mask": valid,
        "has_valid_points": True,
    }

def _compute_outlier_mask(values: np.ndarray, method: str = "mad", threshold: float = 3.5) -> np.ndarray:
    """
    FHS outlier-mask helper for depth-based HWM filtering.
    """
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 4:
        return np.zeros(values.shape, dtype=bool)

    valid_values = values[finite]
    method = str(method).lower()
    if method == "mad":
        center = np.median(valid_values)
        mad = np.median(np.abs(valid_values - center))
        if not np.isfinite(mad) or mad <= 1e-12:
            return np.zeros(values.shape, dtype=bool)
        robust_z = 0.6745 * (values - center) / mad
        outliers = np.zeros(values.shape, dtype=bool)
        outliers[finite] = np.abs(robust_z[finite]) > float(threshold)
        return outliers
    if method == "iqr":
        q1, q3 = np.percentile(valid_values, [25.0, 75.0])
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 1e-12:
            return np.zeros(values.shape, dtype=bool)
        lower = q1 - float(threshold) * iqr
        upper = q3 + float(threshold) * iqr
        outliers = np.zeros(values.shape, dtype=bool)
        outliers[finite] = (values[finite] < lower) | (values[finite] > upper)
        return outliers
    raise ValueError(f"Unsupported HWM outlier method: {method}")

def filter_hwm_outliers_single_scenario(
    results_df: pd.DataFrame,
    method: str = "mad",
    threshold: float = 3.5,
    group_by: str = "comid",
    min_samples: int = 5,
) -> pd.DataFrame:
    """
    Apply the FHS HWM outlier filter to one interpolated scenario table.

    The filter operates on flood depth (`WaterSurfaceElev_m - Elev`) and can
    run globally or per COMID. Rows that lose their scenario values are
    dropped from the returned DataFrame.
    """
    filtered_df = results_df.copy()
    group_by = str(group_by).lower()
    if group_by not in {"global", "comid"}:
        raise ValueError("group_by must be either 'global' or 'comid'.")

    if "Elev" not in filtered_df.columns or "WaterSurfaceElev_m" not in filtered_df.columns:
        return filtered_df

    elev = filtered_df["Elev"].to_numpy(dtype=np.float64, copy=False)
    wse = filtered_df["WaterSurfaceElev_m"].to_numpy(dtype=np.float64, copy=False)
    valid_depth = np.isfinite(wse) & np.isfinite(elev)
    depth = wse - elev
    scenario_mask = np.zeros(len(filtered_df), dtype=bool)

    if group_by == "global":
        candidate_mask = valid_depth
        if np.count_nonzero(candidate_mask) >= int(min_samples):
            local_mask = _compute_outlier_mask(depth[candidate_mask], method=method, threshold=threshold)
            idx = np.where(candidate_mask)[0]
            scenario_mask[idx[local_mask]] = True
    else:
        comids = filtered_df["COMID"].to_numpy()
        for comid in pd.unique(comids):
            group_mask = (comids == comid) & valid_depth
            if np.count_nonzero(group_mask) < int(min_samples):
                continue
            local_mask = _compute_outlier_mask(depth[group_mask], method=method, threshold=threshold)
            idx = np.where(group_mask)[0]
            scenario_mask[idx[local_mask]] = True

    if np.any(scenario_mask):
        cols_to_nan = ["WaterSurfaceElev_m"]
        for candidate in ("TopWidth_m", "Velocity_mps", "Flow"):
            if candidate in filtered_df.columns:
                cols_to_nan.append(candidate)
        filtered_df.loc[scenario_mask, cols_to_nan] = np.nan

    keep_rows = filtered_df["WaterSurfaceElev_m"].notna()
    return filtered_df.loc[keep_rows].copy()

def read_curve2flood_table(table_filename: str) -> pd.DataFrame:
    """
    Read a CSV/TSV/parquet table used by Curve2Flood and FHS helper paths.
    """
    if not table_filename:
        raise ValueError("A table filename is required.")
    lower_name = table_filename.lower()
    if lower_name.endswith(".parquet"):
        return pd.read_parquet(table_filename, engine="fastparquet")
    if lower_name.endswith(".tsv"):
        return pd.read_csv(table_filename, sep="\t")
    return pd.read_csv(table_filename)


def _coerce_fhs_metadata_columns(
    table_df: pd.DataFrame,
    table_filename: str,
    one_based_rc: bool = False,
) -> pd.DataFrame:
    """
    Normalize COMID/row/col/elevation/slope/angle metadata into the shape used
    by the FHS point-generation helpers.
    """
    if table_df.empty:
        return pd.DataFrame(columns=["COMID", "Row", "Col", "Elev", "Slope", "XS_Angle"])

    id_col = find_id_column(table_df.columns, table_filename)
    required = {id_col, "Row", "Col"}
    missing_required = required - set(table_df.columns)
    if missing_required:
        raise ValueError(f"{Path(table_filename).name} is missing required columns: {sorted(missing_required)}")

    # Normalize the source table into the compact metadata layout used by the
    # MPI scenario builders and metadata backfill step.
    work_df = table_df.copy()
    work_df["COMID"] = pd.to_numeric(work_df[id_col], errors="coerce")
    work_df["Row"] = pd.to_numeric(work_df["Row"], errors="coerce")
    work_df["Col"] = pd.to_numeric(work_df["Col"], errors="coerce")

    if "Elev" in work_df.columns:
        work_df["Elev"] = pd.to_numeric(work_df["Elev"], errors="coerce")
    elif "DEM_Elev" in work_df.columns:
        work_df["Elev"] = pd.to_numeric(work_df["DEM_Elev"], errors="coerce")
    elif "BaseElev" in work_df.columns:
        work_df["Elev"] = pd.to_numeric(work_df["BaseElev"], errors="coerce")
    else:
        work_df["Elev"] = np.nan

    if "Slope" in work_df.columns:
        work_df["Slope"] = pd.to_numeric(work_df["Slope"], errors="coerce")
    else:
        work_df["Slope"] = np.nan

    if "XS_Angle" in work_df.columns:
        work_df["XS_Angle"] = pd.to_numeric(work_df["XS_Angle"], errors="coerce")
    else:
        work_df["XS_Angle"] = np.nan

    work_df = work_df.dropna(subset=["COMID", "Row", "Col"]).copy()
    if work_df.empty:
        return pd.DataFrame(columns=["COMID", "Row", "Col", "Elev", "Slope", "XS_Angle"])

    work_df["COMID"] = work_df["COMID"].astype(np.int64)
    work_df["Row"] = work_df["Row"].astype(np.int64)
    work_df["Col"] = work_df["Col"].astype(np.int64)
    if one_based_rc:
        work_df["Row"] -= 1
        work_df["Col"] -= 1

    return work_df[["COMID", "Row", "Col", "Elev", "Slope", "XS_Angle"]]


def merge_missing_fhs_metadata(
    scenario_df: pd.DataFrame,
    metadata_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Fill missing `Elev`, `Slope`, and `XS_Angle` values in a scenario table from
    a secondary metadata table keyed by COMID/Row/Col.
    """
    if scenario_df.empty or metadata_df is None or metadata_df.empty:
        return scenario_df

    # Join on the stable spatial key so missing elevation, slope, and cross-
    # section angle values can be borrowed from an alternate table.
    key_cols = ["COMID", "Row", "Col"]
    rhs = metadata_df.drop_duplicates(subset=key_cols)[key_cols + ["Elev", "Slope", "XS_Angle"]].rename(
        columns={
            "Elev": "Elev_metadata",
            "Slope": "Slope_metadata",
            "XS_Angle": "XS_Angle_metadata",
        }
    )
    merged = scenario_df.merge(rhs, on=key_cols, how="left")
    for col in ("Elev", "Slope", "XS_Angle"):
        fallback_col = f"{col}_metadata"
        if col not in merged.columns:
            merged[col] = merged[fallback_col]
        else:
            merged[col] = merged[col].where(pd.notna(merged[col]), merged[fallback_col])
        merged = merged.drop(columns=[fallback_col])
    return merged


def build_fhs_scenario_dataframe_from_vdt(
    vdt_database_filename: str,
    comid_unique_flow: dict,
    one_based_rc: bool = False,
) -> pd.DataFrame:
    """
    Build a single-scenario HWM-style table directly from a VDT database and a
    COMID->flow mapping, following the interpolation behavior of
    `FHS_FloodMapper_AllInOne.py`.

    Returns a DataFrame with one row per interpolated centerline point and the
    columns needed by the FHS point-expansion and flood-mapping steps.
    """
    if not vdt_database_filename:
        raise ValueError("A VDT database is required for FHS VDT point generation.")

    vdt_df = read_curve2flood_table(vdt_database_filename)
    if vdt_df.empty:
        raise ValueError("The VDT database is empty.")

    q_cols, v_cols, t_cols, wse_cols = get_curve_columns(vdt_df.columns)
    work_df = _coerce_fhs_metadata_columns(vdt_df, vdt_database_filename, one_based_rc=one_based_rc)
    flow_series = work_df["COMID"].map(comid_unique_flow)
    work_df["Flow"] = pd.to_numeric(flow_series, errors="coerce")
    work_df = work_df.dropna(subset=["Flow"]).copy()
    if work_df.empty:
        return pd.DataFrame(columns=[
            "COMID", "Row", "Col", "Elev", "Slope", "XS_Angle",
            "Flow", "Velocity_mps", "TopWidth_m", "WaterSurfaceElev_m",
        ])

    # Interpolate the VDT vectors at the scenario discharge for each stream
    # cell that participates in the current event.
    q_arr = vdt_df.loc[work_df.index, q_cols].to_numpy(dtype=np.float64, copy=False)
    v_arr = vdt_df.loc[work_df.index, v_cols].to_numpy(dtype=np.float64, copy=False)
    t_arr = vdt_df.loc[work_df.index, t_cols].to_numpy(dtype=np.float64, copy=False)
    wse_arr = vdt_df.loc[work_df.index, wse_cols].to_numpy(dtype=np.float64, copy=False)
    target_q_arr = work_df["Flow"].to_numpy(dtype=np.float64, copy=False)

    out_v = np.full(len(work_df), np.nan, dtype=np.float64)
    out_t = np.full(len(work_df), np.nan, dtype=np.float64)
    out_wse = np.full(len(work_df), np.nan, dtype=np.float64)

    for i in range(len(work_df)):
        if not np.isfinite(target_q_arr[i]):
            continue
        vi, ti, wi = clean_and_interp_one_row(target_q_arr[i], q_arr[i, :], v_arr[i, :], t_arr[i, :], wse_arr[i, :])
        out_v[i] = vi
        out_t[i] = ti
        out_wse[i] = wi

    work_df["Velocity_mps"] = out_v
    work_df["TopWidth_m"] = out_t
    work_df["WaterSurfaceElev_m"] = out_wse
    work_df = work_df.dropna(subset=["TopWidth_m", "WaterSurfaceElev_m"]).copy()
    return work_df[[
        "COMID", "Row", "Col", "Elev", "Slope", "XS_Angle",
        "Flow", "Velocity_mps", "TopWidth_m", "WaterSurfaceElev_m",
    ]]


def build_fhs_scenario_dataframe_from_curve_file(
    curve_param_filename: str,
    comid_unique_flow: dict,
    tw_multfact: float,
    one_based_rc: bool = False,
) -> pd.DataFrame:
    """
    Build a single-scenario FHS point table from a Curve2Flood curve-parameter
    file.

    Supported curve-file layouts:
    - legacy coefficient tables with `depth_*`, `tw_*`, and `vel_*` columns
    - ARC reach-average outputs that also carry `q_*`, `v_*`, `t_*`, and
      `wse_*` columns

    When both VDT and curve files exist, `multi_point_interpolation(...)`
    continues to prefer the VDT interpolation path because that is the original
    FHS workflow. This builder is used when the curve file is the only source of
    usable scenario points or when it carries the only available `XS_Angle`
    values.
    """
    if not curve_param_filename:
        raise ValueError("A curve-parameter file is required for curve-driven FHS point generation.")

    curve_df = read_curve2flood_table(curve_param_filename)
    if curve_df.empty:
        raise ValueError("The curve-parameter file is empty.")

    work_df = _coerce_fhs_metadata_columns(curve_df, curve_param_filename, one_based_rc=one_based_rc)
    flow_series = work_df["COMID"].map(comid_unique_flow)
    work_df["Flow"] = pd.to_numeric(flow_series, errors="coerce")
    work_df = work_df.dropna(subset=["Flow"]).copy()
    if work_df.empty:
        return pd.DataFrame(columns=[
            "COMID", "Row", "Col", "Elev", "Slope", "XS_Angle",
            "Flow", "Velocity_mps", "TopWidth_m", "WaterSurfaceElev_m",
        ])

    curve_df = curve_df.loc[work_df.index].copy()

    q_cols = [c for c in curve_df.columns if str(c).startswith("q_")]
    v_cols = [c for c in curve_df.columns if str(c).startswith("v_")]
    t_cols = [c for c in curve_df.columns if str(c).startswith("t_")]
    wse_cols = [c for c in curve_df.columns if str(c).startswith("wse_")]
    has_vdt_style_curve = bool(q_cols and v_cols and t_cols and wse_cols)

    if has_vdt_style_curve:
        # Some curve files already contain VDT-style hydraulic vectors, so
        # route them through the same interpolation logic as a true VDT table.
        q_cols, v_cols, t_cols, wse_cols = get_curve_columns(curve_df.columns)
        q_arr = curve_df[q_cols].to_numpy(dtype=np.float64, copy=False)
        v_arr = curve_df[v_cols].to_numpy(dtype=np.float64, copy=False)
        t_arr = curve_df[t_cols].to_numpy(dtype=np.float64, copy=False)
        wse_arr = curve_df[wse_cols].to_numpy(dtype=np.float64, copy=False)
        target_q_arr = work_df["Flow"].to_numpy(dtype=np.float64, copy=False)

        out_v = np.full(len(work_df), np.nan, dtype=np.float64)
        out_t = np.full(len(work_df), np.nan, dtype=np.float64)
        out_wse = np.full(len(work_df), np.nan, dtype=np.float64)

        for i in range(len(work_df)):
            if not np.isfinite(target_q_arr[i]):
                continue
            vi, ti, wi = clean_and_interp_one_row(target_q_arr[i], q_arr[i, :], v_arr[i, :], t_arr[i, :], wse_arr[i, :])
            out_v[i] = vi
            out_t[i] = ti
            out_wse[i] = wi

        work_df["Velocity_mps"] = out_v
        work_df["TopWidth_m"] = out_t
        work_df["WaterSurfaceElev_m"] = out_wse
        work_df = work_df.dropna(subset=["TopWidth_m", "WaterSurfaceElev_m"]).copy()
        return work_df[[
            "COMID", "Row", "Col", "Elev", "Slope", "XS_Angle",
            "Flow", "Velocity_mps", "TopWidth_m", "WaterSurfaceElev_m",
        ]]

    required_curve_cols = {"depth_a", "depth_b", "tw_a", "tw_b", "vel_a", "vel_b"}
    missing_curve_cols = required_curve_cols - set(curve_df.columns)
    if missing_curve_cols:
        raise ValueError(
            f"{Path(curve_param_filename).name} is missing curve columns {sorted(missing_curve_cols)} "
            "and does not expose VDT-style q_/v_/t_/wse_ fields."
        )

    for col in required_curve_cols:
        curve_df[col] = pd.to_numeric(curve_df[col], errors="coerce")
    if "BaseElev" in curve_df.columns:
        curve_df["BaseElev"] = pd.to_numeric(curve_df["BaseElev"], errors="coerce")
    elif "Elev" in curve_df.columns:
        curve_df["BaseElev"] = pd.to_numeric(curve_df["Elev"], errors="coerce")
    elif "DEM_Elev" in curve_df.columns:
        curve_df["BaseElev"] = pd.to_numeric(curve_df["DEM_Elev"], errors="coerce")
    else:
        curve_df["BaseElev"] = np.nan

    # Otherwise rebuild one hydraulic state per row from the power-law curve
    # coefficients at the current event flow.
    curve_df["Flow"] = work_df["Flow"]
    curve_df["Depth"] = curve_df["depth_a"] * curve_df["Flow"] ** curve_df["depth_b"]
    curve_df["TopWidth"] = curve_df["tw_a"] * curve_df["Flow"] ** curve_df["tw_b"]
    curve_df["Velocity"] = curve_df["vel_a"] * curve_df["Flow"] ** curve_df["vel_b"]
    curve_df = curve_df[curve_df["Depth"] > 0].copy()
    curve_df = curve_df[curve_df["TopWidth"] > 0].copy()
    curve_df["WSE"] = curve_df["Depth"] + curve_df["BaseElev"]

    if curve_df.empty:
        return pd.DataFrame(columns=[
            "COMID", "Row", "Col", "Elev", "Slope", "XS_Angle",
            "Flow", "Velocity_mps", "TopWidth_m", "WaterSurfaceElev_m",
        ])

    curve_df = curve_df.groupby("COMID", group_keys=False)[curve_df.columns].apply(filter_outliers)
    if curve_df.empty:
        return pd.DataFrame(columns=[
            "COMID", "Row", "Col", "Elev", "Slope", "XS_Angle",
            "Flow", "Velocity_mps", "TopWidth_m", "WaterSurfaceElev_m",
        ])

    if "QBaseflow" in curve_df.columns:
        qbase = pd.to_numeric(curve_df["QBaseflow"], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        tw_scale = compute_tw_multfact_scale(
            curve_df["Flow"].to_numpy(dtype=np.float32, copy=False),
            qbase,
        )
    else:
        tw_scale = np.ones(len(curve_df), dtype=np.float32)

    curve_df["TopWidth"] = curve_df["TopWidth"].to_numpy(dtype=np.float32, copy=False) * (float(tw_multfact) * tw_scale)
    work_df = work_df.loc[curve_df.index].copy()
    work_df["Velocity_mps"] = curve_df["Velocity"].to_numpy(dtype=np.float64, copy=False)
    work_df["TopWidth_m"] = curve_df["TopWidth"].to_numpy(dtype=np.float64, copy=False)
    work_df["WaterSurfaceElev_m"] = curve_df["WSE"].to_numpy(dtype=np.float64, copy=False)
    return work_df[[
        "COMID", "Row", "Col", "Elev", "Slope", "XS_Angle",
        "Flow", "Velocity_mps", "TopWidth_m", "WaterSurfaceElev_m",
    ]]

def sample_dem_elevation_at_local_xy(dem_array: np.ndarray, x: float, y: float, dx: float, dy: float) -> float:
    """
    Sample the DEM using local metric coordinates generated by `_grid_xy_from_rc`.
    """
    col = int(np.floor(float(x) / float(dx)))
    row = int(np.floor(-float(y) / float(dy)))
    if row < 0 or col < 0 or row >= dem_array.shape[0] or col >= dem_array.shape[1]:
        return np.nan
    elev = float(dem_array[row, col])
    return elev if np.isfinite(elev) else np.nan

def _normalize_xs_angle(xs_angle: float) -> float:
    xs_angle = float(xs_angle) % np.pi
    if xs_angle < 0:
        xs_angle += np.pi
    return xs_angle

def _get_xs_unit_vector_in_local_grid(xs_angle: float, dx: float, dy: float) -> tuple[float, float]:
    """
    Convert an FHS/VDT `XS_Angle` to a unit vector on Curve2Flood's local
    metric grid.
    """
    xs_angle = _normalize_xs_angle(xs_angle)
    dc = np.cos(xs_angle)
    dr = np.sin(xs_angle)
    vx = float(dx) * float(dc)
    vy = -float(dy) * float(dr)
    norm = float(np.hypot(vx, vy))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("Could not compute a valid cross-section direction vector.")
    return vx / norm, vy / norm

def create_fhs_points_from_scenario_dataframe(
    results_df: pd.DataFrame,
    dem_array: np.ndarray,
    dx: float,
    dy: float,
    topwidth_threshold_m: float | None = None,
    xs_point_spacing_m: float = 100.0,
) -> dict[str, np.ndarray]:
    """
    Convert one FHS-style scenario DataFrame into point arrays for
    `create_fhs_flood_map_from_points`.

    This ports the point-generation behavior of `create_geojson_features(...)`
    into an array-native form suited to Curve2Flood. Centerline points are
    always included. If `topwidth_threshold_m` is provided and `XS_Angle`
    values are available, cross-section extension points are added on both
    sides until the water surface is blocked by terrain or the half-topwidth is
    exhausted.
    """
    if results_df.empty:
        empty_float = np.empty(0, dtype=np.float32)
        empty_int = np.empty(0, dtype=np.int32)
        empty_obj = np.empty(0, dtype=object)
        return {
            "rows": empty_int,
            "cols": empty_int,
            "x": empty_float,
            "y": empty_float,
            "wse": empty_float,
            "topwidth": empty_float,
            "dem": empty_float,
            "ids": empty_int,
            "types": empty_obj,
            "slope": empty_float,
        }

    rows = results_df["Row"].to_numpy(dtype=np.int32, copy=False)
    cols = results_df["Col"].to_numpy(dtype=np.int32, copy=False)
    x, y = _grid_xy_from_rc(rows, cols, dx, dy)
    center_dem = results_df["Elev"].to_numpy(dtype=np.float32, copy=False)
    if np.all(~np.isfinite(center_dem)):
        center_dem = np.full(len(results_df), np.nan, dtype=np.float32)
        inside = (rows >= 0) & (rows < dem_array.shape[0]) & (cols >= 0) & (cols < dem_array.shape[1])
        center_dem[inside] = dem_array[rows[inside], cols[inside]].astype(np.float32)

    # Start with one centerline point per row in the scenario table. Optional
    # cross-section extension points are appended below.
    point_data = {
        "rows": rows.astype(np.int32),
        "cols": cols.astype(np.int32),
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "wse": results_df["WaterSurfaceElev_m"].to_numpy(dtype=np.float32, copy=False),
        "topwidth": results_df["TopWidth_m"].to_numpy(dtype=np.float32, copy=False),
        "dem": center_dem.astype(np.float32),
        "ids": results_df["COMID"].to_numpy(dtype=np.int32, copy=False),
        "types": np.full(len(results_df), "Centerline", dtype=object),
        "slope": results_df["Slope"].to_numpy(dtype=np.float32, copy=False) if "Slope" in results_df.columns else np.full(len(results_df), np.nan, dtype=np.float32),
    }

    if topwidth_threshold_m is None or "XS_Angle" not in results_df.columns:
        return point_data

    extra_rows = []
    extra_cols = []
    extra_x = []
    extra_y = []
    extra_wse = []
    extra_topwidth = []
    extra_dem = []
    extra_ids = []
    extra_types = []
    extra_slope = []

    for _, row in results_df.iterrows():
        # March outward from the centerline along the cross section until the
        # DEM blocks the water surface or the half-top-width is exhausted.
        xs_angle = row["XS_Angle"]
        topwidth_value = row["TopWidth_m"]
        scenario_wse = row["WaterSurfaceElev_m"]
        if not np.isfinite(xs_angle) or not np.isfinite(topwidth_value) or not np.isfinite(scenario_wse):
            continue
        if float(topwidth_value) <= float(topwidth_threshold_m):
            continue

        center_x, center_y = _grid_xy_from_rc(
            np.asarray([int(row["Row"])], dtype=np.int32),
            np.asarray([int(row["Col"])], dtype=np.int32),
            dx,
            dy,
        )
        ux, uy = _get_xs_unit_vector_in_local_grid(float(xs_angle), dx, dy)
        half_width = 0.5 * float(topwidth_value)
        offset_m = float(xs_point_spacing_m)
        side_blocked = {"positive": False, "negative": False}

        while offset_m <= half_width + 1e-9:
            for side_sign, side_label in ((1.0, "positive"), (-1.0, "negative")):
                if side_blocked[side_label]:
                    continue

                px = float(center_x[0] + side_sign * offset_m * ux)
                py = float(center_y[0] + side_sign * offset_m * uy)
                point_elev = sample_dem_elevation_at_local_xy(dem_array, px, py, dx, dy)
                if not np.isfinite(point_elev):
                    side_blocked[side_label] = True
                    continue
                if float(scenario_wse) <= float(point_elev):
                    side_blocked[side_label] = True
                    continue
                if float(scenario_wse) - float(point_elev) > 20.0:
                    LOG.warning(
                        f"Large depth ({float(scenario_wse) - float(point_elev):.2f} m) at COMID {int(row['COMID'])}, "
                        f"{side_label} side, offset {offset_m:.2f} m."
                    )

                extra_rows.append(-1)
                extra_cols.append(-1)
                extra_x.append(px)
                extra_y.append(py)
                extra_wse.append(float(scenario_wse))
                extra_topwidth.append(float(topwidth_value))
                extra_dem.append(float(point_elev))
                extra_ids.append(int(row["COMID"]))
                extra_types.append("XS_Extension")
                extra_slope.append(float(row["Slope"]) if "Slope" in row and np.isfinite(row["Slope"]) else np.nan)

            if all(side_blocked.values()):
                break
            offset_m += float(xs_point_spacing_m)

    if not extra_x:
        return point_data

    for key, arr in {
        "rows": np.asarray(extra_rows, dtype=np.int32),
        "cols": np.asarray(extra_cols, dtype=np.int32),
        "x": np.asarray(extra_x, dtype=np.float32),
        "y": np.asarray(extra_y, dtype=np.float32),
        "wse": np.asarray(extra_wse, dtype=np.float32),
        "topwidth": np.asarray(extra_topwidth, dtype=np.float32),
        "dem": np.asarray(extra_dem, dtype=np.float32),
        "ids": np.asarray(extra_ids, dtype=np.int32),
        "types": np.asarray(extra_types, dtype=object),
        "slope": np.asarray(extra_slope, dtype=np.float32),
    }.items():
        point_data[key] = np.concatenate((point_data[key], arr))

    return point_data

def multi_point_interpolation(
    E: np.ndarray,
    B: np.ndarray,
    RR: np.ndarray,
    CC: np.ndarray,
    T_Rast: np.ndarray,
    W_Rast: np.ndarray,
    S_Rast: np.ndarray | None,
    COMID_Unique_TW,
    COMID_Unique_Depth,
    COMID_Unique_Flow: dict,
    CurveParamFileName: str,
    VDTDatabaseFileName: str,
    TW_MultFact: float,
    dx: float,
    dy: float,
    TopWidthPlausibleLimit: float,
    Set_Depth: float,
    flood_vdt_cells: bool,
    mapper_options: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str]:
    """
    Curve2Flood-side wrapper for the FHS FloodMapper workflow.

    This function ports the FHS end-to-end logic that is relevant inside
    Curve2Flood's per-flow event loop:

    1. Optionally rebuild scenario HWM-style centerline points directly from the
       VDT database and the current COMID->flow mapping.
    2. Optionally filter HWM outliers using the FHS MAD/IQR depth filter.
    3. Optionally expand points across the cross section using `XS_Angle`,
       top width, and DEM blocking.
    4. Generate a flood raster via FHS inverse-distance interpolation,
       corridor/anchor buffering, smoothing, and connectivity enforcement.

    Parameters
    ----------
    E, B, RR, CC, T_Rast, W_Rast, S_Rast : ndarray
        The current Curve2Flood padded DEM/stream arrays and per-cell hydraulic
        rasters for the active flow event.
    COMID_Unique_TW, COMID_Unique_Depth : mapping
        Per-COMID fallback top width and depth used when a VDT-driven point set
        cannot be built.
    COMID_Unique_Flow : dict
        Current event COMID->flow mapping.
    CurveParamFileName, VDTDatabaseFileName : str
        Optional curve and VDT source tables. This function prefers the original
        FHS VDT->HWM interpolation path when the VDT database is available.
        Missing `XS_Angle`, `Slope`, or `Elev` values can be backfilled from the
        other table when possible.
    TW_MultFact : float
        Curve2Flood top-width multiplier. This is required so curve-file driven
        scenario generation produces the same scaled top widths as the rest of
        the run.
    dx, dy : float
        Cell spacing in meters on the local working grid.
    TopWidthPlausibleLimit, Set_Depth : float
        Curve2Flood fixed-depth controls used by the fallback centerline-point
        builder when VDT-driven point generation is unavailable.
    flood_vdt_cells : bool
        If True, centerline cells are forced into the returned flood mask.
    mapper_options : dict, optional
        Optional FHS tuning parameters. Supported keys:
        `topwidth_threshold_m`, `xs_point_spacing_m`,
        `remove_hwm_outliers`, `hwm_outlier_method`,
        `hwm_outlier_threshold`, `hwm_outlier_group_by`,
        `hwm_outlier_min_samples`, `corridor_buffer_m`,
        `anchor_buffer_m`, `use_topwidth_buffers`,
        `corridor_topwidth_factor`, `anchor_topwidth_factor`,
        `connectivity`, `k`, `power`, `max_distance_m`,
        `use_topwidth_max_distance`, `maxdist_topwidth_factor`,
        `min_topwidth_m`, `max_topwidth_m`, `fallback_topwidth_m`,
        `smooth_sigma_pixels`, `apply_wse_sanity_filter`,
        `wse_sanity_tolerance_m`, `fast_mode`, and `permanent_water_mask`.

    Returns
    -------
    tuple
        `(flood, depth, slope, stats_message)` for the current flow event.

    Notes
    -----
    The wrapper prefers rebuilding points from the VDT because that most
    closely matches `FHS_FloodMapper_AllInOne.py`. If VDT metadata is missing,
    it is backfilled from the curve file when possible. If a usable VDT table
    is not available, the function builds the scenario table from the curve
    file directly before falling back to the currently-available Curve2Flood
    centerline rasters.
    """
    options = {
        "topwidth_threshold_m": 200.0,
        "xs_point_spacing_m": 100.0,
        "remove_hwm_outliers": True,
        "hwm_outlier_method": "mad",
        "hwm_outlier_threshold": 3.5,
        "hwm_outlier_group_by": "comid",
        "hwm_outlier_min_samples": 5,
        "corridor_buffer_m": 500.0,
        "anchor_buffer_m": 5.0,
        "use_topwidth_buffers": True,
        "corridor_topwidth_factor": 1.0,
        "anchor_topwidth_factor": 0.1,
        "connectivity": 4,
        "k": 12,
        "power": 2.0,
        "max_distance_m": 1000.0,
        "use_topwidth_max_distance": True,
        "maxdist_topwidth_factor": 1.0,
        "min_topwidth_m": 5.0,
        "max_topwidth_m": 500.0,
        "fallback_topwidth_m": 20.0,
        "smooth_sigma_pixels": 0.25,
        "apply_wse_sanity_filter": True,
        "wse_sanity_tolerance_m": 0.0,
        "fast_mode": False,
        "permanent_water_mask": None,
        "one_based_vdt_rc": False,
    }
    if mapper_options:
        options.update({k: v for k, v in mapper_options.items() if v is not None})

    point_data = None
    stats_message = "curve2flood: no scenario points were generated"
    scenario_builder_errors = []

    if Set_Depth <= 0.0:
        # Collect metadata tables up front so any missing Elev, Slope, or
        # XS_Angle fields can be backfilled after a scenario is built.
        metadata_candidates = []
        if VDTDatabaseFileName:
            try:
                metadata_candidates.append(
                    _coerce_fhs_metadata_columns(
                        read_curve2flood_table(VDTDatabaseFileName),
                        VDTDatabaseFileName,
                        one_based_rc=bool(options["one_based_vdt_rc"]),
                    )
                )
            except Exception as ex:
                scenario_builder_errors.append(f"VDT metadata unavailable: {ex}")
        if CurveParamFileName:
            try:
                metadata_candidates.append(
                    _coerce_fhs_metadata_columns(
                        read_curve2flood_table(CurveParamFileName),
                        CurveParamFileName,
                        one_based_rc=bool(options["one_based_vdt_rc"]),
                    )
                )
            except Exception as ex:
                scenario_builder_errors.append(f"Curve metadata unavailable: {ex}")

        scenario_builders = []
        if VDTDatabaseFileName:
            scenario_builders.append((
                "VDT database",
                lambda: build_fhs_scenario_dataframe_from_vdt(
                    vdt_database_filename=VDTDatabaseFileName,
                    comid_unique_flow=COMID_Unique_Flow,
                    one_based_rc=bool(options["one_based_vdt_rc"]),
                ),
            ))
        if CurveParamFileName:
            scenario_builders.append((
                "curve parameter file",
                lambda: build_fhs_scenario_dataframe_from_curve_file(
                    curve_param_filename=CurveParamFileName,
                    comid_unique_flow=COMID_Unique_Flow,
                    tw_multfact=TW_MultFact,
                    one_based_rc=bool(options["one_based_vdt_rc"]),
                ),
            ))

        for source_name, scenario_builder in scenario_builders:
            try:
                # Prefer the VDT-based hydraulic reconstruction, then fall back
                # to the curve file if it is the only viable source.
                scenario_df = scenario_builder()
                for metadata_df in metadata_candidates:
                    scenario_df = merge_missing_fhs_metadata(scenario_df, metadata_df)
                if options["remove_hwm_outliers"]:
                    scenario_df = filter_hwm_outliers_single_scenario(
                        scenario_df,
                        method=str(options["hwm_outlier_method"]),
                        threshold=float(options["hwm_outlier_threshold"]),
                        group_by=str(options["hwm_outlier_group_by"]),
                        min_samples=int(options["hwm_outlier_min_samples"]),
                    )
                if scenario_df.empty:
                    continue
                point_data = create_fhs_points_from_scenario_dataframe(
                    results_df=scenario_df,
                    dem_array=E[1:-1, 1:-1],
                    dx=dx,
                    dy=dy,
                    topwidth_threshold_m=options["topwidth_threshold_m"],
                    xs_point_spacing_m=float(options["xs_point_spacing_m"]),
                )
                break
            except Exception as ex:
                scenario_builder_errors.append(f"{source_name} scenario generation failed: {ex}")

        if point_data is None and scenario_builder_errors:
            LOG.warning(
                "FHS scenario point generation failed; falling back to centerline-only points. Reasons: "
                + " | ".join(scenario_builder_errors)
            )

    if point_data is None:
        # Last resort: use the centerline hydraulics already present inside
        # Curve2Flood instead of an externally rebuilt FHS scenario table.
        point_data = build_fhs_points_from_curve2flood_inputs(
            RR=RR,
            CC=CC,
            T_Rast=T_Rast,
            W_Rast=W_Rast,
            S_Rast=S_Rast,
            E=E,
            B=B,
            COMID_Unique_TW=dict(COMID_Unique_TW),
            COMID_Unique_Depth=dict(COMID_Unique_Depth),
            TopWidthPlausibleLimit=TopWidthPlausibleLimit,
            Set_Depth=Set_Depth,
            dx=dx,
            dy=dy,
        )

    # Once the point cloud is ready, the remaining work is the lower-level FHS
    # raster engine: buffer, interpolate WSE, then derive flood/depth/slope.
    result = create_fhs_flood_map_from_points(
        dem=E[1:-1, 1:-1],
        point_rows=point_data["rows"],
        point_cols=point_data["cols"],
        point_x=point_data["x"],
        point_y=point_data["y"],
        point_wse=point_data["wse"],
        point_topwidth=point_data["topwidth"],
        point_ids=point_data["ids"],
        point_dem=point_data["dem"],
        point_types=point_data["types"],
        point_slope=point_data["slope"] if S_Rast is not None else None,
        dx=dx,
        dy=dy,
        prefix="curve2flood",
        corridor_buffer_m=float(options["corridor_buffer_m"]),
        anchor_buffer_m=float(options["anchor_buffer_m"]),
        use_topwidth_buffers=bool(options["use_topwidth_buffers"]),
        corridor_topwidth_factor=float(options["corridor_topwidth_factor"]),
        anchor_topwidth_factor=float(options["anchor_topwidth_factor"]),
        connectivity=int(options["connectivity"]),
        k=int(options["k"]),
        power=float(options["power"]),
        max_distance_m=float(options["max_distance_m"]),
        use_topwidth_max_distance=bool(options["use_topwidth_max_distance"]),
        maxdist_topwidth_factor=float(options["maxdist_topwidth_factor"]),
        min_topwidth_m=float(options["min_topwidth_m"]),
        max_topwidth_m=float(options["max_topwidth_m"]),
        fallback_topwidth_m=float(options["fallback_topwidth_m"]),
        smooth_sigma_pixels=float(options["smooth_sigma_pixels"]),
        apply_wse_sanity_filter=bool(options["apply_wse_sanity_filter"]),
        wse_sanity_tolerance_m=float(options["wse_sanity_tolerance_m"]),
        permanent_water_mask=options["permanent_water_mask"],
        fast_mode=bool(options["fast_mode"]),
    )
    stats_message = result["stats_message"]

    flood = result["flood"]
    depth = result["depth"]
    slope = result["slope"]

    if flood_vdt_cells:
        # Keep the original centerline cells wet even if interpolation or
        # buffering would otherwise omit some of them.
        valid_rows = point_data["rows"] >= 0
        valid_cols = point_data["cols"] >= 0
        valid_centerline = valid_rows & valid_cols
        flood[point_data["rows"][valid_centerline], point_data["cols"][valid_centerline]] = 1

    return flood, depth, slope, stats_message

@njit("float32[:](float32)", cache=True, parallel=True)
def create_gaussian_kernel_1d(sigma):
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    center = kernel_size // 2

    kernel = np.empty(kernel_size, dtype=np.float32)

    for i in prange(kernel_size):
        x = i - center
        kernel[i] = np.exp(- (x**2) / (2.0 * sigma**2))

    sum_val = np.sum(kernel)
    kernel /= sum_val

    return kernel

@njit("float32[:, :](float32[:, :], float32[:])", cache=True)
def convolve_rows(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    nrows, ncols = image.shape
    klen = len(kernel)
    pad = klen // 2
    output = np.empty_like(image)

    for r in range(nrows):
        for c in range(ncols):
            acc = 0.0
            weight_sum = 0.0
            c_start = max(0, c - pad)
            c_end = min(ncols, c + pad + 1)
            k_start = pad - (c - c_start)

            for k in range(c_end - c_start):
                val = image[r, c_start + k]
                if val <= -9998.0:
                    w = kernel[k_start + k]
                    acc += val * w
                    weight_sum += w

            output[r, c] = acc / weight_sum if weight_sum > 0 else image[r, c]
    return output

@njit("float32[:, :](float32[:, :], float32[:])", cache=True, parallel=True)
def convolve_cols(image, kernel):
    nrows, ncols = image.shape
    klen = len(kernel)
    pad = klen // 2
    output = np.zeros_like(image)

    for c in prange(ncols):
        for r in range(nrows):
            acc = 0.0
            weight_sum = 0.0
            r_start = max(0, r - pad)
            r_end = min(nrows, r + pad + 1)
            k_start = pad - (r - r_start)

            for k in range(r_end - r_start):
                val = image[r_start + k, c]
                if val != -9999.0:
                    acc += val * kernel[k_start + k]
                    weight_sum += kernel[k_start + k]

            output[r, c] = acc / weight_sum if weight_sum > 0 else image[r, c]
    return output

@njit("float32[:, :](float32[:, :], float32)", cache=True)
def gaussian_blur_separable(image, sigma):
    kernel = create_gaussian_kernel_1d(sigma)
    blurred = convolve_rows(image, kernel)
    blurred = convolve_cols(blurred, kernel)
    return blurred


@njit(cache=True, parallel=True)
def spread_Bathy(
    nrows: int,
    ncols: int,
    WeightBox: np.ndarray,
    TW_for_WeightBox_ElipseMask: int,
    Bathy: np.ndarray,
    ARBathyMask: np.ndarray
):

    # Arrays to accumulate weighted sums
    bathy_times_weight = np.zeros_like(Bathy, dtype=np.float32)
    total_weight       = np.zeros_like(Bathy, dtype=np.float32)

    # 3) For each valid bathy cell, spread its value using WeightBox
    tw = TW_for_WeightBox_ElipseMask
    valid_donor = (Bathy > -98.99) & (ARBathyMask == 1)

    for r in prange(1, nrows + 1):
        for c in range(1, ncols + 1):
            if ARBathyMask[r, c] != 1:
                continue

            # window in Bathy space (same pattern as CreateSimpleFloodMap)
            r_min = max(r - tw, 1)
            r_max = min(r + tw + 1, nrows + 1)
            c_min = max(c - tw, 1)
            c_max = min(c + tw + 1, ncols + 1)

            # corresponding window in WeightBox (centered on [tw, tw])
            w_r_min = tw - (r - r_min)
            w_c_min = tw - (c - c_min)

            acc_val = 0.0
            acc_w = 0.0

            for rr in range(r_min, r_max):
                wr = w_r_min + (rr - r_min)
                for cc in range(c_min, c_max):
                    if not valid_donor[rr, cc]:
                        continue
                    wc = w_c_min + (cc - c_min)
                    w = WeightBox[wr, wc]

                    acc_val += Bathy[rr, cc] * w
                    acc_w += w

            bathy_times_weight[r, c] = acc_val
            total_weight[r, c] = acc_w

    return bathy_times_weight, total_weight

def Create_Topobathy_Dataset(
    E: np.ndarray,
    nrows: int,
    ncols: int,
    WeightBox: np.ndarray,
    TW_for_WeightBox_ElipseMask: int,
    Bathy: np.ndarray,
    ARBathyMask: np.ndarray,
    Bathy_Use_Banks: bool
):
    """
    Fill and smooth bathymetry using the same WeightBox kernel
    used in CreateSimpleFloodMap.

    E, Bathy, ARBathyMask are all (nrows+2, ncols+2).
    """
    # ------------------------------------------------------------
    # 1) PRE-CLEANUP: Outside ARBathyMask, Bathy = DEM BEFORE weighting
    # ------------------------------------------------------------
    mask = (Bathy < -98.99) & (ARBathyMask == 1)
    Bathy[mask] = E[mask]

    # 2) Identify valid bathy donors inside the water mask
    #    (same nodata threshold as before: > -98.99)
    bathy_times_weight, total_weight = spread_Bathy(nrows, ncols, WeightBox, TW_for_WeightBox_ElipseMask, Bathy, ARBathyMask)

    # 5) Start from original Bathy, and fill only where Bathy was invalid
    filled = Bathy.copy()

    invalid = (Bathy <= -98.99) | np.isnan(Bathy)

    use_weight = invalid & (total_weight > 1e-10)
    use_dem    = invalid & ~use_weight

    filled[use_weight] = bathy_times_weight[use_weight] / total_weight[use_weight]
    filled[use_dem] = E[use_dem]

    # 6) Optional extra smoothing (you can keep or weaken this)
    sigma_value = 1.0
    filled = gaussian_blur_separable(filled, sigma=sigma_value)

    # 7) Outside the AR bathy mask, always use DEM
    mask = ARBathyMask != 1
    filled[mask] = E[mask]

    # 8) Final safety net: any remaining bad values from DEM
    mask = (filled <= -98.99) | (filled < -9998.0) | np.isnan(filled)
    filled[mask] = E[mask]

    # 9) Honor Bathy_Use_Banks: keep bathy from being above DEM if requested
    if Bathy_Use_Banks == False:
        np.minimum(filled, E, out=filled)

    # 10) Return interior (arrays are padded by 1)
    return filled[1:nrows+1, 1:ncols+1]

def Calculate_Depth_TopWidth_TWMax_Velocity(E, CurveParamFileName, VDTDatabaseFileName, COMID_Unique_Flow, COMID_Unique, Q_Fraction, T_Rast, W_Rast, S_Rast, TW_MultFact, TopWidthPlausibleLimit, dx, dy, Set_Depth, quiet, fast_vdt, linkno_to_twlimit=None):    # Initialize all dictionaries
    COMID_Unique_TW = {}
    COMID_Unique_Depth = {}
    COMID_Unique_Velocity = {}
    COMID_Unique_WSE_Mean = {}
    COMID_Unique_WSE_Std = {}
    
    if Set_Depth>0.0:
        # Initialize all to -9999
        COMID_Unique_TW = {}
        COMID_Unique_Depth = {}
        for comid in COMID_Unique:
            # Overwrite with limits if they are > 0
            if TopWidthPlausibleLimit > 0:
                COMID_Unique_TW[comid] = TopWidthPlausibleLimit
            else:
                
                COMID_Unique_TW[comid] = -9999.0
            if Set_Depth > 0:
                COMID_Unique_Depth[comid] = Set_Depth
            else:
                COMID_Unique_Depth[comid] = -9999.0

        TopWidthMax = TopWidthPlausibleLimit 
    #Mike switched to default to VDT Database instead of Curve.  We can change this in the future.
    elif len(VDTDatabaseFileName)>1:
        (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, T_Rast, W_Rast, S_Rast) = Calculate_TW_D_V_ForEachCOMID_VDTDatabase(E, VDTDatabaseFileName, COMID_Unique_Flow, COMID_Unique, 
                                                                                                                T_Rast, W_Rast, S_Rast, TW_MultFact, fast_vdt, dx, dy)
    elif len(CurveParamFileName)>1:  
        (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, T_Rast, W_Rast, S_Rast) = Calculate_TW_D_V_ForEachCOMID_CurveFile(CurveParamFileName, COMID_Unique_Flow, COMID_Unique,  T_Rast, W_Rast, S_Rast, TW_MultFact, dx, dy)

    LOG.info('Maximum Top Width = ' + str(TopWidthMax))
    
    if not quiet:
        for idx, comid in enumerate(COMID_Unique):
            if COMID_Unique_TW[comid]>TopWidthPlausibleLimit:
                LOG.warning(f"Ignoring {comid}  {COMID_Unique_Flow[comid]}  {COMID_Unique_Flow[comid]*Q_Fraction}  {COMID_Unique_Depth[comid]}  {COMID_Unique_TW[comid]}")  

    if TopWidthPlausibleLimit < TopWidthMax:
        TopWidthMax = TopWidthPlausibleLimit
    
    #Create a Weight Box that can be used for all of the cells
    X_cells = np.round(TopWidthMax/dx,0)
    Y_cells = np.round(TopWidthMax/dy,0)
    TW = int(max(Y_cells,X_cells))  #This is how many cells we will be looking at surrounding our stream cell
    
    return COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, TW, T_Rast, W_Rast, S_Rast

def Curve2Flood(E, B, RR, CC, nrows, ncols, dx, dy, COMID_Unique, 
                COMID_Unique_Flow, CurveParamFileName, VDTDatabaseFileName, 
                Q_Fraction, TopWidthPlausibleLimit, TW_MultFact, WeightBox, 
                TW_for_WeightBox_ElipseMask, LocalFloodOption, Set_Depth, 
                quiet, flood_vdt_cells, T_Rast, W_Rast, S_Rast, OutDEP, 
                flowdir, OutWSE,
                parallel, fast_vdt, mapper: str = "Curve2Flood-Kernel Weighted",
                mapper_options: dict | None = None,
                linkno_to_twlimit=None, linkno_to_order=None, linkno_to_downstream=None):
        
    # Calculate an Average Top Width and Depth for each stream reach.
    # The Depths are purposely adjusted to the DEM that you are using (this addresses issues with using the original or bathy dem)
    (COMID_Unique_TW, COMID_Unique_Depth,  TopWidthMax, 
     TW, T_Rast, W_Rast, S_Rast) = Calculate_Depth_TopWidth_TWMax_Velocity(E, CurveParamFileName, VDTDatabaseFileName, COMID_Unique_Flow, 
                                                                                           COMID_Unique, Q_Fraction, T_Rast, W_Rast, S_Rast, TW_MultFact, 
                                                                                           TopWidthPlausibleLimit, dx, dy, Set_Depth, quiet, fast_vdt, 
                                                                                           linkno_to_twlimit=linkno_to_twlimit)

    #Create a simple Flood Map Data
    search_dist_for_min_elev = 0
    LOG.info('Creating Rough Flood Map Data...')

    # In Curve2Flood(...) just before CreateSimpleFloodMap(...)
    COMID_Unique_TW_Python = COMID_Unique_TW
    COMID_Unique_Depth_Python = COMID_Unique_Depth
    keys_tw  = np.asarray(list(COMID_Unique_TW.keys()), dtype=np.int32)
    vals_tw  = np.asarray(list(COMID_Unique_TW.values()), dtype=np.float32)
    keys_dep = np.asarray(list(COMID_Unique_Depth.keys()), dtype=np.int32)
    vals_dep = np.asarray(list(COMID_Unique_Depth.values()), dtype=np.float32)
    COMID_Unique_TW    = create_numba_dict_from(keys_tw,  vals_tw)
    COMID_Unique_Depth = create_numba_dict_from(keys_dep, vals_dep)


    # we need to compute stream ownership if we are using FLDPLN
    if mapper == "Curve2Flood-FLDPLNpy":
        owner = compute_stream_ownership(B, E)
        owner_order = compute_owner_order(owner, linkno_to_order)
        max_id = int(np.max(B))
        order_lookup = np.zeros(max_id + 1, dtype=np.int32)
        downstream_lookup = np.zeros(max_id + 1, dtype=np.int32)
        if linkno_to_order:
            for key, val in linkno_to_order.items():
                if key <= max_id:
                    order_lookup[int(key)] = int(val)
        if linkno_to_downstream:
            for key, val in linkno_to_downstream.items():
                if key <= max_id:
                    downstream_lookup[int(key)] = int(val)
    else:
        # make empty types for each of these that won't be used 
        owner = np.zeros_like(B, dtype=np.int32)
        owner_order = np.zeros_like(B, dtype=np.int32)
        order_lookup = np.zeros(1, dtype=np.int32)
        downstream_lookup = np.zeros(1, dtype=np.int32)

    if parallel and mapper in {"Curve2Flood-Kernel Weighted", "Curve2Flood-FLDPLNpy"}:
        Flood_array, Depth_array, Slope_array  = CreateSimpleFloodMapParallel(RR, CC, T_Rast, W_Rast, S_Rast, 
                                                                    E, B, flowdir, 
                                                                    nrows, ncols, search_dist_for_min_elev, 
                                                                    TopWidthMax, dx, dy, LocalFloodOption, 
                                                                    COMID_Unique_TW, COMID_Unique_Depth, 
                                                                    WeightBox, 
                                                                    TW_for_WeightBox_ElipseMask, TopWidthPlausibleLimit, 
                                                                    Set_Depth, flood_vdt_cells, OutDEP,
                                                                    mapper)
    elif mapper in {"Curve2Flood-Kernel Weighted", "Curve2Flood-FLDPLNpy"}:
        Flood_array, Depth_array, Slope_array  = CreateSimpleFloodMap(RR, CC, T_Rast, W_Rast, S_Rast, 
                                                                    E, B, flowdir, 
                                                                    nrows, ncols, search_dist_for_min_elev, 
                                                                    TopWidthMax, dx, dy, LocalFloodOption, 
                                                                    COMID_Unique_TW, COMID_Unique_Depth, 
                                                                    WeightBox, 
                                                                    TW_for_WeightBox_ElipseMask, TopWidthPlausibleLimit, 
                                                                    Set_Depth, flood_vdt_cells, OutDEP,
                                                                    mapper, OutWSE)
    elif mapper == "Curve2Flood-Multi-Point Interpolation":
        # this is the entry point for the functionality from FHS_FloodMapper_AllInOne.py that creates a flood map via multi-point inverse distance interpolation and buffering instead of the low-level raster spreading logic in CreateSimpleFloodMap.
        if parallel:
            LOG.warning("The Curve2Flood-Multi-Point Interpolation mapper runs at Python level and ignores the low-level parallel CreateSimpleFloodMapParallel path.")
        Flood_array, Depth_array, Slope_array, stats_message = multi_point_interpolation(
            E=E,
            B=B,
            RR=RR,
            CC=CC,
            T_Rast=T_Rast,
            W_Rast=W_Rast,
            S_Rast=S_Rast,
            COMID_Unique_TW=COMID_Unique_TW_Python,
            COMID_Unique_Depth=COMID_Unique_Depth_Python,
            COMID_Unique_Flow=COMID_Unique_Flow,
            CurveParamFileName=CurveParamFileName,
            VDTDatabaseFileName=VDTDatabaseFileName,
            TW_MultFact=TW_MultFact,
            dx=dx,
            dy=dy,
            TopWidthPlausibleLimit=TopWidthPlausibleLimit,
            Set_Depth=Set_Depth,
            flood_vdt_cells=flood_vdt_cells,
            mapper_options=mapper_options,
        )
        LOG.info(stats_message)
        
    
    return Flood_array, Depth_array, Slope_array
    

def Set_Stream_Locations(nrows: int, ncols: int, infilename: str):
    S = np.full((nrows, ncols), -9999, dtype=np.int32)  #Create an array
    if infilename.endswith('.parquet'):
        df = pd.read_parquet(infilename, columns=['Row', 'Col', 'COMID'], engine='fastparquet')
    else:
        df = pd.read_csv(infilename, usecols=['Row', 'Col', 'COMID'])
        
    S[df['Row'].values, df['Col'].values] = df['COMID'].values

    return S

def Flood_WaterLC_and_STRM_Cells_in_Flood_Map(Flood_Ensemble, S, LC_array, watervalue):
    # (LC, ncols, nrows, cellsize, yll, yur, xll, xur, lat, lc_geotransform, lc_projection) = Read_Raster_GDAL(LandCoverFile)
    '''
    # Streams identified in LC
    LC = np.where(LC == watervalue, 1, 0)   # Mark streams with 1, other areas as 0
    
    # Streams identified in SN
    SN = np.where(S > 0, 1, 0)  # Mark streams with 1, other areas with 0
    
    # Combine LC and SN values
    F = np.where(SN == 1, 1, LC)  # Prioritize SN stream values, else take LC
    
    # Any cell shown as water in the LC or the STRM RAster are now always shown as flooded in the Flood_Ensemble
    Flood_Ensemble = np.where(F > 0, 100, Flood_Ensemble)
    '''
    # Identify streams in LC (1 for water, 0 otherwise)
    LC_array = (LC_array == watervalue).astype(int)

    # Identify streams in SN (1 for streams, 0 otherwise)
    SN = (S > 0).astype(int)

    # Combine LC and SN values, prioritizing SN
    F = SN | LC_array  # Logical OR operation prioritizes SN over LC

    # Update Flood_Ensemble: mark flooded cells (100) wherever F > 0
    Flood_Ensemble[F > 0] = 100

    return Flood_Ensemble

def Flood_Flooded_Cells_in_Map(Array_Ensemble, Flood_Ensemble, eps=0.01):
    """
    This function fills NaN values in the input array (Array_Ensemble) for cells that are marked as flooded
    in the Flood_Ensemble. It uses a nearest-neighbor approach to fill NaN values from the nearest valid
    flooded cells. If any flooded cells remain NaN after this process, they are assigned a small positive
    value (eps). Cells outside the flooded area remain NaN.

    Parameters:

    Array_Ensemble (np.ndarray): 2D array containing depth, water surface elevation, or velocity values, with NaNs for missing data.
    Flood_Ensemble (np.ndarray): 2D array indicating flooded cells (values > 0) and non-flooded cells (values <= 0).
    eps (float): Small positive value to assign to any remaining NaN flooded cells after filling.

    Returns:
    np.ndarray: 2D array with NaNs filled for flooded cells, and small positive values assigned where necessary.
    """

    # valid sources are flooded cells with a real (non-NaN) value
    source_mask =  (Flood_Ensemble > 0) & (~np.isnan(Array_Ensemble))
    # targets are flooded cells that are currently NaN
    target_mask = (Flood_Ensemble > 0) & (np.isnan(Array_Ensemble))

    # Copy to avoid modifying original array (optional)
    filled = Array_Ensemble.copy().astype(np.float32)

    # find the closest depth or WSE values from valid source cells
    if np.any(source_mask):
        # EDT returns indices of the nearest ZERO in the input,
        # so pass the inverse to point toward TRUE source cells.
        _, (ny, nx) = distance_transform_edt(~source_mask, return_indices=True)
        # nearest neighbor values from donors
        nn_vals = Array_Ensemble[ny, nx]
        filled[target_mask] = nn_vals[target_mask]

    # If anything inside Flood_Ensemble is STILL NaN, give it a tiny positive depth
    # (this happens when a flooded blob has zero donors anywhere)
    still_nan = (Flood_Ensemble > 0) & np.isnan(filled)
    if np.any(still_nan):
        filled[still_nan] = eps

    # keep everything outside Flood_Ensemble as NaN
    filled[(Flood_Ensemble <= 0) & (~np.isnan(filled))] = np.nan
    
    return filled    

def remove_cells_not_connected(flood_array: np.ndarray, streams_array: np.ndarray) -> np.ndarray:
    """
    This function identifies connected components in the first raster and retains only those components 
    that are (hydraulically) connected to positive cells in the second raster. Connectivity is defined as 8-connected 
    (including edges and corners).
    Parameters:
    -----------
    flood_array : np.ndarray
        A 2D array representing the first raster. Cells with positive values are considered for connectivity.
    streams_array : np.ndarray
        A 2D array representing the second raster. Positive cells in this raster determine the valid connections.
    Returns:
    --------
    np.ndarray
        A 2D array where cells in `flood_array` that are not connected to positive cells in `streams_array` are removed 
        (set to zero). The output retains the shape of `flood_array`.
    Notes:
    ------
    - Connectivity is determined using an 8-connected neighborhood, which includes horizontal, vertical, 
      and diagonal neighbors.
    - The function uses labeled connected components to identify and filter regions in `flood_array`.
    """

    # Define connectivity (8-connected: edges + corners)
    structure = generate_binary_structure(2, 2)
    
    # Label connected components in flood_array
    labeled_array, num_features = label(flood_array, structure)
    
    # Find labels that connect to positive cells in streams_array
    touching_labels = np.unique(labeled_array[(streams_array > 0) & (labeled_array > 0)])
    
    # Create mask of valid regions
    mask = np.isin(labeled_array, touching_labels)
    
    # Keep only connected chunks in flood_array
    return flood_array * mask

def ReadInputFile(lines,P):
    num_lines = len(lines)
    for i in range(num_lines):
        ls = lines[i].strip().split(None, 1)
        if len(ls)>1 and ls[0]==P:
            if P in ['LocalFloodOption', 'FloodLocalOnly']:
                return True
            if P=='Set_Depth' or P=='FloodSpreader_SpecifyDepth':
                return float(ls[1])
            if P in ['Bathy_Use_Banks', 'Flood_WaterLC_and_STRM_Cells', 'Make_Output_GPKG']:
                if "True" in ls[1]:
                    return True
                elif "False" in ls[1] or ls[1] == '':
                    return False    
            return ls[1]   
    if P=='Q_Fraction':
        return 1.0
    if P=='TopWidthPlausibleLimit':
        return 1000.0
    if P=='TW_MultFact':
        return 3.0
    if P=='Set_Depth' or P=='FloodSpreader_SpecifyDepth':
        return float(-1.1)
    if P in ['LocalFloodOption', 'FloodLocalOnly', 'Bathy_Use_Banks', 'Flood_WaterLC_and_STRM_Cells']:
        return False
    if P == 'Make_Output_GPKG':
        return True    
    if P=='LAND_WaterValue':
        return 80
    if P=='OutDEP' or P=='OutWSE':
        return ""

    return ''

def read_input_file(input_file: str) -> dict[str, str | bool | float]:
    """
    Compatibility wrapper that parses the whitespace-delimited input file into
    a parameter dictionary.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    params: dict[str, str | bool | float] = {}
    for line in lines:
        ls = line.strip().split(None, 1)
        if len(ls) > 1:
            params[ls[0]] = ls[1]

    for flag in ("LocalFloodOption", "FloodLocalOnly"):
        if flag in params:
            params[flag] = str(params[flag]).strip().lower() in ("true", "1", "yes", "y")
    return params

def read_geometry_and_get_linkno_mappings(StrmShp_File: str, 
                                 COMID_Unique, 
                                 StrmOrder_Field, 
                                 TopWidthPlausibleLimit,
                                 Downstream_Link_Field) -> tuple[dict | None, dict | None, dict | None]:
    linkno_to_downstream = None
    linkno_to_twlimit = None
    linkno_to_order = None
    if not StrmShp_File:
        return linkno_to_twlimit, linkno_to_order, linkno_to_downstream
    
    # Read the shapefile
    LOG.info('Opening ' + StrmShp_File)
    Strm_gdf = gpd.read_file(StrmShp_File, use_arrow=True)
    # filter the Strm_gdf to only include the COMIDs in the COMID_Unique array
    Strm_gdf = Strm_gdf[Strm_gdf['LINKNO'].isin(COMID_Unique)]

    # change the TopWidthPlausibleLimit to be weighted by the stream order column in Strm_gdf
    order_field = StrmOrder_Field if StrmOrder_Field in Strm_gdf.columns else 'StrmOrder'
    if order_field in Strm_gdf.columns:
        Strm_gdf['TopWidthPlausibleLimit'] = (Strm_gdf[order_field]/max(Strm_gdf[order_field].values)) * TopWidthPlausibleLimit
        linkno_to_order = dict(zip(Strm_gdf['LINKNO'], Strm_gdf[order_field]))
        # drop all columns in the GDF except for the LINKNO/COMID column and the TopWidthPlausibleLimit column
        Strm_gdf = Strm_gdf[['LINKNO', 'TopWidthPlausibleLimit']]
        # Build a lookup dictionary from the GDF
        linkno_to_twlimit = dict(zip(Strm_gdf['LINKNO'], Strm_gdf['TopWidthPlausibleLimit']))
    else:
        linkno_to_twlimit = None
        linkno_to_order = None

    if Downstream_Link_Field:
        try:
            Strm_gdf_down = gpd.read_file(StrmShp_File, use_arrow=True)
            if Downstream_Link_Field in Strm_gdf_down.columns:
                linkno_to_downstream = dict(zip(Strm_gdf_down['LINKNO'], Strm_gdf_down[Downstream_Link_Field]))
            else:
                linkno_to_downstream = None
            del Strm_gdf_down
        except Exception:
            linkno_to_downstream = None
    
    return linkno_to_twlimit, linkno_to_order, linkno_to_downstream

def create_positive_max_array(array_list: list[np.ndarray]) -> np.ndarray:
    """Create a maximum value array from a list of arrays, ignoring NaNs."""
    # Convert list to stacked array of shape (N, rows, cols)
    arr_stack = np.stack(array_list, axis=0).astype(np.float32)

    # Positive-value mask
    positive_mask = arr_stack > 0.0
    masked_arr = np.ma.array(arr_stack, mask=~positive_mask)

    # Max across the stack, ignoring masked values
    max_vals = masked_arr.max(axis=0).filled(np.nan).astype(np.float32)
    
    return max_vals.astype(np.float32)

def create_depth(
                array_list: list[np.ndarray],
                Flood_Ensemble: np.ndarray,
    ):
    min_depth_m = np.float32(0.01)
    max_vals = create_positive_max_array(array_list)

    # match the flood extent of Flood_Ensemble
    max_vals = Flood_Flooded_Cells_in_Map(max_vals, Flood_Ensemble, eps=0.01)

    # # ------------------------------------------------------------------
    # # Find nearest non-zero stream value for every cell (by index)
    # # ------------------------------------------------------------------
    # streams_flat = streams.ravel()
    # m_flat = max_vals.ravel()  # not strictly needed, but handy if you mask later

    # # Indices of non-zero stream cells (i.e., real stream IDs)
    # nz_idx = np.flatnonzero(streams_flat)
    # if nz_idx.size == 0:
    #     raise ValueError("streams has no non-zero values")

    # # All positions in the flattened grid
    # pos = np.arange(streams_flat.size)

    # # For each position, find where it would be inserted among non-zero indices
    # insert_pos = np.searchsorted(nz_idx, pos)

    # # Candidate nearest indices to the left and right in nz_idx
    # left_idx = np.clip(insert_pos - 1, 0, nz_idx.size - 1)
    # right_idx = np.clip(insert_pos, 0, nz_idx.size - 1)

    # left_nz = nz_idx[left_idx]
    # right_nz = nz_idx[right_idx]

    # # Distances to left and right non-zero positions
    # left_dist = np.abs(pos - left_nz)
    # right_dist = np.abs(pos - right_nz)

    # # Choose nearest non-zero index (tie -> left)
    # nearest_nz = np.where(left_dist <= right_dist, left_nz, right_nz)

    # # Map each cell to the nearest non-zero stream ID
    # nearest_stream_ids_flat = streams_flat[nearest_nz]
    # nearest_stream_ids = nearest_stream_ids_flat.reshape(streams.shape)
    # # ------------------------------------------------------------------

    # # "Wet" cells: where max_vals is finite and not nodata-ish
    # wet = np.isfinite(max_vals) & (max_vals > -9998.0)

    # # SegID_Array: each cell gets the nearest stream ID
    # SegID_Array = nearest_stream_ids.astype(np.int32)
    # SegID_Array[~wet] = 0  # 0 = non-wet / no-stream zone

    # # make an array of the unique stream IDs present in SegID_Array
    # unique_stream_ids = np.unique(SegID_Array)

    # for stream_id in unique_stream_ids:
    #     if stream_id == 0:
    #         continue  # skip background / non-wet

    #     # All cells (stream + floodplain) that belong to this nearest-stream zone
    #     zone_mask = (SegID_Array == stream_id)
    #     if not np.any(zone_mask):
    #         continue

    #     # Only use finite water surface elevations in this zone
    #     zone_mask_finite = zone_mask & np.isfinite(max_vals)
    #     zone_vals = max_vals[zone_mask_finite]
    #     if zone_vals.size < 3:
    #         # not enough data for percentiles
    #         continue

    #     p25 = np.percentile(zone_vals, 25)
    #     p75 = np.percentile(zone_vals, 75)
    #     median_val = np.median(zone_vals)

    #     # Identify outliers in this zone
    #     outliers = (zone_vals < p25) | (zone_vals > p75)
    #     if np.any(outliers):
    #         # Write back only for outliers in this zone
    #         idx_row, idx_col = np.where(zone_mask_finite)
    #         max_vals[idx_row[outliers], idx_col[outliers]] = median_val


    return max_vals


@njit("DictType(int32, float32)(int32[:], float32[:])", cache=True)
def create_numba_dict_from(keys: np.ndarray, values: np.ndarray) -> dict[int, float]:
    # The Dict.empty() constructs a typed dictionary.
    # The key and value typed must be explicitly declared.
    d = Dict.empty(
        key_type=types.int32,
        value_type=types.float32,
    )
    for i in range(len(keys)):
        d[keys[i]] = values[i]
    return d

@njit("DictType(int32, int32)(int32[:], int32[:])", cache=True)
def create_numba_dict_from_int32(keys: np.ndarray, values: np.ndarray) -> dict[int, int]:
    d = Dict.empty(
        key_type=types.int32,
        value_type=types.int32,
    )
    for i in range(len(keys)):
        d[keys[i]] = values[i]
    return d

def create_numba_dict_from_weightbox(keys: np.ndarray, values: list[np.ndarray]) -> dict[int, np.ndarray]:
    d = Dict.empty(
        key_type=types.int32,
        value_type=types.float32[:, :],
    )
    for i in range(len(keys)):
        d[keys[i]] = values[i]
    return d

def create_weightbox_cache_dict() -> dict[int, np.ndarray]:
    return Dict.empty(
        key_type=types.int32,
        value_type=types.float32[:, :],
    )

def create_bathymetry(E: np.ndarray, nrows: int, ncols: int, dem_geotransform: tuple, dem_projection: str, BathyFromARFileName: str, BathyWaterMaskFileName: str, 
                      Flood_Ensemble: np.ndarray, BathyOutputFileName: str, WeightBox: np.ndarray, TW_for_WeightBox_ElipseMask: int, Bathy_Use_Banks: bool, bathymetry_creation_options: list[str] = None):
    LOG.info('Working on Bathymetry')
    ds: gdal.Dataset = gdal.Open(BathyFromARFileName)
    ARBathy = np.full((nrows+2, ncols+2), -9999.0, dtype=np.float32)  #Create an array that is slightly larger than the Bathy Raster Array
    # Read raster as float32
    ARBathy[1:-1, 1:-1] = ds.ReadAsArray().astype(np.float32)
    ds = None

    ARBathyMask = np.zeros((nrows+2,ncols+2), dtype=np.bool_)
    if os.path.exists(BathyWaterMaskFileName):
        ds = gdal.Open(BathyWaterMaskFileName)
        ARBathyMask[1:-1, 1:-1] = ds.ReadAsArray() > 0
        ds = None
    else:
        ARBathyMask[1:-1, 1:-1] = Flood_Ensemble > 0

    ARBathy[np.isnan(ARBathy)] = -99.000  #This converts all nan values to a -99
    ARBathy = ARBathy * ARBathyMask
    ARBathy[ARBathyMask != 1] = -9999.000
    # Bathy = Create_Topobathy_Dataset(RR, CC, E, B, nrows, ncols, WeightBox, TW_for_WeightBox_ElipseMask, Bathy_Yes, ARBathy, ARBathyMask)
    ARBathy = Create_Topobathy_Dataset(E, nrows, ncols, WeightBox, TW_for_WeightBox_ElipseMask, ARBathy, ARBathyMask, Bathy_Use_Banks).astype(np.float32)  # enforce again just in case

    # write the Bathy output raster
    Write_Output_Raster(BathyOutputFileName, ARBathy, ncols, nrows, dem_geotransform, dem_projection, "GTiff", gdal.GDT_Float32, bathymetry_creation_options)

def compute_stream_ownership(B: np.ndarray, E: np.ndarray) -> np.ndarray:
    mask = B > 0
    owner = np.zeros_like(B, dtype=np.int32)
    if not np.any(mask):
        return owner

    _, indices = ndi.distance_transform_edt(~mask, return_indices=True)
    r_idx = indices[0]
    c_idx = indices[1]
    owner = B[r_idx, c_idx].astype(np.int32)
    owner[E <= -9998.0] = 0

    return owner

def compute_owner_order(owner: np.ndarray, linkno_to_order: dict | None) -> np.ndarray:
    if linkno_to_order is None or len(linkno_to_order) == 0:
        return np.zeros_like(owner, dtype=np.int32)

    max_owner = int(np.max(owner))
    order_lookup = np.zeros(max_owner + 1, dtype=np.int32)
    for key, val in linkno_to_order.items():
        if key <= max_owner:
            order_lookup[int(key)] = int(val)

    return order_lookup[owner]

@njit(cache=True)
def compute_owner_mix_zone(owner: np.ndarray, mix_radius_cells: int) -> np.ndarray:
    nrows, ncols = owner.shape
    boundary = np.zeros_like(owner, dtype=np.uint8)

    for r in range(1, nrows - 1):
        for c in range(1, ncols - 1):
            oid = owner[r, c]
            if oid <= 0:
                continue
            if (
                owner[r - 1, c] != oid or
                owner[r + 1, c] != oid or
                owner[r, c - 1] != oid or
                owner[r, c + 1] != oid
            ):
                boundary[r, c] = 1

    if mix_radius_cells <= 0:
        return boundary

    mix_zone = np.zeros_like(owner, dtype=np.uint8)
    for r in range(nrows):
        for c in range(ncols):
            if boundary[r, c] == 0:
                continue
            r_min = r - mix_radius_cells
            if r_min < 0:
                r_min = 0
            r_max = r + mix_radius_cells + 1
            if r_max > nrows:
                r_max = nrows
            c_min = c - mix_radius_cells
            if c_min < 0:
                c_min = 0
            c_max = c + mix_radius_cells + 1
            if c_max > ncols:
                c_max = ncols

            for rr in range(r_min, r_max):
                for cc in range(c_min, c_max):
                    if owner[rr, cc] > 0:
                        mix_zone[rr, cc] = 1

    return mix_zone

def Curve2Flood_MainFunction(input_file: str = None,
                             args: dict = None, 
                             quiet: bool = False,
                             flood_vdt_cells: bool = True,
                             bathymetry_creation_options: list[str] = None,
                             parallel: bool = False,
                             fast_vdt: bool = False):

    """
    Main function that takes runs the flood mapping. If an input file is provided, it reads the parameters from the file.
    If no input file is provided, it uses the parameters from the args dictionary. The args dictionary should contain Python objects
    that can be converted to strings.

    Parameters:
    ----------
    input_file : str
        Path to the input file containing parameters.
    args : dict
        Dictionary of parameters.
    quiet : bool
        If True, suppresses warning messages and progress bars.
    flood_vdt_cells : bool
        If True, includes VDT cells in the flood map.
    bathymetry_creation_options : list[str]
        List of options for bathymetry raster creation.
    parallel : bool
        If True, enables parallel processing for simple floodmap. In tests, this increase flood mapping speed 2x, 
        with a slight change in values (~0.0003% of flooded cells differ).

    """
    if input_file:
        #Open the Input File
        with open(input_file,'r') as infile:
            if input_file.lower().endswith(('.yaml', '.yml')):
                # If it's a YAML file, parse it with PyYAML and convert to the expected list of lines format
                data = yaml.safe_load(infile)
                lines = [f"{key}\t{value}\n" for key, value in data.items()]
            else:
                lines = infile.readlines()
    elif args:
        # Use the args dictionary to extract parameters
        # Hacky way to convert args to lines
        lines = []
        for key, value in args.items():
            lines.append(f"{key}\t{value}\n")
    else:
        LOG.error("No input file or arguments provided.")
        return

    DEM_File = ReadInputFile(lines,'DEM_File')
    STRM_File = ReadInputFile(lines,'Stream_File')
    LAND_File = ReadInputFile(lines,'LU_Raster_SameRes')
    StrmShp_File = ReadInputFile(lines,'StrmShp_File')
    Make_Output_GPKG = ReadInputFile(lines,'Make_Output_GPKG')
    Flow_Direction_File = ReadInputFile(lines,'Flow_Direction_File')
    StrmOrder_Field = ReadInputFile(lines,'StrmOrder_Field')
    Downstream_Link_Field = ReadInputFile(lines,'Downstream_Link_Field')
    Flood_File = ReadInputFile(lines,'OutFLD')
    OutDEP = ReadInputFile(lines,'OutDEP')
    OutWSE = ReadInputFile(lines,'OutWSE')
    OutVEL = ReadInputFile(lines,'OutVEL')
    LU_Manning_n = ReadInputFile(lines,'LU_Manning_n')
    FloodImpact_File = ReadInputFile(lines,'FloodImpact_File')
    FlowFileName: str = ReadInputFile(lines,'COMID_Flow_File') or ReadInputFile(lines,'Comid_Flow_File')
    VDTDatabaseFileName = ReadInputFile(lines,'Print_VDT_Database')
    CurveParamFileName = ReadInputFile(lines,'Print_Curve_File')
    mapper = ReadInputFile(lines,'mapper')
    Q_Fraction = ReadInputFile(lines,'Q_Fraction')
    TopWidthPlausibleLimit = ReadInputFile(lines,'TopWidthPlausibleLimit')
    TW_MultFact = ReadInputFile(lines,'TW_MultFact')
    Set_Depth = ReadInputFile(lines,'Set_Depth')
    Set_Depth = float(Set_Depth)
    Set_Depth2 = ReadInputFile(lines,'FloodSpreader_SpecifyDepth')  #This is the nomenclature for FloodSpreader
    Set_Depth2 = float(Set_Depth2)
    if Set_Depth2>0.0 and Set_Depth<0.0:
        Set_Depth = Set_Depth2
    LocalFloodOption = ReadInputFile(lines,'LocalFloodOption')
    LocalFloodOption2 = ReadInputFile(lines,'FloodLocalOnly')  #This is the nomenclature for FloodSpreader
    if LocalFloodOption2==True:
        LocalFloodOption = True
    BathyWaterMaskFileName = ReadInputFile(lines,'BathyWaterMask')
    BathyFromARFileName = ReadInputFile(lines,'BATHY_Out_File')
    BathyOutputFileName = ReadInputFile(lines,'FSOutBATHY')
    Flood_WaterLC_and_STRM_Cells = ReadInputFile(lines,'Flood_WaterLC_and_STRM_Cells')
    LAND_WaterValue = ReadInputFile(lines,'LAND_WaterValue')
    LAND_WaterValue = int(LAND_WaterValue)
    # Find the True/False variable to use the bank elevations to calculate the depth of the bathymetry estimate
    Bathy_Use_Banks = ReadInputFile(lines,'Bathy_Use_Banks')


    # Some checks
    if not FlowFileName:
        LOG.error("Flow file name is required.")
        return


    if not Flood_File:
        LOG.error("Flood file name is required.")
        return
    
    Q_Fraction = float(Q_Fraction)
    TopWidthPlausibleLimit = float(TopWidthPlausibleLimit)
    TW_MultFact = float(TW_MultFact)

    if mapper in (None, ""):
        # defaults to using the old method, if none is specified
        mapper = "Curve2Flood-Kernel Weighted"

    mapper_options = {
        "topwidth_threshold_m": _parse_optional_float(ReadInputFile(lines, 'MPI_TopWidth_Threshold_m'), 200.0),
        "xs_point_spacing_m": _parse_optional_float(ReadInputFile(lines, 'MPI_XS_Point_Spacing_m'), 100.0),
        "remove_hwm_outliers": _parse_optional_bool(ReadInputFile(lines, 'MPI_Remove_HWM_Outliers'), True),
        "hwm_outlier_method": ReadInputFile(lines, 'MPI_HWM_Outlier_Method') or "mad",
        "hwm_outlier_threshold": _parse_optional_float(ReadInputFile(lines, 'MPI_HWM_Outlier_Threshold'), 3.5),
        "hwm_outlier_group_by": ReadInputFile(lines, 'MPI_HWM_Outlier_Group_By') or "comid",
        "hwm_outlier_min_samples": _parse_optional_int(ReadInputFile(lines, 'MPI_HWM_Outlier_Min_Samples'), 5),
        "corridor_buffer_m": _parse_optional_float(ReadInputFile(lines, 'MPI_Corridor_Buffer_m'), 500.0),
        "anchor_buffer_m": _parse_optional_float(ReadInputFile(lines, 'MPI_Anchor_Buffer_m'), 5.0),
        "use_topwidth_buffers": _parse_optional_bool(ReadInputFile(lines, 'MPI_Use_TopWidth_Buffers'), True),
        "corridor_topwidth_factor": _parse_optional_float(ReadInputFile(lines, 'MPI_Corridor_TopWidth_Factor'), 1.0),
        "anchor_topwidth_factor": _parse_optional_float(ReadInputFile(lines, 'MPI_Anchor_TopWidth_Factor'), 0.1),
        "connectivity": _parse_optional_int(ReadInputFile(lines, 'MPI_Connectivity'), 4),
        "k": _parse_optional_int(ReadInputFile(lines, 'MPI_K'), 12),
        "power": _parse_optional_float(ReadInputFile(lines, 'MPI_Power'), 2.0),
        "max_distance_m": _parse_optional_float(ReadInputFile(lines, 'MPI_Max_Distance_m'), 1000.0),
        "use_topwidth_max_distance": _parse_optional_bool(ReadInputFile(lines, 'MPI_Use_TopWidth_Max_Distance'), True),
        "maxdist_topwidth_factor": _parse_optional_float(ReadInputFile(lines, 'MPI_MaxDist_TopWidth_Factor'), 1.0),
        "min_topwidth_m": _parse_optional_float(ReadInputFile(lines, 'MPI_Min_TopWidth_m'), 5.0),
        "max_topwidth_m": _parse_optional_float(ReadInputFile(lines, 'MPI_Max_TopWidth_m'), 500.0),
        "fallback_topwidth_m": _parse_optional_float(ReadInputFile(lines, 'MPI_Fallback_TopWidth_m'), 20.0),
        "smooth_sigma_pixels": _parse_optional_float(ReadInputFile(lines, 'MPI_Smooth_Sigma_Pixels'), 0.25),
        "apply_wse_sanity_filter": _parse_optional_bool(ReadInputFile(lines, 'MPI_Apply_WSE_Sanity_Filter'), True),
        "wse_sanity_tolerance_m": _parse_optional_float(ReadInputFile(lines, 'MPI_WSE_Sanity_Tolerance_m'), 0.0),
        "fast_mode": _parse_optional_bool(ReadInputFile(lines, 'MPI_Fast_Mode'), False),
        "one_based_vdt_rc": _parse_optional_bool(ReadInputFile(lines, 'MPI_One_Based_VDT_RC'), False),
    }
    if mapper == "Curve2Flood-FLDPLNpy":
        required = {
            "Flow_Direction_File": Flow_Direction_File,
        }
        missing = [name for name, val in required.items() if val is None or val == ""]
        if missing:
            raise ValueError(
                "Curve2Flood-FLDPLNpy is the chosen mapper but required parameters are missing: "
                + ", ".join(missing)
            )

    LOG.info('Opening ' + DEM_File)
    ds: gdal.Dataset = gdal.Open(DEM_File)
    dem_geotransform = ds.GetGeoTransform()
    dem_projection = ds.GetProjection()
    nrows = ds.RasterYSize
    ncols = ds.RasterXSize
    cellsize = dem_geotransform[1]
    yll = dem_geotransform[3] - nrows * abs(dem_geotransform[5])
    yur = dem_geotransform[3]
    
    E = np.full((nrows+2, ncols+2), -9999.0, dtype=np.float32)  #Create an array that is slightly larger than the STRM Raster Array and fill it with -9999.0
    E[1:-1, 1:-1] = ds.ReadAsArray()
    ds = None  

    if Flow_Direction_File and mapper == "Curve2Flood-FLDPLNpy":
        (FDR, fdr_ncols, fdr_nrows, _, _, _, _, _, _, _, _) = Read_Raster_GDAL(Flow_Direction_File)
        if fdr_ncols != ncols or fdr_nrows != nrows:
            LOG.warning("Flow direction raster size does not match DEM; flow-direction gating will be disabled.")
            FlowDir = np.zeros((nrows+2, ncols+2), dtype=np.int32)
        else:
            FlowDir = np.zeros((nrows+2, ncols+2), dtype=np.int32)
            FlowDir[1:-1, 1:-1] = FDR.astype(np.int32)
    else:
        FlowDir = np.empty((3, 3), dtype=np.int32)  # dummy array to avoid errors downstream; won't be used if flow direction file is missing

    LOG.info("Executing flood mapping logic...")

    #Get the Stream Locations from the Curve or VDT File
    if Set_Depth>0.0:
        (S, ncols, nrows, cellsize, yll, yur, xll, xur, lat, dem_geotransform, dem_projection) = Read_Raster_GDAL(STRM_File)
    elif len(VDTDatabaseFileName)>1:
        S = Set_Stream_Locations(nrows, ncols, VDTDatabaseFileName)
    elif len(CurveParamFileName)>1:
        S = Set_Stream_Locations(nrows, ncols, CurveParamFileName)
    else:
        LOG.error('NEED EITHER A CURVE PARAMATER FILE OR A VDT DATABASE FILE')
        return
    
    #Check the coordinate system of the rasters and if they are not in meters or degrees, end with log an error message and stop processing
    unit_aliases = {
        'meter': 'meter',
        'meters': 'meter',
        'metre': 'meter',
        'metres': 'meter',
        'degree': 'degree',
        'degrees': 'degree'
    }
    raster_projections = {
        'DEM': dem_projection,
    }
    for raster_name, raster_projection in raster_projections.items():
        try:
            raster_crs = CRS.from_wkt(raster_projection)
        except Exception as ex:
            LOG.error(f'Unable to parse CRS for {raster_name} raster: {ex}')
            return

        axis_units = {(axis.unit_name or '').strip().lower() for axis in raster_crs.axis_info if axis is not None}
        axis_units.discard('')
        if not axis_units:
            LOG.error(f'Unable to determine CRS units for {raster_name} raster.')
            return

        invalid_units = [u for u in sorted(axis_units) if unit_aliases.get(u) not in {'meter', 'degree'}]
        if invalid_units:
            LOG.error(f'{raster_name} raster CRS units are not meters or degrees: {", ".join(invalid_units)}')
            return
    
    #Get Cellsize Information directly from DEM geotransform.
    #Supports rotated grids by using vector magnitude for each pixel axis.
    dem_cell_size_x = np.hypot(dem_geotransform[1], dem_geotransform[2])
    dem_cell_size_y = np.hypot(dem_geotransform[4], dem_geotransform[5])
    dx, dy, dproject = convert_cell_size(dem_cell_size_x, dem_cell_size_y, yll, yur, dem_projection)
    LOG.info('Cellsize X = ' + str(dx))
    LOG.info('Cellsize Y = ' + str(dy))
    
    #Get list of Unique Stream IDs.  Also find where all the cell values are.
    B = np.zeros((nrows+2,ncols+2), dtype=np.int32)  #Create an array that is slightly larger than the STRM Raster Array
    B[1:-1, 1:-1] = S

    (RR,CC) = np.where(B > 0)

    COMID_Unique = np.unique(B[RR, CC]) # Always sorted
    COMID_Unique = COMID_Unique.astype(int) # Ensure it's treated as integers

    # Open the StrmShp_File if provided
    _tuple = read_geometry_and_get_linkno_mappings(StrmShp_File, COMID_Unique, StrmOrder_Field, TopWidthPlausibleLimit, Downstream_Link_Field)
    linkno_to_twlimit, linkno_to_order, linkno_to_downstream = _tuple
    
    # COMID Flow File Read-in
    LOG.info('Opening and Reading ' + FlowFileName)
    with open(FlowFileName, 'r') as infile:
        header = infile.readline()
        line = infile.readline()
    
    #Order from highest to lowest flow
    ls = line.split(',')
    num_flows = pd.read_csv(FlowFileName, nrows=0).shape[1] - 1  #Subtract 1 for the COMID Column
    LOG.info('Evaluating ' + str(num_flows) + ' Flow Events')
    
    #Creating the initial Weight Box
    LOG.info('Creating the Weight Box')
    TW = int( max( np.round(TopWidthPlausibleLimit/dx,0), np.round(TopWidthPlausibleLimit/dy,0) ) )  #This is how many cells we will be looking at surrounding our stream cell
    TW_for_WeightBox_ElipseMask = TW
    WeightBox = create_weightbox(TW_for_WeightBox_ElipseMask, dx, dy)


    #If you're setting a set-depth value for all streams, just need to simulate one flood event
    if Set_Depth>=0.0:
        num_flows = 1
    
    # Create comid to flow dict
    COMID_Unique_Flow = {}

    # Create initial rasters once, outside the loop
    T_Rast = np.empty((nrows,ncols), np.float32)
    W_Rast = np.empty((nrows,ncols), np.float32)
    if OutVEL:
        S_Rast = np.empty((nrows,ncols), np.float32)
    else:
        S_Rast = None  # will be read later if needed      

    #Go through all the Flow Events
    Flood_Ensemble = np.zeros((nrows, ncols), dtype=np.float16 if num_flows > 1 else np.uint8)  # use uint8 if only one flow, otherwise float16 for percentage calculation
    Depth_array_list = []
    Slope_array_list = []
    for flow_event_num in range(num_flows):
        LOG.info('Working on Flow Event ' + str(flow_event_num))
        # clear out last events values
        T_Rast[:] = -1.0
        W_Rast[:] = -1.0
        #Get an Average Flow rate associated with each stream reach.
        if Set_Depth<=0.000000001:
            COMID_Unique_Flow = FindFlowRateForEachCOMID_Ensemble(FlowFileName, flow_event_num)
        Flood_array_this_flow, Depth_array, Slope_array = Curve2Flood(E, B, RR, CC, nrows, ncols, dx, dy, COMID_Unique, 
                                                            COMID_Unique_Flow, CurveParamFileName, VDTDatabaseFileName,
                                                            Q_Fraction, TopWidthPlausibleLimit, TW_MultFact, WeightBox, 
                                                            TW_for_WeightBox_ElipseMask, LocalFloodOption, Set_Depth, 
                                                            quiet, flood_vdt_cells, T_Rast, W_Rast, S_Rast, OutDEP, 
                                                            FlowDir, OutWSE,
                                                            parallel, fast_vdt, mapper, mapper_options,
                                                            linkno_to_twlimit=linkno_to_twlimit, 
                                                            linkno_to_order=linkno_to_order,
                                                            linkno_to_downstream=linkno_to_downstream)        
        Flood_array_this_flow = remove_cells_not_connected(Flood_array_this_flow, S)
        Flood_Ensemble += Flood_array_this_flow
        Depth_array_list.append(Depth_array)
        Slope_array_list.append(Slope_array)

    # clear some memory by deleting large arrays that are no longer needed
    del T_Rast
    del W_Rast    

    # Combine all flow events into a single ensemble
    #Turn into a percentage
    Flood_Ensemble *= 100
    if num_flows > 1:
        Flood_Ensemble /= num_flows
        Flood_Ensemble = Flood_Ensemble.astype(np.uint8)

    # If Flood_WaterLC_and_STRM_Cells or OutVEL is selected, we need to read in the Land Cover Raster
    if Flood_WaterLC_and_STRM_Cells or OutVEL:
        (LC_array, ncols, nrows, cellsize, yll, yur, xll, xur, lat, lc_geotransform, lc_projection) = Read_Raster_GDAL(LAND_File)

    # If selected, we can also flood cells based on the Land Cover and the Stream Raster
    if Flood_WaterLC_and_STRM_Cells:
        LOG.info('Flooding the Water-Related Land Cover and STRM cells')
        Flood_Ensemble = Flood_WaterLC_and_STRM_Cells_in_Flood_Map(Flood_Ensemble, S, LC_array, LAND_WaterValue)

    # Remove crop circles and other disconnected cells
    Flood_Ensemble = remove_cells_not_connected(Flood_Ensemble, S)

    if Set_Depth < 0:
        LOG.info('Creating Ensemble Flood Map...' + str(Flood_File))

    # Write the output raster
    out_ds: gdal.Dataset = gdal.GetDriverByName("GTiff").Create(Flood_File, ncols, nrows, 1, gdal.GDT_Byte, options=["COMPRESS=LZW", "PREDICTOR=2"])
    out_ds.SetGeoTransform(dem_geotransform)
    out_ds.SetProjection(dem_projection)
    out_ds.WriteArray(Flood_Ensemble)
    out_ds.FlushCache()
    out_ds = None  # Close the dataset to ensure it's written to disk


    if OutDEP or OutWSE or OutVEL:
        Depth_Array = create_depth(Depth_array_list, Flood_Ensemble)

        if OutDEP:
            # Convert NaN ? NoData sentinel
            out_band_data = np.where(np.isnan(Depth_Array), -9999.0, Depth_Array).astype(np.float32)

            # --- Write GeoTIFF ---
            ds: gdal.Dataset = gdal.GetDriverByName("GTiff").Create(
                OutDEP, ncols, nrows, 1, gdal.GDT_Float32,
                options=["COMPRESS=LZW", "PREDICTOR=2", "TILED=YES"]
            )
            if ds is None:
                raise RuntimeError(f"Failed to create output raster: {OutDEP}")

            ds.SetGeoTransform(dem_geotransform)
            ds.SetProjection(dem_projection)

            band: gdal.Band = ds.GetRasterBand(1)
            band.WriteArray(out_band_data)
            band.SetNoDataValue(-9999.0)
            band.FlushCache()
            ds.FlushCache()

            # Cleanup
            band = None
            ds = None

    if OutWSE:
        WSE_Array = np.where((Depth_Array > 0) & (E[1:-1, 1:-1] > -9998.0), Depth_Array+E[1:-1, 1:-1], np.nan).astype(np.float32)

        
        # --- Write GeoTIFF ---
        driver = gdal.GetDriverByName("GTiff")
        ds: gdal.Dataset = driver.Create(
            OutWSE, ncols, nrows, 1, gdal.GDT_Float32,
            options=["COMPRESS=LZW", "PREDICTOR=2", "TILED=YES"]
        )
        if ds is None:
            raise RuntimeError(f"Failed to create output raster: {OutWSE}")

        ds.SetGeoTransform(dem_geotransform)
        ds.SetProjection(dem_projection)

        band = ds.GetRasterBand(1)
        band.WriteArray(WSE_Array)
        band.SetNoDataValue(-9999.0)
        band.FlushCache()
        ds.FlushCache()

        # Cleanup
        band = None
        ds = None
    
    # If the FLDPLN model was used, we need to calculate the Slope_array here
    # We will just use the S_Rast and Flood_array to find the closest slope from S_Rast for each flooded cell
    if mapper == "Curve2Flood-FLDPLNpy" and S_Rast is not None:
        Slope_array = Flood_Flooded_Cells_in_Map(S_Rast.astype(np.float32), Flood_Ensemble.astype(np.uint8), eps=0.0002)
        Slope_array = np.where((Slope_array <= 0), np.nan, Slope_array).astype(np.float32)
        # smooth with Gaussian filter
        sigma_value = 1.00
        Slope_array = gaussian_blur_separable(Slope_array.astype(np.float32), sigma=sigma_value)
        Slope_array_list = [Slope_array]

    if OutVEL:
        # Create the velocity output raster
        create_velocity(OutVEL, Depth_Array, LU_Manning_n, LC_array, Slope_array_list, dem_geotransform, dem_projection, ncols, nrows, Flood_Ensemble, S)


    if StrmShp_File and Make_Output_GPKG:
        # convert the raster to a geodataframe
        flood_gdf = Write_Output_Raster_As_GeoDataFrame(Flood_Ensemble, ncols, nrows, dem_geotransform, dem_projection, gdal.GDT_Byte)
        
        # the name of our flood shapefile
        shp_output_filename = f"{Flood_File[:-4]}.gpkg"

        # save the geodataframe (do not specify the driver, it will be inferred from the file extension)
        flood_gdf.to_file(shp_output_filename)

    if BathyFromARFileName and BathyOutputFileName:
        create_bathymetry(E, nrows, ncols, dem_geotransform, dem_projection, 
                          BathyFromARFileName, BathyWaterMaskFileName, Flood_Ensemble, 
                          BathyOutputFileName, WeightBox, TW_for_WeightBox_ElipseMask, 
                          Bathy_Use_Banks, bathymetry_creation_options)

    # Example of simulated execution
    LOG.info("Flood mapping completed.")

    return
