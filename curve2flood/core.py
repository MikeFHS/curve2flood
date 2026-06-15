#This code looks at a DEM raster to find the dimensions, then writes a script to create a STRM raster.
# built-in imports
import os
import json
from pathlib import Path

# third-party imports
    
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import CRS, Geod
from numba.core import types
from numba.typed import Dict
from numba import njit, prange
from osgeo import gdal, osr, ogr

from shapely.geometry import shape
from scipy.ndimage import label, generate_binary_structure, distance_transform_edt

from curve2flood import LOG
from curve2flood.spreaders import (
    fldpln, create_kernel_weighted_spread_map, filter_outliers, compute_tw_multfact_scale, multi_point_interpolation
)

gdal.UseExceptions()

COMID_FLOW_DICT_TYPE = dict[np.int32, np.float32]

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
        options=["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES"]
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
    T_Rast[curve_df['Row'], curve_df['Col']] = curve_df['TopWidth']
    W_Rast[curve_df['Row'], curve_df['Col']] = curve_df['Depth'] + curve_df['BaseElev']
    if S_Rast is not None:
        S_Rast[curve_df['Row'], curve_df['Col']] = curve_df['Slope']

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
    # vdt_df = vdt_df[vdt_df['COMID'] == 441234660].copy()
    # remove this id: 441302073
    # vdt_df = vdt_df[vdt_df['COMID'] != 441302073].copy()

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
    
    # Map results back to the unique COMID list
    comid_result_df = pd.DataFrame({'COMID': COMID_Unique})
    comid_result_df = comid_result_df.merge(median_values, on='COMID', how='left').fillna(0)
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

def Write_Output_Raster(s_output_filename, raster_data, ncols, nrows, dem_geotransform, dem_projection, s_file_format, s_output_type, creation_options: list[str] = None):   
    o_driver = gdal.GetDriverByName(s_file_format)  #Typically will be a GeoTIFF "GTiff"
    #o_metadata = o_driver.GetMetadata()

    if creation_options is None:
        creation_options = ["COMPRESS=DEFLATE", 'PREDICTOR=2']
    
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

# @njit(cache=True)
def CreateSimpleFloodMap(RR, CC, T_Rast, W_Rast, S_Rast, E, B, 
                         flowdir, nrows, ncols, sd, TW_m, dx, dy, 
                         LocalFloodOption, 
                         COMID_Unique_TW: COMID_FLOW_DICT_TYPE,
                         COMID_Unique_Depth: COMID_FLOW_DICT_TYPE,
                         WeightBox, TW_for_WeightBox_ElipseMask, 
                         TopWidthPlausibleLimit, Set_Depth, flood_vdt_cells, OutDEP,
                         mapper, OutWSE, COMID_Unique):
    if mapper == "Curve2Flood-Kernel Weighted":
        # this is the original curve2flood flood mapping method
        WSE_array, Total_Weight, Slope_Times_Weight = create_kernel_weighted_spread_map(
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
            W_Rast_Padded = np.pad(W_Rast, pad_width=1, mode='constant', constant_values=np.nan)

        # This spreads WSE outward from currently-flooded cells to dry cells as long as
        # neighboring ground elevation E can be overtopped.

        WSE_array = fldpln(W_Rast_Padded, E, flowdir, B, COMID_Unique)
    else:
        raise ValueError(f"Invalid mapper option: {mapper}")

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
    if S_Rast is not None and mapper == "Curve2Flood-Kernel Weighted":
        Slope_divided_by_weight = Slope_Times_Weight / Total_Weight
        Slope_array = np.where((WSE_array > E) & (E > -9998.0), Slope_divided_by_weight, np.nan).astype(np.float32)
        Slope_array = np.where((Slope_array <= 0), 0.0002, Slope_array).astype(np.float32)
        return Flooded_array[1:-1, 1:-1], Depth_array[1:-1, 1:-1], Slope_array[1:-1, 1:-1]


    return Flooded_array[1:-1, 1:-1], Depth_array[1:-1, 1:-1], None




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
    # if Bathy_Use_Banks == False:
    np.minimum(filled, E, out=filled)

    # 10) Return interior (arrays are padded by 1)
    return filled[1:nrows+1, 1:ncols+1]

def Calculate_Depth_TopWidth_TWMax_Velocity(E, CurveParamFileName, VDTDatabaseFileName, COMID_Unique_Flow, COMID_Unique, Q_Fraction, T_Rast, W_Rast, S_Rast, TW_MultFact, TopWidthPlausibleLimit, dx, dy, Set_Depth, quiet, fast_vdt, linkno_to_twlimit=None):    # Initialize all dictionaries
    COMID_Unique_TW = {}
    COMID_Unique_Depth = {}

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
                linkno_to_twlimit=None):
        
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

    if mapper in {"Curve2Flood-Kernel Weighted", "Curve2Flood-FLDPLNpy"}:
        Flood_array, Depth_array, Slope_array  = CreateSimpleFloodMap(RR, CC, T_Rast, W_Rast, S_Rast, 
                                                                    E, B, flowdir, 
                                                                    nrows, ncols, search_dist_for_min_elev, 
                                                                    TopWidthMax, dx, dy, LocalFloodOption, 
                                                                    COMID_Unique_TW, COMID_Unique_Depth, 
                                                                    WeightBox, 
                                                                    TW_for_WeightBox_ElipseMask, TopWidthPlausibleLimit, 
                                                                    Set_Depth, flood_vdt_cells, OutDEP,
                                                                    mapper, OutWSE, COMID_Unique)
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

def read_geometry_and_get_linkno_mappings(StrmShp_File: str, 
                                 COMID_Unique, 
                                 StrmOrder_Field, 
                                 TopWidthPlausibleLimit,) -> tuple[dict | None, dict | None, dict | None]:
    linkno_to_twlimit = None
    if not StrmShp_File:
        return linkno_to_twlimit
    
    # Read the shapefile
    LOG.info('Opening ' + StrmShp_File)
    Strm_gdf = gpd.read_file(StrmShp_File, use_arrow=True)
    # filter the Strm_gdf to only include the COMIDs in the COMID_Unique array
    Strm_gdf = Strm_gdf[Strm_gdf['LINKNO'].isin(COMID_Unique)]

    # change the TopWidthPlausibleLimit to be weighted by the stream order column in Strm_gdf
    order_field = StrmOrder_Field if StrmOrder_Field in Strm_gdf.columns else 'StrmOrder'
    if order_field in Strm_gdf.columns:
        Strm_gdf['TopWidthPlausibleLimit'] = (Strm_gdf[order_field]/max(Strm_gdf[order_field].values)) * TopWidthPlausibleLimit
        # drop all columns in the GDF except for the LINKNO/COMID column and the TopWidthPlausibleLimit column
        Strm_gdf = Strm_gdf[['LINKNO', 'TopWidthPlausibleLimit']]
        # Build a lookup dictionary from the GDF
        linkno_to_twlimit = dict(zip(Strm_gdf['LINKNO'], Strm_gdf['TopWidthPlausibleLimit']))
    else:
        linkno_to_twlimit = None
    
    return linkno_to_twlimit

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
    max_vals = create_positive_max_array(array_list)

    # match the flood extent of Flood_Ensemble
    max_vals = Flood_Flooded_Cells_in_Map(max_vals, Flood_Ensemble, eps=0.01)

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
    Flood_File = ReadInputFile(lines,'OutFLD')
    OutDEP = ReadInputFile(lines,'OutDEP')
    OutWSE = ReadInputFile(lines,'OutWSE')
    OutVEL = ReadInputFile(lines,'OutVEL')
    LU_Manning_n = ReadInputFile(lines,'LU_Manning_n')
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
    linkno_to_twlimit = read_geometry_and_get_linkno_mappings(StrmShp_File, COMID_Unique, StrmOrder_Field, TopWidthPlausibleLimit)

    #Order from highest to lowest flow
    LOG.info('Opening and Reading ' + FlowFileName)
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
        W_Rast[:] = np.nan
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
                                                            linkno_to_twlimit=linkno_to_twlimit)        
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
    out_ds: gdal.Dataset = gdal.GetDriverByName("GTiff").Create(Flood_File, ncols, nrows, 1, gdal.GDT_Byte, options=["COMPRESS=DEFLATE", "PREDICTOR=2"])
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
                options=["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES"]
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
            options=["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES"]
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
