#This code looks at a DEM raster to find the dimensions, then writes a script to create a STRM raster.
# built-in imports
import os
import json
from pathlib import Path

# third-party imports
    
import yaml
import numpy as np
import pandas as pd
import polars as pl
import geopandas as gpd
import polars.selectors as cs
from pyproj import CRS, Geod
from numba.core import types
from numba.typed import Dict
from numba import njit, prange
from osgeo import gdal, osr, ogr

from shapely.geometry import shape
from scipy.ndimage import label, generate_binary_structure, distance_transform_edt, uniform_filter

from curve2flood import LOG
from curve2flood.spreaders import (
    make_flood_map, create_kernel_weighted_spread_map, filter_outliers, compute_tw_multfact_scale, multi_point_interpolation
)

gdal.UseExceptions()

COMID_FLOW_DICT_TYPE = dict[np.int32, np.float32]

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


def create_velocity(params, OutVEL, Depth_Array, LC_array, Slope_array_list,
                           geotransform, projection, ncols, nrows,
                           Flood_Ensemble):
    """
    """

    # find the maximum slope across ensembles, ignoring NaNs when calculating an average with NaNs mixed in
    # Stack them into one 3D array
    Slope_Array = create_positive_max_array(Slope_array_list)

    # Read Manning's n raster ---
    da_input_mannings = read_manning_table(params['LU_Manning_n'], LC_array).astype(np.float32)

    # use Manning's solutions that assumes each pixel is a rectangular channel
    VEL_Array = (1/(da_input_mannings))*((Depth_Array)**(2/3))*((Slope_Array)**(1/2))

    VEL_Array = Flood_Flooded_Cells_in_Map(VEL_Array, Flood_Ensemble, eps=0.01)

    nodata_value = np.nan
    out_band_data = np.where(np.isnan(VEL_Array), nodata_value, VEL_Array).astype(np.float32)

    driver = gdal.GetDriverByName("GTiff")
    ds: gdal.Dataset = driver.Create(
        OutVEL, ncols, nrows, 1, gdal.GDT_Float32,
        options=[f"COMPRESS={params['compression']}", "PREDICTOR=2", "TILED=YES"]
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
        flow_df = pd.read_parquet(FlowFileName)
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
        curve_df = pd.read_parquet(CurveParamFileName)
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
        'TopWidth''median',
        'Depth''median',
        'WSE''median',
        'Velocity''median',
        'Row''first',
        'Col''first'
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

def calculate_interpolated_vdt(
        VDTDatabaseFileName: str,
        COMID_Unique_Flow: dict,
        E_DEM: np.ndarray,
        TW_MultFact: float,
        ) -> pl.DataFrame:
    LOG.debug('\nOpening and Reading ' + VDTDatabaseFileName)
    
    # Read the VDT Database into a DataFrame
    if VDTDatabaseFileName.endswith('.parquet'):
        vdt_df = pl.read_parquet(VDTDatabaseFileName)
    else:
        vdt_df = pl.read_csv(VDTDatabaseFileName)
        
    if vdt_df.is_empty():
        raise ValueError("The VDT Database file is empty or could not be read properly.")
    
    # Add COMID flow information
    comid_flow_df = pl.DataFrame(data=list(COMID_Unique_Flow.items()), schema=['COMID', 'Flow'], orient='row')
    vdt_df = vdt_df.join(comid_flow_df, on='COMID', how='inner')

    # Ensure row and col are integers
    vdt_df = vdt_df.with_columns([
        pl.col('Row').cast(pl.Int32),
        pl.col('Col').cast(pl.Int32)
    ])
    
    # Extract flow, baseflow, elevation, and Slope values
    flow = vdt_df['Flow'].to_numpy().astype(np.float32, copy=False)
    qb = vdt_df['QBaseflow'].to_numpy().astype(np.float32, copy=False)
    e_dem = E_DEM[vdt_df['Row'].to_numpy() + 1, vdt_df['Col'].to_numpy() + 1]

    # Extract flow, TopWidth, and WSE values for interpolation
    flow_values = vdt_df.select(cs.starts_with('q_')).to_numpy().astype(np.float32, copy=False)
    top_width_values = vdt_df.select(cs.starts_with('t_')).to_numpy().astype(np.float32, copy=False)
    wse_values = vdt_df.select(cs.starts_with('wse_')).to_numpy().astype(np.float32, copy=False)
    vel_values = vdt_df.select(cs.starts_with('v_')).to_numpy().astype(np.float32, copy=False)
    elev_values = vdt_df['Elev'].to_numpy().astype(np.float32, copy=False)

    tw_scale = compute_tw_multfact_scale(flow, qb)
    tw_mult_fact = (TW_MultFact * tw_scale).astype(np.float32)
    top_width, depth, wse, velocity = vdt_interpolate(flow, qb, flow_values, top_width_values, elev_values, wse_values, vel_values, e_dem, tw_mult_fact)

    # Add the interpolated values back to the DataFrame
    vdt_df = vdt_df.with_columns([
        pl.Series('TopWidth', top_width),
        pl.Series('Depth', depth),
        pl.Series('WSE', wse),
        pl.Series('Velocity', velocity),
    ])

    # Drop rows with NaN values introduced during outlier removal
    vdt_df = vdt_df.drop_nans(subset=['TopWidth', 'Depth', 'WSE', 'Velocity'])

    # Round the interpolated TopWidth, WSE, and Velocity to 2 decimal places
    vdt_df = vdt_df.with_columns(
        pl.col('TopWidth').round(2),
        pl.col('WSE').round(2),
        pl.col('Velocity').round(2),
    )

    # Apply the outlier filtering function to each COMID group
    cols = ['TopWidth', 'WSE', 'Velocity']
    for col in cols:
        vdt_df = vdt_df.with_columns(
            pl.col(col).quantile(0.01).over("COMID").alias("q01"),
            pl.col(col).quantile(0.99).over("COMID").alias("q99"),
        ).filter(
            pl.col(col).is_between(pl.col("q01"), pl.col("q99"))
        ).drop("q01", "q99")


    return vdt_df
    
def Calculate_TW_D_V_ForEachCOMID_VDTDatabase(E_DEM, VDTDatabaseFileName: str, COMID_Unique_Flow: dict, COMID_Unique, T_Rast, W_Rast, S_Rast, TW_MultFact, dx, dy):    
    vdt_df = calculate_interpolated_vdt(VDTDatabaseFileName, COMID_Unique_Flow, E_DEM, TW_MultFact)
    # Fill T_Rast, W_Rast, and S_Rast
    T_Rast[vdt_df['Row'], vdt_df['Col']] = vdt_df['TopWidth']
    W_Rast[vdt_df['Row'], vdt_df['Col']] = vdt_df['WSE']
    if S_Rast is not None:
        S_Rast[vdt_df['Row'], vdt_df['Col']] = vdt_df['Slope']    
    
    # Calculate median values by COMID
    median_values = vdt_df.group_by('COMID').agg([
        pl.col('TopWidth').median(),
        pl.col('Depth').median(),
        pl.col('WSE').median(),
        pl.col('Velocity').median()
    ])
    
    # Map results back to the unique COMID list
    comid_result_df = pl.DataFrame({'COMID': COMID_Unique})
    comid_result_df = comid_result_df.join(median_values, on='COMID', how='left').fill_nan(0)
    comid_result_df = comid_result_df.with_columns(
        pl.col('COMID').cast(pl.Int32),
        pl.col('TopWidth').cast(pl.Float32),
        pl.col('Depth').cast(pl.Float32),
        pl.col('Velocity').cast(pl.Float32)
    )
    
    # Create dicts
    COMID_Unique_TW = dict(zip(comid_result_df['COMID'].to_numpy(), comid_result_df['TopWidth'].to_numpy()))
    COMID_Unique_Depth = dict(zip(comid_result_df['COMID'].to_numpy(), comid_result_df['Depth'].to_numpy()))

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
        LOG.warning('Raster appears south-up (positive pixel height); flipping to north-up' + str(InRAST_Name))
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

def Write_Output_Raster(s_output_filename, raster_data, ncols, nrows, dem_geotransform, dem_projection, s_file_format, s_output_type, compression, creation_options: list[str] = None):   
    o_driver = gdal.GetDriverByName(s_file_format)  #Typically will be a GeoTIFF "GTiff"
    #o_metadata = o_driver.GetMetadata()

    if creation_options is None:
        creation_options = [f"COMPRESS={compression}", 'PREDICTOR=2']
    
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

def CreateSimpleKernelFloodMap(params: dict, RR, CC, T_Rast, W_Rast, S_Rast, E, B, 
                         nrows, ncols, TW_m, dx, dy, 
                         COMID_Unique_TW: COMID_FLOW_DICT_TYPE,
                         COMID_Unique_Depth: COMID_FLOW_DICT_TYPE,
                         WeightBox, TW_for_WeightBox_ElipseMask, flood_vdt_cells,):
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
        TW_m,
        dx,
        dy,
        params['LocalFloodOption'],
        COMID_Unique_TW,
        COMID_Unique_Depth,
        WeightBox,
        TW_for_WeightBox_ElipseMask,
        params['TopWidthPlausibleLimit'],
        params['Set_Depth'],
    )
    
    # Do not flood cells where WSE is below E and E/WSE are nan values
    mask = (WSE_array > E) & (E > -9998.0)
    Flooded_array = mask.astype(np.uint8, copy=False)

    # Also make sure all the Cells that have Stream are counted as flooded.
    if flood_vdt_cells:
        Flooded_array[RR, CC] = 1

    # Create the Depth array
    if params['OutDEP'] or S_Rast is not None or params['OutWSE']:
        Depth_array = np.where(mask, WSE_array - E, np.nan).astype(np.float32)
    else:
        Depth_array = np.empty((3, 3), dtype=np.float32) # Dummy array if not used

    # if you want, create the slope array
    if S_Rast is not None and params['mapper'] == "Curve2Flood-Kernel Weighted":
        Slope_divided_by_weight = Slope_Times_Weight / Total_Weight
        Slope_array = np.where(mask, Slope_divided_by_weight, np.nan).astype(np.float32)
        Slope_array = np.where((Slope_array <= 0), 0.0002, Slope_array).astype(np.float32)
        return Flooded_array[1:-1, 1:-1], Depth_array[1:-1, 1:-1], Slope_array[1:-1, 1:-1]
    
    return Flooded_array[1:-1, 1:-1], Depth_array[1:-1, 1:-1], None

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

def uniform_smoothing(mask: np.ndarray, values: np.ndarray):
    window_size = 3
    weighted = np.where(mask, values, 0)
    value_sum = uniform_filter(weighted, size=window_size, mode='nearest') * window_size
    count = uniform_filter(mask.astype(np.float32, copy=False), size=window_size, mode='nearest') * window_size
    np.divide(value_sum, count, out=values, where=count > 0)

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
    mask = (Bathy < -98.99) & (ARBathyMask == 0)
    Bathy[mask] = E[mask]

    # 2) Spread Bathy values using the WeightBox kernel, accumulating weighted sums and total weights
    bathy_times_weight, total_weight = spread_Bathy(nrows, ncols, WeightBox, TW_for_WeightBox_ElipseMask, Bathy, ARBathyMask)

    # 3) Start from original Bathy, and fill only where Bathy was invalid
    invalid = (Bathy <= -98.99) | np.isnan(Bathy)

    use_weight = invalid & (total_weight > 1e-10)
    use_dem    = invalid & ~use_weight

    Bathy[use_weight] = bathy_times_weight[use_weight] / total_weight[use_weight]
    Bathy[use_dem] = E[use_dem]

    # 4) Optional extra smoothing (you can keep or weaken this)
    # We smooth using a fun math trick, only within the AR bathy mask, to avoid smoothing the banks in!
    # uniform_smoothing(ARBathyMask, Bathy)

    # 5) Outside the AR bathy mask, always use DEM
    mask = ARBathyMask != 1
    Bathy[mask] = E[mask]

    # 6) Final safety net: any remaining bad values from DEM
    mask = (Bathy <= -98.99) | np.isnan(Bathy)
    Bathy[mask] = E[mask]
    
    # 7) Honor Bathy_Use_Banks: keep bathy from being above DEM if requested
    if not Bathy_Use_Banks:
        np.minimum(Bathy, E, out=Bathy)

def Calculate_Depth_TopWidth_TWMax_Velocity(params: dict, E, COMID_Unique_Flow, COMID_Unique, T_Rast, W_Rast, S_Rast, dx, dy, quiet):    # Initialize all dictionaries
    COMID_Unique_TW = {}
    COMID_Unique_Depth = {}

    Set_Depth = params['Set_Depth']
    TopWidthPlausibleLimit = params['TopWidthPlausibleLimit']
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
    elif params['VDTDatabaseFileName']:
        (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, T_Rast, W_Rast, S_Rast) = Calculate_TW_D_V_ForEachCOMID_VDTDatabase(E, params['VDTDatabaseFileName'], COMID_Unique_Flow, COMID_Unique, 
                                                                                                                T_Rast, W_Rast, S_Rast, params['TW_MultFact'], dx, dy)
    elif params['CurveParamFileName']:  
        (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, T_Rast, W_Rast, S_Rast) = Calculate_TW_D_V_ForEachCOMID_CurveFile(params['CurveParamFileName'], COMID_Unique_Flow, COMID_Unique,  T_Rast, W_Rast, S_Rast, params['TW_MultFact'], dx, dy)

    LOG.info('Maximum Top Width = ' + str(TopWidthMax))
    
    if not quiet:
        for idx, comid in enumerate(COMID_Unique):
            if COMID_Unique_TW[comid]>TopWidthPlausibleLimit:
                LOG.warning(f"Ignoring {comid}  {COMID_Unique_Flow[comid]}  {COMID_Unique_Flow[comid]*params['Q_Fraction']}  {COMID_Unique_Depth[comid]}  {COMID_Unique_TW[comid]}")  

    if TopWidthPlausibleLimit < TopWidthMax:
        TopWidthMax = TopWidthPlausibleLimit
    
    #Create a Weight Box that can be used for all of the cells
    X_cells = np.round(TopWidthMax/dx,0)
    Y_cells = np.round(TopWidthMax/dy,0)
    TW = int(max(Y_cells,X_cells))  #This is how many cells we will be looking at surrounding our stream cell
    
    return COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, TW, T_Rast, W_Rast, S_Rast

def make_fldpln_flood_map(
        params: dict, 
        COMID_Unique_Flow, 
        E, 
        filled_dem,
        stream_info,
        flow_dir,
        fldpln_library,
        streams_gdf,
        ):
    vdt_df = calculate_interpolated_vdt(
        params['VDTDatabaseFileName'],
        COMID_Unique_Flow,
        E,
        params['TW_MultFact']
    )

    wse_array = make_flood_map(
        E[1:-1, 1:-1],
        filled_dem,
        vdt_df,
        flow_dir,
        fldpln_library,
        stream_info,
        streams_gdf,
        max_wse_rise=params['max_wse_rise'],
        median_filter_size=params['FLDPLN_Median_Filter_Size'],
        dof_scale=params['FLDPLN_DoF_Scale'],
        dof_offset=params['FLDPLN_DoF_Offset'],
        missing_fsp_interpolation=params['FLDPLN_Missing_FSP_Interpolation'],
        dof_signal=params['FLDPLN_DoF_Signal'],
        threshold_mode=params['FLDPLN_Threshold_Mode'],
    )

    Flood_array = (wse_array > E[1:-1, 1:-1]).astype(np.uint8)
    Depth_array = np.where(Flood_array, wse_array - E[1:-1, 1:-1], np.nan).astype(np.float32)

    return Flood_array, Depth_array, None  # Slope_array is not computed in this method

def Curve2Flood(params: dict, E, B, RR, CC, nrows, ncols, dx, dy, COMID_Unique, 
                COMID_Unique_Flow, WeightBox, 
                TW_for_WeightBox_ElipseMask, 
                quiet, flood_vdt_cells, T_Rast, W_Rast, S_Rast,
                flowdir,
                filled_dem,
                stream_info,
                fldpln_library,
                streams_gdf,):
    if params['mapper'] == "Curve2Flood-FLDPLNpy":
        return make_fldpln_flood_map(
            params,
            COMID_Unique_Flow,
            E,
            filled_dem,
            stream_info,
            flowdir,
            fldpln_library,
            streams_gdf,
        )

    # Calculate an Average Top Width and Depth for each stream reach.
    # The Depths are purposely adjusted to the DEM that you are using (this addresses issues with using the original or bathy dem)
    (COMID_Unique_TW, COMID_Unique_Depth,  TopWidthMax, 
     TW, T_Rast, W_Rast, S_Rast) = Calculate_Depth_TopWidth_TWMax_Velocity(
         params,E, COMID_Unique_Flow, COMID_Unique,
         T_Rast, W_Rast, S_Rast, 
         dx, dy, quiet)

    #Create a simple Flood Map Data
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

    if params['mapper'] == "Curve2Flood-Kernel Weighted":
        Flood_array, Depth_array, Slope_array  = CreateSimpleKernelFloodMap(params, RR, CC, T_Rast, W_Rast, S_Rast, 
                                                                    E, B, 
                                                                    nrows, ncols, 
                                                                    TopWidthMax, dx, dy, 
                                                                    COMID_Unique_TW, COMID_Unique_Depth, 
                                                                    WeightBox, 
                                                                    TW_for_WeightBox_ElipseMask, 
                                                                    flood_vdt_cells)
    elif params['mapper'] == "Curve2Flood-Multi-Point Interpolation":
        # this is the entry point for the functionality from FHS_FloodMapper_AllInOne.py that creates a flood map via multi-point inverse distance interpolation and buffering instead of the low-level raster spreading logic in CreateSimpleFloodMap.
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
            CurveParamFileName=params['CurveParamFileName'],
            VDTDatabaseFileName=params['VDTDatabaseFileName'],
            TW_MultFact=params['TW_MultFact'],
            dx=dx,
            dy=dy,
            TopWidthPlausibleLimit=params['TopWidthPlausibleLimit'],
            Set_Depth=params['Set_Depth'],
            flood_vdt_cells=flood_vdt_cells,
            mapper_options=params,
        )
        LOG.info(stats_message)
        
    
    return Flood_array, Depth_array, Slope_array
    

def Set_Stream_Locations(nrows: int, ncols: int, infilename: str):
    S = np.full((nrows, ncols), -9999, dtype=np.int32)  #Create an array
    if infilename.endswith('.parquet'):
        df = pd.read_parquet(infilename, columns=['Row', 'Col', 'COMID'])
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

def Flood_Flooded_Cells_in_Map(Array_Ensemble: np.ndarray, Flood_Ensemble: np.ndarray, eps=0.01):
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
    flood_mask = Flood_Ensemble > 0
    nan_mask = np.isnan(Array_Ensemble)
    source_mask =  (flood_mask) & (~nan_mask)
    
    # targets are flooded cells that are currently NaN
    target_mask = (flood_mask) & (nan_mask)

    # Copy to avoid modifying original array (optional)
    filled = Array_Ensemble.astype(np.float32, copy=True)

    # find the closest depth or WSE values from valid source cells
    if np.any(source_mask):
        # EDT returns indices of the nearest ZERO in the input,
        # so pass the inverse to point toward TRUE source cells.
        ny, nx = distance_transform_edt(~source_mask, return_distances=False, return_indices=True)
        # nearest neighbor values from donors
        nn_vals = Array_Ensemble[ny, nx]
        filled[target_mask] = nn_vals[target_mask]

    # If anything inside Flood_Ensemble is STILL NaN, give it a tiny positive depth
    # (this happens when a flooded blob has zero donors anywhere)
    still_nan = (flood_mask) & np.isnan(filled)
    if np.any(still_nan):
        filled[still_nan] = eps

    # keep everything outside Flood_Ensemble as NaN
    filled[Flood_Ensemble <= 0] = np.nan
    
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

def create_positive_max_array(array_list: list[np.ndarray]) -> np.ndarray:
    """Create a maximum value array from a list of arrays, ignoring NaNs."""
    # Convert list to stacked array of shape (N, rows, cols)
    arr_stack = np.stack(array_list, axis=0, dtype=np.float32)
    result = np.nanmax(arr_stack, where=arr_stack > 0, axis=0, initial=-np.inf)
    result[result == -np.inf] = np.nan  # Replace -inf with NaN for cells that had no positive values
    return result

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

def create_bathymetry(params: dict, E: np.ndarray, nrows: int, ncols: int, dem_geotransform: tuple, dem_projection: str,
                      Flood_Ensemble: np.ndarray, WeightBox: np.ndarray, TW_for_WeightBox_ElipseMask: int, bathymetry_creation_options: list[str] = None):
    LOG.info('Working on Bathymetry')
    ds: gdal.Dataset = gdal.Open(params['BathyFromARFileName'])
    ARBathy = np.full((nrows+2, ncols+2), -9999.0, dtype=np.float32)  #Create an array that is slightly larger than the Bathy Raster Array
    # Read raster as float32
    ARBathy[1:-1, 1:-1] = ds.ReadAsArray().astype(np.float32)
    ds = None

    ARBathyMask = np.zeros((nrows+2,ncols+2), dtype=np.bool_)
    if os.path.exists(params['BathyWaterMaskFileName']):
        ds = gdal.Open(params['BathyWaterMaskFileName'])
        ARBathyMask[1:-1, 1:-1] = ds.ReadAsArray() > 0
        ds = None
    else:
        ARBathyMask[1:-1, 1:-1] = Flood_Ensemble > 0

    ARBathy[np.isnan(ARBathy)] = -99.000  #This converts all nan values to a -99
    ARBathy = ARBathy * ARBathyMask
    ARBathy[ARBathyMask != 1] = -9999.000
    Create_Topobathy_Dataset(E, nrows, ncols, WeightBox, TW_for_WeightBox_ElipseMask, ARBathy, ARBathyMask, params['Bathy_Use_Banks'])

    # write the Bathy output raster
    Write_Output_Raster(params['BathyOutputFileName'], ARBathy[1:-1, 1:-1], ncols, nrows, dem_geotransform, dem_projection, "GTiff", gdal.GDT_Float32, params['compression'], bathymetry_creation_options)

def to_bool(value):
    if isinstance(value, str):
        return not value.lower() == 'false'
    return bool(value)

def get_params(input_file: str = None, args: dict = None):
    if input_file:
        #Open the Input File
        with open(input_file,'r') as infile:
            if input_file.lower().endswith(('.yaml', '.yml')):
                # If it's a YAML file, parse it with PyYAML and convert to the expected list of lines format
                data = yaml.safe_load(infile)
            else:
                data = {}
                for line in infile.readlines():
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key_value = line.split(None, 1)
                        if len(key_value) == 2:
                            key, value = key_value
                            data[key] = value
                        elif len(key_value) == 1:
                            data[key_value[0]] = ''
    elif args:
        data = args
    else:
        LOG.error("No input file or arguments provided.")
        return
    
    params = {
        'DEM_File': data.get('DEM_File', ''),
        'STRM_File': data.get('Stream_File', ''),
        'LAND_File': data.get('LU_Raster_SameRes', ''),
        'StrmShp_File': data.get('StrmShp_File', ''),
        'Make_Output_GPKG': to_bool(data.get('Make_Output_GPKG', True)),
        'StrmOrder_Field': data.get('StrmOrder_Field', ''),
        'Flood_File': data.get('OutFLD', ''),
        'OutDEP': data.get('OutDEP', ''),
        'OutWSE': data.get('OutWSE', ''),
        'OutVEL': data.get('OutVEL', ''),
        'LU_Manning_n': data.get('LU_Manning_n', ''),
        'FlowFileName': data.get('COMID_Flow_File', data.get('Comid_Flow_File', '')),
        'VDTDatabaseFileName': data.get('Print_VDT_Database', ''),
        'CurveParamFileName': data.get('Print_Curve_File', ''),
        'mapper': data.get('mapper', "Curve2Flood-Kernel Weighted"),
        'Q_Fraction': float(data.get('Q_Fraction', 1.0)),
        'TopWidthPlausibleLimit': float(data.get('TopWidthPlausibleLimit', 1000.0)),
        'TW_MultFact': float(data.get('TW_MultFact', 3.0)),
        'Set_Depth': min(float(data.get('Set_Depth', -1.1)), float(data.get('FloodSpreader_SpecifyDepth', -1.1))),
        'LocalFloodOption': to_bool(data.get('LocalFloodOption', False)) or to_bool(data.get('FloodLocalOnly', False)),
        'BathyWaterMaskFileName': data.get('BathyWaterMask', ''),
        'BathyFromARFileName': data.get('BATHY_Out_File', ''),
        'BathyOutputFileName': data.get('FSOutBATHY', ''),
        'Flood_WaterLC_and_STRM_Cells': to_bool(data.get('Flood_WaterLC_and_STRM_Cells', False)),
        'LAND_WaterValue': int(data.get('LAND_WaterValue', 80)),
        'Bathy_Use_Banks': to_bool(data.get('Bathy_Use_Banks', False)),

        # FLDPLN inpputs
        'Flow_Direction_File': data.get('Flow_Direction_File', ''),
        'Filled_DEM_File': data.get('Filled_DEM_File', ''),
        'Stream_Info_File': data.get('Stream_Info_File', ''),
        'FLDPLN_Library': data.get('FLDPLN_Library', ''),
        'max_wse_rise': float(data.get('max_wse_rise', 0.01)),
        'FLDPLN_Median_Filter_Size': int(data.get('FLDPLN_Median_Filter_Size', data.get('median_filter_size', 53))),
        'FLDPLN_DoF_Scale': float(data.get('FLDPLN_DoF_Scale', data.get('dof_scale', 1.55))),
        'FLDPLN_DoF_Offset': float(data.get('FLDPLN_DoF_Offset', data.get('dof_offset', -0.1))),
        'FLDPLN_Missing_FSP_Interpolation': data.get('FLDPLN_Missing_FSP_Interpolation', data.get('missing_fsp_interpolation', 'ffill')),
        'FLDPLN_DoF_Signal': data.get('FLDPLN_DoF_Signal', data.get('dof_signal', 'min')),
        'FLDPLN_Threshold_Mode': data.get('FLDPLN_Threshold_Mode', data.get('threshold_mode', 'normal')),

        # Multipoint options
        'topwidth_threshold_m': float(data.get('MPI_TopWidth_Threshold_m', 200.0)),
        'xs_point_spacing_m': float(data.get('MPI_XS_Point_Spacing_m', 100.0)),
        'remove_hwm_outliers': to_bool(data.get('MPI_Remove_HWM_Outliers', True)),
        'hwm_outlier_method': data.get('MPI_HWM_Outlier_Method', 'mad'),
        'hwm_outlier_threshold': float(data.get('MPI_HWM_Outlier_Threshold', 3.5)),
        'hwm_outlier_group_by': data.get('MPI_HWM_Outlier_Group_By', 'comid'),
        'hwm_outlier_min_samples': int(data.get('MPI_HWM_Outlier_Min_Samples', 5)),
        'corridor_buffer_m': float(data.get('MPI_Corridor_Buffer_m', 500.0)),
        'anchor_buffer_m': float(data.get('MPI_Anchor_Buffer_m', 5.0)),
        'use_topwidth_buffers': to_bool(data.get('MPI_Use_TopWidth_Buffers', True)),
        'corridor_topwidth_factor': float(data.get('MPI_Corridor_TopWidth_Factor', 1.0)),
        'anchor_topwidth_factor': float(data.get('MPI_Anchor_TopWidth_Factor', 0.1)),
        'connectivity': int(data.get('MPI_Connectivity', 4)),
        'k': int(data.get('MPI_K', 12)),
        'power': float(data.get('MPI_Power', 2.0)),
        'max_distance_m': float(data.get('MPI_Max_Distance_m', 1000.0)),
        'use_topwidth_max_distance': to_bool(data.get('MPI_Use_TopWidth_Max_Distance', True)),
        'maxdist_topwidth_factor': float(data.get('MPI_MaxDist_TopWidth_Factor', 1.0)),
        'min_topwidth_m': float(data.get('MPI_Min_TopWidth_m', 5.0)),
        'max_topwidth_m': float(data.get('MPI_Max_TopWidth_m', 500.0)),
        'fallback_topwidth_m': float(data.get('MPI_Fallback_TopWidth_m', 20.0)),
        'smooth_sigma_pixels': float(data.get('MPI_Smooth_Sigma_Pixels', 0.25)),
        'apply_wse_sanity_filter': to_bool(data.get('MPI_Apply_WSE_Sanity_Filter', True)),
        'wse_sanity_tolerance_m': float(data.get('MPI_WSE_Sanity_Tolerance_m', 0.0)),
        'fast_mode': to_bool(data.get('MPI_Fast_Mode', False)),
        'one_based_vdt_rc': to_bool(data.get('MPI_One_Based_VDT_RC', False)),

        # Miscellaneous
        "compression": data.get("compression", "LZW"),
    }

    return params

def validate_params(params: dict):
    required_params = []
    if params['Flood_File'] or params['OutDEP'] or params['OutWSE'] or params['OutVEL'] or (params['BathyOutputFileName'] and params['BathyFromARFileName'] and not path_exists(params['BathyWaterMaskFileName'])):
        required_params.append('FlowFileName')

    if params['mapper'] == "Curve2Flood-FLDPLNpy":
        required_params.extend([
            'Flow_Direction_File',
            'Filled_DEM_File',
            'Stream_Info_File',
            'FLDPLN_Library',
        ])
    missing_params = [param for param in required_params if not params.get(param)]
    if missing_params:
        raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

def main_flood_ouputs(
        params: dict, 
        E: np.ndarray, 
        WeightBox: np.ndarray, 
        TW_for_WeightBox_ElipseMask: int, 
        dx: float, 
        dy: float, 
        dem_projection: str,
        dem_geotransform: tuple,
        quiet: bool = False, 
        flood_vdt_cells: bool = False):
    nrows, ncols = E[1:-1, 1:-1].shape

    if params['mapper'] == "Curve2Flood-FLDPLNpy":
        FlowDir = gdal.Open(params['Flow_Direction_File']).ReadAsArray().astype(np.uint8, copy=False)
        if FlowDir.shape != (nrows, ncols):
            LOG.error(f"Flow direction raster dimensions ({FlowDir.shape[1]}x{FlowDir.shape[0]}) do not match DEM dimensions ({ncols}x{nrows}).")
            raise ValueError("Flow direction raster dimensions do not match DEM dimensions.")


        filled_dem: np.ndarray = gdal.Open(params['Filled_DEM_File']).ReadAsArray()
        if filled_dem.shape != (nrows, ncols):
            LOG.error(f"Filled DEM raster dimensions ({filled_dem.shape[1]}x{filled_dem.shape[0]}) do not match DEM dimensions ({ncols}x{nrows}).")
            raise ValueError("Filled DEM raster dimensions do not match DEM dimensions.")
        
        if Path(params['Stream_Info_File']).suffix == '.parquet':
            stream_info: pd.DataFrame = pd.read_parquet(params['Stream_Info_File'])
        else:
            stream_info: pd.DataFrame = pd.read_csv(params['Stream_Info_File'])
        if Path(params['StrmShp_File']).suffix == '.parquet':
            streams_gdf = gpd.read_parquet(params['StrmShp_File'])
        else:
            streams_gdf = gpd.read_file(params['StrmShp_File'], use_arrow=True, ignore_geometry=True)

        if Path(params['FLDPLN_Library']).suffix == '.parquet':
            fldpln_library = pl.scan_parquet(params['FLDPLN_Library'])
        else:
            fldpln_library = pl.scan_csv(params['FLDPLN_Library'])
    else:
        FlowDir = None
        filled_dem = None
        stream_info = None
        streams_gdf = None
        fldpln_library = None

    LOG.info("Executing flood mapping logic...")

    #Get the Stream Locations from the Curve or VDT File
    if params['Set_Depth'] > 0.0:
        (S, ncols, nrows, cellsize, yll, yur, xll, xur, lat, __loader__, _) = Read_Raster_GDAL(params['STRM_File'])
    elif params['VDTDatabaseFileName']:
        S = Set_Stream_Locations(nrows, ncols, params['VDTDatabaseFileName'])
    elif params['CurveParamFileName']:
        S = Set_Stream_Locations(nrows, ncols, params['CurveParamFileName'])
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
    
    #Get list of Unique Stream IDs.  Also find where all the cell values are.
    B = np.zeros((nrows+2,ncols+2), dtype=np.int32)  #Create an array that is slightly larger than the STRM Raster Array
    B[1:-1, 1:-1] = S

    (RR,CC) = np.where(B > 0)

    COMID_Unique = np.unique(B[RR, CC]) # Always sorted
    COMID_Unique = COMID_Unique.astype(int) # Ensure it's treated as integers

    #Order from highest to lowest flow
    LOG.info('Opening and Reading ' + params['FlowFileName'])
    num_flows = pd.read_csv(params['FlowFileName'], nrows=0).shape[1] - 1  #Subtract 1 for the COMID Column
    LOG.info('Evaluating ' + str(num_flows) + ' Flow Events')

    #If you're setting a set-depth value for all streams, just need to simulate one flood event
    if params['Set_Depth'] >= 0.0:
        num_flows = 1
    
    # Create initial rasters once, outside the loop
    if params['mapper'] == "Curve2Flood-FLDPLNpy":
        T_Rast = None
        W_Rast = None
    else:
        T_Rast = np.empty((nrows,ncols), np.float32)
        W_Rast = np.empty((nrows,ncols), np.float32)

    if params['OutVEL']:
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
        if T_Rast is not None:
            T_Rast[:] = -1.0
        if W_Rast is not None:
            W_Rast[:] = np.nan

        #Get an Average Flow rate associated with each stream reach.
        if params['Set_Depth'] <= 0.000000001:
            COMID_Unique_Flow = FindFlowRateForEachCOMID_Ensemble(params['FlowFileName'], flow_event_num)
        Flood_array_this_flow, Depth_array, Slope_array = Curve2Flood(
            params, E, B, RR, CC, nrows, ncols, dx, dy, COMID_Unique, 
            COMID_Unique_Flow, WeightBox, TW_for_WeightBox_ElipseMask, 
            quiet, flood_vdt_cells, T_Rast, W_Rast, S_Rast, FlowDir,
            filled_dem, stream_info, fldpln_library, streams_gdf
            )        
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

    OutVEL = params['OutVEL']
    Flood_WaterLC_and_STRM_Cells = params['Flood_WaterLC_and_STRM_Cells']
    LAND_File = params['LAND_File']
    OutDEP = params['OutDEP']
    OutWSE = params['OutWSE']
    Flood_File = params['Flood_File']

    # If Flood_WaterLC_and_STRM_Cells or OutVEL is selected, we need to read in the Land Cover Raster
    if Flood_WaterLC_and_STRM_Cells or OutVEL:
        (LC_array, ncols, nrows, cellsize, yll, yur, xll, xur, lat, lc_geotransform, lc_projection) = Read_Raster_GDAL(LAND_File)

    # If selected, we can also flood cells based on the Land Cover and the Stream Raster
    if Flood_WaterLC_and_STRM_Cells:
        LOG.info('Flooding the Water-Related Land Cover and STRM cells')
        Flood_Ensemble = Flood_WaterLC_and_STRM_Cells_in_Flood_Map(Flood_Ensemble, S, LC_array, params['LAND_WaterValue'])

    # Remove crop circles and other disconnected cells
    Flood_Ensemble = remove_cells_not_connected(Flood_Ensemble, S)

    if Flood_File:
        if params['Set_Depth'] < 0:
            LOG.info('Creating Ensemble Flood Map...' + str(Flood_File))

        # Write the output raster
        out_ds: gdal.Dataset = gdal.GetDriverByName("GTiff").Create(Flood_File, ncols, nrows, 1, gdal.GDT_Byte, options=[f"COMPRESS={params['compression']}", "PREDICTOR=2"])
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
                options=[f"COMPRESS={params['compression']}", "PREDICTOR=2", "TILED=YES"]
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
            options=[f"COMPRESS={params['compression']}", "PREDICTOR=2", "TILED=YES"]
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
    if params['mapper'] == "Curve2Flood-FLDPLNpy" and S_Rast is not None:
        Slope_array = Flood_Flooded_Cells_in_Map(S_Rast.astype(np.float32), Flood_Ensemble.astype(np.uint8), eps=0.0002)
        mask = Slope_array <= 0
        Slope_array = np.where(mask, np.nan, Slope_array).astype(np.float32)
        uniform_smoothing(mask, Slope_array)
        Slope_array_list = [Slope_array]

    if OutVEL:
        # Create the velocity output raster
        create_velocity(params, OutVEL, Depth_Array, LC_array, Slope_array_list, dem_geotransform, dem_projection, ncols, nrows, Flood_Ensemble)

    if params['StrmShp_File'] and params['Make_Output_GPKG'] and Flood_File:
        # convert the raster to a geodataframe
        flood_gdf = Write_Output_Raster_As_GeoDataFrame(Flood_Ensemble, ncols, nrows, dem_geotransform, dem_projection, gdal.GDT_Byte)
        
        # the name of our flood shapefile
        shp_output_filename = f"{Flood_File[:-4]}.gpkg"

        # save the geodataframe (do not specify the driver, it will be inferred from the file extension)
        flood_gdf.to_file(shp_output_filename)

    return Flood_Ensemble

def path_exists(path: str | os.PathLike) -> bool:
    """Check if a given path exists."""
    return path and Path(path).exists()

def Curve2Flood_MainFunction(input_file: str = None,
                             args: dict = None, 
                             quiet: bool = False,
                             flood_vdt_cells: bool = True,
                             bathymetry_creation_options: list[str] = None,
                             **kwargs):

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
    **kwargs
        Additional keyword arguments, for backwards compatibility with older versions of the function that may have used different parameter names.

    """
    params = get_params(input_file, args)
    validate_params(params)

    LOG.info('Opening ' + params['DEM_File'])
    ds: gdal.Dataset = gdal.Open(params['DEM_File'])
    dem_geotransform = ds.GetGeoTransform()
    dem_projection = ds.GetProjection()
    nrows = ds.RasterYSize
    ncols = ds.RasterXSize
    yll = dem_geotransform[3] - nrows * abs(dem_geotransform[5])
    yur = dem_geotransform[3]
    
    E = np.full((nrows+2, ncols+2), -9999.0, dtype=np.float32)  #Create an array that is slightly larger than the STRM Raster Array and fill it with -9999.0
    E[1:-1, 1:-1] = ds.ReadAsArray()
    ds = None  

    #Get Cellsize Information directly from DEM geotransform.
    #Supports rotated grids by using vector magnitude for each pixel axis.
    dem_cell_size_x = np.hypot(dem_geotransform[1], dem_geotransform[2])
    dem_cell_size_y = np.hypot(dem_geotransform[4], dem_geotransform[5])
    dx, dy, dproject = convert_cell_size(dem_cell_size_x, dem_cell_size_y, yll, yur, dem_projection)
    LOG.info('Cellsize X = ' + str(dx))
    LOG.info('Cellsize Y = ' + str(dy))

    #Creating the initial Weight Box
    LOG.info('Creating the Weight Box')
    TW_for_WeightBox_ElipseMask = int( max( np.round(params['TopWidthPlausibleLimit']/dx,0), np.round(params['TopWidthPlausibleLimit']/dy,0) ) )  #This is how many cells we will be looking at surrounding our stream cell
    if params['mapper'] == "Curve2Flood-FLDPLNpy":
        WeightBox = None
    else:
        WeightBox = create_weightbox(TW_for_WeightBox_ElipseMask, dx, dy)

    Flood_Ensemble = None
    if params['Flood_File'] or params['OutDEP'] or params['OutWSE'] or params['OutVEL'] or (params['BathyOutputFileName'] and params['BathyFromARFileName'] and not path_exists(params['BathyWaterMaskFileName'])):
        Flood_Ensemble = main_flood_ouputs(params, E, WeightBox, TW_for_WeightBox_ElipseMask, dx, dy, dem_projection, dem_geotransform, quiet=quiet, flood_vdt_cells=flood_vdt_cells)

    if params['BathyFromARFileName'] and params['BathyOutputFileName']:
        create_bathymetry(params, E, nrows, ncols, dem_geotransform, dem_projection, 
                           Flood_Ensemble, WeightBox, TW_for_WeightBox_ElipseMask, 
                          bathymetry_creation_options)

    # Example of simulated execution
    LOG.info("Flood mapping completed.")

    return
