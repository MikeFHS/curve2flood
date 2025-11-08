
#This code looks at a DEM raster to find the dimensions, then writes a script to create a STRM raster.
# built-in imports
import gc
import sys
import os
from datetime import datetime
import json

# third-party imports
try:
    import gdal 
    import osr 
    import ogr
    #from gdalconst import GA_ReadOnly
except: 
    from osgeo import gdal
    from osgeo import osr
    from osgeo import ogr
    #from osgeo.gdalconst import GA_ReadOnly

from numba import njit
from numba.core import types
from numba.typed import Dict
    
import numpy as np
import pandas as pd
import geopandas as gpd
# from scipy.interpolate import interp1d
from scipy.ndimage import label, generate_binary_structure, distance_transform_edt
from shapely.geometry import shape

import rasterio
from rasterio.features import rasterize

from curve2flood import LOG

gdal.UseExceptions()



def convert_cell_size(dem_cell_size, dem_lower_left, dem_upper_right):
    """
    Determines the x and y cell sizes based on the geographic location

    Parameters
    ----------
    None. All input data is available in the parent object

    Returns
    -------
    None. All output data is set into the object

    """

    ### Get the cell size ###
    d_lat = np.fabs((dem_lower_left + dem_upper_right) / 2)

    ### Determine if conversion is needed
    if dem_cell_size > 0.5:
        # This indicates that the DEM is projected, so no need to convert from geographic into projected.
        x_cell_size = dem_cell_size
        y_cell_size = dem_cell_size
        projection_conversion_factor = 1

    else:
        # Reprojection from geographic coordinates is needed
        assert d_lat > 1e-16, "Please use lat and long values greater than or equal to 0."

        # Determine the latitude range for the model
        if d_lat >= 0 and d_lat <= 10:
            d_lat_up = 110.61
            d_lat_down = 110.57
            d_lon_up = 109.64
            d_lon_down = 111.32
            d_lat_base = 0.0

        elif d_lat > 10 and d_lat <= 20:
            d_lat_up = 110.7
            d_lat_down = 110.61
            d_lon_up = 104.64
            d_lon_down = 109.64
            d_lat_base = 10.0

        elif d_lat > 20 and d_lat <= 30:
            d_lat_up = 110.85
            d_lat_down = 110.7
            d_lon_up = 96.49
            d_lon_down = 104.65
            d_lat_base = 20.0

        elif d_lat > 30 and d_lat <= 40:
            d_lat_up = 111.03
            d_lat_down = 110.85
            d_lon_up = 85.39
            d_lon_down = 96.49
            d_lat_base = 30.0

        elif d_lat > 40 and d_lat <= 50:
            d_lat_up = 111.23
            d_lat_down = 111.03
            d_lon_up = 71.70
            d_lon_down = 85.39
            d_lat_base = 40.0

        elif d_lat > 50 and d_lat <= 60:
            d_lat_up = 111.41
            d_lat_down = 111.23
            d_lon_up = 55.80
            d_lon_down = 71.70
            d_lat_base = 50.0

        elif d_lat > 60 and d_lat <= 70:
            d_lat_up = 111.56
            d_lat_down = 111.41
            d_lon_up = 38.19
            d_lon_down = 55.80
            d_lat_base = 60.0

        elif d_lat > 70 and d_lat <= 80:
            d_lat_up = 111.66
            d_lat_down = 111.56
            d_lon_up = 19.39
            d_lon_down = 38.19
            d_lat_base = 70.0

        elif d_lat > 80 and d_lat <= 90:
            d_lat_up = 111.69
            d_lat_down = 111.66
            d_lon_up = 0.0
            d_lon_down = 19.39
            d_lat_base = 80.0

        else:
            raise AttributeError('Please use legitimate (0-90) lat and long values.')

        ## Convert the latitude ##
        d_lat_conv = d_lat_down + (d_lat_up - d_lat_down) * (d_lat - d_lat_base) / 10
        y_cell_size = dem_cell_size * d_lat_conv * 1000.0  # Converts from degrees to m

        ## Longitude Conversion ##
        d_lon_conv = d_lon_down + (d_lon_up - d_lon_down) * (d_lat - d_lat_base) / 10
        x_cell_size = dem_cell_size * d_lon_conv * 1000.0  # Converts from degrees to m

        ## Make sure the values are in bounds ##
        if d_lat_conv < d_lat_down or d_lat_conv > d_lat_up or d_lon_conv < d_lon_up or d_lon_conv > d_lon_down:
            raise ArithmeticError("Problem in conversion from geographic to projected coordinates")

        ## Calculate the conversion factor ##
        projection_conversion_factor = 1000.0 * (d_lat_conv + d_lon_conv) / 2.0
    return x_cell_size, y_cell_size, projection_conversion_factor

def FindFlowRateForEachCOMID_Ensemble(FlowFileName: str, flow_event_num: int) -> dict:  
    if FlowFileName.endswith('.parquet'):
        flow_df = pd.read_parquet(FlowFileName)
    else:
        flow_df = pd.read_csv(FlowFileName, usecols=[0, flow_event_num + 1])

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
    return group

def Calculate_TW_D_ForEachCOMID_CurveFile(CurveParamFileName: str, COMID_Unique_Flow: dict, COMID_Unique, T_Rast, W_Rast, TW_MultFact):

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
    curve_df = curve_df[curve_df['Depth']>0]
    curve_df = curve_df[curve_df['TopWidth']>0]
    curve_df['WSE'] = curve_df['Depth'] + curve_df['BaseElev']

    # TO DO!!!  This needs to be redone to focus the bounds based on the COMID. This is taking out too many values (I think)
    # Apply the outlier filtering function to each COMID group
    curve_df = curve_df.groupby('COMID', group_keys=False).apply(filter_outliers)

    # Fill in the T_Rast and W_Rast
    for index, row in curve_df.iterrows():
        T_Rast[int(row['Row']), int(row['Col'])] = row['TopWidth'] * TW_MultFact
        W_Rast[int(row['Row']), int(row['Col'])] = row['Depth'] + row['BaseElev']

    # Calculate median values by COMID
    mean_values = curve_df.groupby('COMID').agg({
        'TopWidth': 'max',
        'Depth': 'mean',
        'WSE': 'mean'
    })
    
    # Map results back to the unique COMID list
    comid_result_df = pd.DataFrame({'COMID': COMID_Unique})
    comid_result_df = comid_result_df.merge(mean_values, on='COMID', how='left').fillna(0)

    # Create dicts
    comid_result_df['COMID'] = comid_result_df['COMID'].astype(np.int32)
    comid_result_df['TopWidth'] = comid_result_df['TopWidth'].astype(np.float32)
    comid_result_df['Depth'] = comid_result_df['Depth'].astype(np.float32)
    
    # Create dicts
    COMID_Unique_TW = comid_result_df.set_index('COMID')['TopWidth'].to_dict()
    COMID_Unique_Depth = comid_result_df.set_index('COMID')['Depth'].to_dict()
    
    # Get the maximum TopWidth for all COMIDs
    TopWidthMax = comid_result_df['TopWidth'].max()

    return (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, T_Rast, W_Rast)

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

@njit("Tuple((float32[:], float32[:], float32[:]))(float32[:], float32[:], float32[:,:], float32[:, :], float32[:], float32[:, :], float64[:], float32)",
      cache=True)
def vdt_interpolate(flow: np.ndarray,
                    qb: np.ndarray, 
                    flow_values: np.ndarray, 
                    top_width_values: np.ndarray,
                    elev_values: np.ndarray,
                    wse_values: np.ndarray,
                    e_dem: np.ndarray,
                    tw_mult_fact: float) -> tuple[np.ndarray, ...]:
    top_width = np.empty_like(flow)
    depth = np.empty_like(flow)
    wse = np.empty_like(flow)
    baseflow_tw = np.empty_like(flow)

    # Loop through each row in the DataFrame, interpolate as needed
    for i in range(len(flow)):
        if flow[i] <= qb[i]:
            # Below baseflow
            top_width[i] = Find_TopWidth_at_Baseflow_when_using_VDT(qb[i], flow_values[i], top_width_values[i])
            depth[i] = 0.001
            wse[i] = elev_values[i]
        elif flow[i] >= flow_values[i][-1]:
            # Above the maximum flow value
            top_width[i] = top_width_values[i][-1]
            wse[i] = wse_values[i][-1]
            depth[i] = wse[i] - e_dem[i]
        else:
            # Interpolate
            wse[i] = interp1d_numba(flow_values[i], wse_values[i], flow[i])
            top_width[i] = interp1d_numba(flow_values[i], top_width_values[i], flow[i])
            wse[i] = max(wse[i], e_dem[i])
            depth[i] = max(wse[i] - e_dem[i], 0.001)

        baseflow_tw[i] = Find_TopWidth_at_Baseflow_when_using_VDT(qb[i], flow_values[i], top_width_values[i])

    # Ensure TopWidth respects baseflow and scale
    top_width = np.maximum(top_width, baseflow_tw) * tw_mult_fact
    
    return top_width, depth, wse

def Calculate_TW_D_ForEachCOMID_VDTDatabase(E_DEM, VDTDatabaseFileName: str, COMID_Unique_Flow: dict, COMID_Unique, T_Rast, W_Rast, TW_MultFact):    
    LOG.debug('\nOpening and Reading ' + VDTDatabaseFileName)
    
    # Read the VDT Database into a DataFrame
    if VDTDatabaseFileName.endswith('.parquet'):
        vdt_df = pd.read_parquet(VDTDatabaseFileName)
    else:
        vdt_df = pd.read_csv(VDTDatabaseFileName, engine='pyarrow')
        
    if vdt_df.empty:
        raise ValueError("The VDT Database file is empty or could not be read properly.")
    
    # Add COMID flow information
    comid_flow_df = pd.DataFrame(COMID_Unique_Flow.items(), columns=['COMID', 'Flow'])
    vdt_df = vdt_df.merge(comid_flow_df, on='COMID', how='left')    

    # Ensure row and col are integers
    vdt_df['Row'] = vdt_df['Row'].astype(int)
    vdt_df['Col'] = vdt_df['Col'].astype(int)
    
    # Extract the column indices for interpolation
    flow_cols = [list(vdt_df.columns).index(col) for col in vdt_df.columns if col.startswith('q_')]
    top_width_cols = [list(vdt_df.columns).index(col) for col in vdt_df.columns if col.startswith('t_')]
    wse_cols = [list(vdt_df.columns).index(col) for col in vdt_df.columns if col.startswith('wse_')]
    
    # Extract flow, baseflow, and elevation values
    flow = vdt_df['Flow'].values.astype(np.float32)
    qb = vdt_df['QBaseflow'].values.astype(np.float32)
    e_dem = E_DEM[vdt_df['Row'].values + 1, vdt_df['Col'].values + 1]

    # Extract flow, TopWidth, and WSE values for interpolation
    flow_values = vdt_df.iloc[:, flow_cols].values.astype(np.float32)
    top_width_values = vdt_df.iloc[:, top_width_cols].values.astype(np.float32)
    wse_values = vdt_df.iloc[:, wse_cols].values.astype(np.float32)
    elev_values = vdt_df['Elev'].values.astype(np.float32)

    top_width, depth, wse = vdt_interpolate(flow, qb, flow_values, top_width_values, elev_values, wse_values, e_dem, TW_MultFact)

    # Add the calculated values to the DataFrame
    vdt_df['TopWidth'] = top_width
    vdt_df['Depth'] = depth
    vdt_df['WSE'] = wse

    # Remove outliers by COMID
    for col in ['TopWidth', 'Depth', 'WSE']:
        grp_mean = vdt_df.groupby('COMID')[col].transform('mean')
        grp_std = vdt_df.groupby('COMID')[col].transform('std')

        lower = grp_mean - 2 * grp_std
        upper = grp_mean + 2 * grp_std

        # Set outliers to NaN
        vdt_df[col] = vdt_df[col].where((vdt_df[col] >= lower) & (vdt_df[col] <= upper))

    # Drop rows with NaN values introduced during outlier removal
    vdt_df = vdt_df.dropna(subset=['TopWidth', 'Depth', 'WSE'])
    
    # Fill T_Rast and W_Rast
    T_Rast[vdt_df['Row'], vdt_df['Col']] = vdt_df['TopWidth']
    W_Rast[vdt_df['Row'], vdt_df['Col']] = vdt_df['WSE']
    
    # Calculate median values by COMID
    mean_values = vdt_df.groupby('COMID').agg({
        'TopWidth': 'max',
        'Depth': 'mean',
        'WSE': 'mean'
    })
    
    # Map results back to the unique COMID list
    comid_result_df = pd.DataFrame({'COMID': COMID_Unique})
    comid_result_df = comid_result_df.merge(mean_values, on='COMID', how='left').fillna(0)
    comid_result_df['COMID'] = comid_result_df['COMID'].astype(np.int32)
    comid_result_df['TopWidth'] = comid_result_df['TopWidth'].astype(np.float32)
    comid_result_df['Depth'] = comid_result_df['Depth'].astype(np.float32)
    
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
    try:
        dataset = gdal.Open(InRAST_Name, gdal.GA_ReadOnly)     
    except RuntimeError:
        sys.exit(" ERROR: Field Raster File cannot be read!")
    # Retrieve dimensions of cell size and cell count then close DEM dataset
    geotransform = dataset.GetGeoTransform()
    # Continue grabbing geospatial information for this use...
    band = dataset.GetRasterBand(1)
    RastArray = band.ReadAsArray()
    #global ncols, nrows, cellsize, yll, yur, xll, xur
    ncols=band.XSize
    nrows=band.YSize
    band = None
    cellsize = geotransform[1]
    yll = geotransform[3] - nrows * np.fabs(geotransform[5])
    yur = geotransform[3]
    xll = geotransform[0]
    xur = xll + (ncols)*geotransform[1]
    lat = np.fabs((yll+yur)/2.0)
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


#   Convert_GDF_to_Output_Raster(Flood_File, flood_gdf, 'Value', ncols, nrows, dem_geotransform, dem_projection, "GTiff", gdal.GDT_Int32)
def Convert_GDF_to_Output_Raster(s_output_filename, gdf, Param, ncols, nrows, dem_geotransform, dem_projection, s_file_format, s_output_type):   
    LOG.info(s_output_filename)

    # Rasterize geometries
    LOG.info('Rasterizing geometries')
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[Param]))  # Replace 'value_column' with your column name
    raster_data = rasterize(
        shapes=shapes,
        out_shape=(nrows, ncols),
        transform=dem_geotransform,
        fill=0,  # Value to use for areas not covered by geometries
        dtype=s_output_type
    )
    LOG.info('Writing output file')
    # Write raster to file
    with rasterio.open(
        s_output_filename,
        "w",
        driver=s_file_format,
        height=nrows,
        width=ncols,
        count=1,
        dtype=s_output_type,
        crs=gdf.crs.to_string(),  # Use the GeoDataFrame's CRS
        transform=dem_geotransform,
    ) as dst:
        dst.write(raster_data, 1)
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
    memory_driver = ogr.GetDriverByName('Memory')
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
def CreateSimpleFloodMap(RR, CC, T_Rast, W_Rast, E, B, nrows, ncols, sd, TW_m, dx, dy, LocalFloodOption, COMID_Unique_TW: dict, COMID_Unique_Depth: dict, WeightBox, TW_for_WeightBox_ElipseMask, TW, TW_MultFact, TopWidthPlausibleLimit, Set_Depth, flood_vdt_cells, OutDEP, OutWSE):
       
    COMID_Averaging_Method = 0
    
    WSE_Times_Weight = np.zeros((nrows+2,ncols+2), dtype=np.float32)
    Total_Weight = np.zeros((nrows+2,ncols+2), dtype=np.float32)
    
    #Now go through each cell
    num_nonzero = len(RR)
    for i in range(num_nonzero):
        r = RR[i]
        c = CC[i]
        r_use = r
        c_use = c
        E_Min = E[r,c]
        
        COMID_Value = B[r,c]
        #Now start with rows and start flooding everything in site
        if Set_Depth>0.0:
            WSE = float(E[r_use,c_use] + Set_Depth)
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
        else:
            #These are Based on the AutoRoute Results, not averaged for COMID
            WSE = W_Rast[r-1,c-1]  #Have to have the '-1' because of the Row and Col being inset on the B raster.
            COMID_TW_m = T_Rast[r-1,c-1]
            #print(str(WSE) + '  ' + str(COMID_TW_m))
        


        if WSE < 0.001 or COMID_TW_m < 0.00001 or (WSE - E[r,c]) < 0.001:
            continue

        
        if COMID_TW_m > TW_m:
            COMID_TW_m = TW_m

        COMID_TW = int(max(np.round(COMID_TW_m / dx), np.round(COMID_TW_m / dy)))
          #This is how many cells we will be looking at surrounding our stream cell


        
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
        if LocalFloodOption:
            #Find what would flood local
            E_Box = E[r_min:r_max,c_min:c_max]
            FloodLocalMask = FloodAllLocalAreas(WSE, E_Box, r_min, r_max, c_min, c_max, r_use, c_use)
            WSE_Times_Weight[r_min:r_max, c_min:c_max] += (WSE * weight_slice * FloodLocalMask)
        else:
            WSE_Times_Weight[r_min:r_max, c_min:c_max] += (WSE * weight_slice)

        Total_Weight[r_min:r_max,c_min:c_max] += weight_slice

        # # Mike added this in February 2024, trying to limit erroneous flooding in Montana. 
        # if limit_low_elev_flooding==True:
        #     #StrmElevBox = E[r_min:r_max,c_min:c_max] * np.where(B[r_min:r_max,c_min:c_max] > 0, 1, np.nan)  #Get a numpy array with either stream cell elevation, or a nan value
        #     #min_strm_elev = np.nanmin( E[r_min:r_max,c_min:c_max] * np.where(B[r_min:r_max,c_min:c_max] > 0, 1, np.nan) )
        #     min_strm_elev = E_Min = E[r_use, c_use]
        #     ElevMask[w_r_min:w_r_max,w_c_min:w_c_max] = (E[r_min:r_max,c_min:c_max] >= min_strm_elev).astype(int)
        # #else the ElevMask is set to all ones
       
        # if LocalFloodOption==True:
        #     WSE_Times_Weight[r_min:r_max,c_min:c_max] = WSE_Times_Weight[r_min:r_max,c_min:c_max] + WSE * WeightBox[w_r_min:w_r_max,w_c_min:w_c_max] * ElipseMask[COMID_TW, w_r_min:w_r_max,w_c_min:w_c_max] * FloodLocalMask
        #     Total_Weight[r_min:r_max,c_min:c_max] = Total_Weight[r_min:r_max,c_min:c_max] + WeightBox[w_r_min:w_r_max,w_c_min:w_c_max] * ElipseMask[COMID_TW, w_r_min:w_r_max,w_c_min:w_c_max] * FloodLocalMask
        # else:
        #     WSE_Times_Weight[r_min:r_max,c_min:c_max] = WSE_Times_Weight[r_min:r_max,c_min:c_max] + WSE * WeightBox[w_r_min:w_r_max,w_c_min:w_c_max] * ElipseMask[COMID_TW, w_r_min:w_r_max,w_c_min:w_c_max]
        #     Total_Weight[r_min:r_max,c_min:c_max] = Total_Weight[r_min:r_max,c_min:c_max] + WeightBox[w_r_min:w_r_max,w_c_min:w_c_max] * ElipseMask[COMID_TW, w_r_min:w_r_max,w_c_min:w_c_max] 
        
        
        ###INSTEAD OF ENFORCING BATHY VALUES ON EVERYWHERE ELSE, Just build the bathymetry in another separate function!!!!
        '''
        #Bathymetry
        if Bathy_Yes==1:
            #Find all the bathymetry points within the box that are rivers (ARBathyMask) and have a bathymetry value from ARC (ARBathy)
            (bathy_r_list, bathy_c_list) = np.where(ARBathy[r_min:r_max,c_min:c_max] > 0.1)
            num_bathy_pnts = len(bathy_r_list)
            if num_bathy_pnts>0:
                for bbb in range(num_bathy_pnts):
                    bathy_r = bathy_r_list[bbb] + r_min
                    bathy_c = bathy_c_list[bbb] + c_min
                    
                    b_r_min = bathy_r-COMID_TW_Bathy
                    b_r_max = bathy_r+COMID_TW_Bathy+1
                    if b_r_min<1:
                        b_r_min = 1 
                    if b_r_max>-1:
                        b_r_max=nrows+1
                    b_c_min = bathy_c-COMID_TW_Bathy
                    b_c_max = bathy_c+COMID_TW_Bathy+1
                    if b_c_min<1:
                        b_c_min = 1 
                    if b_c_max>-1:
                        b_c_max=ncols+1
                    
                    w_r_min = TW_for_WeightBox_ElipseMask-(bathy_r-b_r_min)
                    w_r_max = TW_for_WeightBox_ElipseMask+b_r_max-bathy_r
                    w_c_min = TW_for_WeightBox_ElipseMask-(bathy_c-b_c_min)
                    w_c_max = TW_for_WeightBox_ElipseMask+b_c_max-bathy_c
                    
                    
                    Bathy_Times_Weight[b_r_min:b_r_max,b_c_min:b_c_max] = Bathy_Times_Weight[b_r_min:b_r_max,b_c_min:b_c_max] + ARBathy[bathy_r,bathy_c] * WeightBox[w_r_min:w_r_max,w_c_min:w_c_max] * ElipseMask[COMID_TW_Bathy, w_r_min:w_r_max,w_c_min:w_c_max] ###* ARBathyMask[b_r_min:b_r_max,b_c_min:b_c_max]
                    Bathy_Total_Weight[b_r_min:b_r_max,b_c_min:b_c_max] = Bathy_Total_Weight[b_r_min:b_r_max,b_c_min:b_c_max] + WeightBox[w_r_min:w_r_max,w_c_min:w_c_max] * ElipseMask[COMID_TW_Bathy, w_r_min:w_r_max,w_c_min:w_c_max] ###* ARBathyMask[b_r_min:b_r_max,b_c_min:b_c_max]
                    ARBathy[bathy_r,bathy_c]=0  #Since we evaluated this one, no need to evaluate again
        '''

    # divide the values in WSE_Times_Weight by the 
    # values in Total_Weight
    # WSE_divided_by_weight = WSE_Times_Weight / Total_Weight
    Total_Weight = np.where(Total_Weight == 0, 1e-12, Total_Weight)  # Avoid division by zero
    WSE_divided_by_weight = WSE_Times_Weight / Total_Weight


    # Create the Flooded array
    Flooded_array = np.where((WSE_divided_by_weight>E)&(E>-9998.0), 1, 0).astype(np.uint8)

    # Also make sure all the Cells that have Stream are counted as flooded.
    if flood_vdt_cells:
        for i in range(num_nonzero):
            Flooded_array[RR[i],CC[i]] = 1

    # Create the Depth array
    if OutDEP:
        Depth_array = np.where((WSE_divided_by_weight > E) & (E > -9998.0), WSE_divided_by_weight-E, np.nan).astype(np.float32)
    else:
        Depth_array = np.empty((3, 3), dtype=np.float32) # Dummy array if not used

    # Create the WSE array
    if OutWSE:
        WSE_array = np.where((WSE_divided_by_weight > E) & (E > -9998.0), WSE_divided_by_weight, np.nan).astype(np.float32)
    else:
        WSE_array = np.empty((3, 3), dtype=np.float32) # Dummy array if not used

    return Flooded_array[1:-1, 1:-1], Depth_array[1:-1, 1:-1], WSE_array[1:-1, 1:-1]

@njit("float32[:](float32)", cache=True)
def create_gaussian_kernel_1d(sigma):
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    center = kernel_size // 2

    kernel = np.empty(kernel_size, dtype=np.float32)

    for i in range(kernel_size):
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
                if val != -9999.0:
                    w = kernel[k_start + k]
                    acc += val * w
                    weight_sum += w

            output[r, c] = acc / weight_sum if weight_sum > 0 else image[r, c]
    return output

@njit("float32[:, :](float32[:, :], float32[:])", cache=True)
def convolve_cols(image, kernel):
    nrows, ncols = image.shape
    klen = len(kernel)
    pad = klen // 2
    output = np.zeros_like(image)

    for c in range(ncols):
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


# @njit(cache=True)
def Create_Topobathy_Dataset(E, nrows, ncols, WeightBox, TW_for_WeightBox_ElipseMask, Bathy, ARBathyMask, Bathy_Use_Banks):
    #Bathymetry
    
    # Max_TW_to_Search_for_Bathy_Point = 13
    
    # (r_cells_to_evaluate, c_cells_to_evaluate) = np.where(ARBathy==-99.000)
    # # LOG.info('Number of Bathy Cells to Fill: ' + str(len(r_cells_to_evaluate)))
    # num_cells_to_evaluate = len(r_cells_to_evaluate)
    # if num_cells_to_evaluate>0:

    #     for bbb in range(num_cells_to_evaluate):
    #         bathy_r = r_cells_to_evaluate[bbb]
    #         bathy_c = c_cells_to_evaluate[bbb]
            
    #         for COMID_TW_Bathy in range(1,Max_TW_to_Search_for_Bathy_Point):
    #             b_r_min = max(bathy_r - COMID_TW_Bathy, 1)
    #             b_r_max = min(bathy_r + COMID_TW_Bathy + 1, nrows + 1)
    #             b_c_min = max(bathy_c - COMID_TW_Bathy, 1)
    #             b_c_max = min(bathy_c + COMID_TW_Bathy + 1, ncols + 1)
                
    #             (r_has_bathy, c_has_bathy) = np.where( ((ARBathy[b_r_min:b_r_max,b_c_min:b_c_max]*ARBathyMask[b_r_min:b_r_max,b_c_min:b_c_max]) >=-98.99) | ((ARBathy[b_r_min:b_r_max,b_c_min:b_c_max]*ARBathyMask[b_r_min:b_r_max,b_c_min:b_c_max]) != 0.000))
                
    #             if len(r_has_bathy)>0:
                    
    #                 #This should be a list of rows and columns within the ARBathy that actually have bathymetry values
    #                 r_has_bathy = r_has_bathy + b_r_min
    #                 c_has_bathy = c_has_bathy + b_c_min
                    
    #                 #This should be a list of rows and columns for the Weight and Elipse Rasters.
    #                 w_r = TW_for_WeightBox_ElipseMask-(r_has_bathy-b_r_min)
    #                 w_c = TW_for_WeightBox_ElipseMask-(c_has_bathy-b_c_min)
                    
    #                 # Build the weighted average of the bathymetry values
    #                 total = 0.0
    #                 weight_total = 0.0
    #                 for i in range(len(r_has_bathy)):
    #                     r = r_has_bathy[i]
    #                     c = c_has_bathy[i]
    #                     w_r_i = w_r[i]
    #                     w_c_i = w_c[i]
    #                     val = ARBathy[r, c]
    #                     weight = WeightBox[w_r_i, w_c_i]
    #                     total += val * weight
    #                     weight_total += weight

    #                 if weight_total > 0:
    #                     Bathy[bathy_r, bathy_c] = total / weight_total


    # replace the remaining bad values with the nearest valid points
    # this Scipy function doesn't work with Numba, so Numba is turned off for this function
    mask = (Bathy > -98.99)
    # Find nearest valid points for all cells
    indices = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    # Replace invalid points with the values from the nearest valid points
    Bathy[~mask] = Bathy[indices[0][~mask], indices[1][~mask]]

    # smooth the bathymetry before we finish
    sigma_value = 1.0
    Bathy = gaussian_blur_separable(Bathy, sigma=sigma_value)

    # filter out any bathy data not in the water mask
    # Bathy = np.where(ARBathyMask==1, Bathy, E)
    mask = ARBathyMask != 1
    Bathy[mask] = E[mask]

    # Replace *any remaining bad values*, including -99, -9999, or NaN
    bad_mask = (Bathy <= -98.99) | (Bathy < -9998.0)
    Bathy[bad_mask] = E[bad_mask]

    # replace any values below -9998.0 with a np.nan
    Bathy = np.where(Bathy < -9998.0, np.nan, Bathy)

    # if we used the WSE and baseflow, we don't want the bathymetry to be above the WSE
    # bathymetry may be higher if we use bank elevations and bankfull discharge estimate
    if not Bathy_Use_Banks:
        Bathy[Bathy > E] = E[Bathy > E]

    return Bathy[1:nrows+1, 1:ncols+1]

def Calculate_Depth_TopWidth_TWMax(E, CurveParamFileName, VDTDatabaseFileName, COMID_Unique_Flow, COMID_Unique, Q_Fraction, T_Rast, W_Rast, TW_MultFact, TopWidthPlausibleLimit, dx, dy, Set_Depth, quiet, linkno_to_twlimit=None):
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
        (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, T_Rast, W_Rast) = Calculate_TW_D_ForEachCOMID_VDTDatabase(E, VDTDatabaseFileName, COMID_Unique_Flow, COMID_Unique, T_Rast, W_Rast, TW_MultFact)
    elif len(CurveParamFileName)>1:
        (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, T_Rast, W_Rast) = Calculate_TW_D_ForEachCOMID_CurveFile(CurveParamFileName, COMID_Unique_Flow, COMID_Unique,  T_Rast, W_Rast, TW_MultFact)

    LOG.info('Maximum Top Width = ' + str(TopWidthMax))
    
    if not quiet:
        for idx, comid in enumerate(COMID_Unique):
            if COMID_Unique_TW[comid]>TopWidthPlausibleLimit:
                LOG.warning(f"Ignoring {comid}  {COMID_Unique_Flow[comid]}  {COMID_Unique_Flow[comid]*Q_Fraction}  {COMID_Unique_Depth[comid]}  {COMID_Unique_TW[comid]}")  

    if TopWidthPlausibleLimit < TopWidthMax:
        TopWidthMax = TopWidthPlausibleLimit
    
    #Create a Weight Box and an Elipse Mask that can be used for all of the cells
    X_cells = np.round(TopWidthMax/dx,0)
    Y_cells = np.round(TopWidthMax/dy,0)
    TW = int(max(Y_cells,X_cells))  #This is how many cells we will be looking at surrounding our stream cell
    
    return COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, TW, T_Rast, W_Rast

def Curve2Flood(E, B, RR, CC, nrows, ncols, dx, dy, COMID_Unique, COMID_Unique_Flow, CurveParamFileName, VDTDatabaseFileName, Q_Fraction, TopWidthPlausibleLimit, TW_MultFact, WeightBox, TW_for_WeightBox_ElipseMask, LocalFloodOption, Set_Depth, quiet, flood_vdt_cells, T_Rast, W_Rast, OutDEP, OutWSE, linkno_to_twlimit=None):    
    #Calculate an Average Top Width and Depth for each stream reach.
    #  The Depths are purposely adjusted to the DEM that you are using (this addresses issues with using the original or bathy dem)
    (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, TW, T_Rast, W_Rast) = Calculate_Depth_TopWidth_TWMax(E, CurveParamFileName, VDTDatabaseFileName, COMID_Unique_Flow, COMID_Unique, Q_Fraction, T_Rast, W_Rast, TW_MultFact, TopWidthPlausibleLimit, dx, dy, Set_Depth, quiet, linkno_to_twlimit=linkno_to_twlimit)
    
    #Create a simple Flood Map Data
    search_dist_for_min_elev = 0
    LOG.info('Creating Rough Flood Map Data...')

    # In Curve2Flood(...) just before CreateSimpleFloodMap(...)
    keys_tw  = np.asarray(list(COMID_Unique_TW.keys()), dtype=np.int32)
    vals_tw  = np.asarray(list(COMID_Unique_TW.values()), dtype=np.float32)
    keys_dep = np.asarray(list(COMID_Unique_Depth.keys()), dtype=np.int32)
    vals_dep = np.asarray(list(COMID_Unique_Depth.values()), dtype=np.float32)

    COMID_Unique_TW    = create_numba_dict_from(keys_tw,  vals_tw)
    COMID_Unique_Depth = create_numba_dict_from(keys_dep, vals_dep)

    Flood_array, Depth_array, WSE_array = CreateSimpleFloodMap(RR, CC, T_Rast, W_Rast, E, B, nrows, ncols, search_dist_for_min_elev, TopWidthMax, dx, dy, LocalFloodOption, COMID_Unique_TW, COMID_Unique_Depth, WeightBox, TW_for_WeightBox_ElipseMask, TW, TW_MultFact, TopWidthPlausibleLimit, Set_Depth, flood_vdt_cells, OutDEP, OutWSE)
    
    return Flood_array, Depth_array, WSE_array


def Set_Stream_Locations(nrows: int, ncols: int, infilename: str):
    S = np.full((nrows, ncols), -9999, dtype=np.int32)  #Create an array
    if infilename.endswith('.parquet'):
        df = pd.read_parquet(infilename, columns=['Row', 'Col', 'COMID'])
    else:
        df = pd.read_csv(infilename, usecols=['Row', 'Col', 'COMID'], engine='pyarrow')
        
    S[df['Row'].values, df['Col'].values] = df['COMID'].values
    return S

def Flood_WaterLC_and_STRM_Cells_in_Flood_Map(Flood_Ensemble, S, LandCoverFile, watervalue):
    (LC, ncols, nrows, cellsize, yll, yur, xll, xur, lat, lc_geotransform, lc_projection) = Read_Raster_GDAL(LandCoverFile)
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
    LC = (LC == watervalue).astype(int)

    # Identify streams in SN (1 for streams, 0 otherwise)
    SN = (S > 0).astype(int)

    # Combine LC and SN values, prioritizing SN
    F = SN | LC  # Logical OR operation prioritizes SN over LC

    # Update Flood_Ensemble: mark flooded cells (100) wherever F > 0
    Flood_Ensemble[F > 0] = 100

    return Flood_Ensemble

def Flood_Flooded_Cells_in_Depth_and_WSE_Map(Depth_or_WSE_Ensemble, Flood_Ensemble, eps=0.01):

    # valid sources are flooded cells with a real (non-NaN) value
    source_mask =  (Flood_Ensemble > 0) & (~np.isnan(Depth_or_WSE_Ensemble))
    # targets are flooded cells that are currently NaN
    target_mask = (Flood_Ensemble > 0) & (np.isnan(Depth_or_WSE_Ensemble))

    # Copy to avoid modifying original array (optional)
    filled = Depth_or_WSE_Ensemble.copy()

    # find the closest depth or WSE values from valid source cells
    if np.any(source_mask):
        # EDT returns indices of the nearest ZERO in the input,
        # so pass the inverse to point toward TRUE source cells.
        _, (ny, nx) = distance_transform_edt(~source_mask, return_indices=True)
        # nearest neighbor values from donors
        nn_vals = Depth_or_WSE_Ensemble[ny, nx]
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
        ls = lines[i].strip().split()
        if len(ls)>1 and ls[0]==P:
            if P in ['LocalFloodOption', 'FloodLocalOnly', 'Flood_WaterLC_and_STRM_Cells', 'Bathy_Use_Banks']:
                return True
            if P=='Set_Depth' or P=='FloodSpreader_SpecifyDepth':
                return float(ls[1])
            return ls[1]
        
    if P=='Q_Fraction':
        return 1.0
    if P=='TopWidthPlausibleLimit':
        return 1000.0
    if P=='TW_MultFact':
        return 3.0
    if P=='Set_Depth' or P=='FloodSpreader_SpecifyDepth':
        return float(-1.1)
    if P in ['LocalFloodOption', 'FloodLocalOnly', 'Flood_WaterLC_and_STRM_Cells', 'Bathy_Use_Banks']:
        return False
    if P=='LAND_WaterValue':
        return 80
    if P=='OutDEP' or P=='OutWSE':
        return ""

    return ''


def create_depth_or_wse(
    num_flows: int,
    array_list: list[np.ndarray],
    Flood_Ensemble: np.ndarray,
    streams: np.ndarray,
    geotransform: tuple,
    projection: str,
    ncols: int,
    nrows: int,
    fname: str,
    nodata_value: float = -9999.0,   # use a numeric NoData sentinel (safer than NaN for many readers)
):
    # --- average across ensembles, ignoring NaNs correctly ---
    # Sum of non-NaN values
    sum_arr = np.nansum(array_list, axis=0).astype(np.float32)

    # Count of valid (non-NaN) members at each cell
    valid_count = np.sum(~np.isnan(array_list), axis=0)

    # Avoid divide-by-zero; initialize with NaN where no valid members
    avg = np.full_like(Flood_Ensemble, np.nan, dtype=np.float32)
    np.divide(sum_arr, valid_count, out=avg, where=(valid_count > 0))

    # Optional rounding (match your original 0.01 precision)
    avg = np.round(avg, 2)

    # Your domain-specific masks
    avg = Flood_Flooded_Cells_in_Depth_and_WSE_Map(avg, Flood_Ensemble)
    # avg = remove_cells_not_connected(avg, streams)


    # Treat tiny values as no water; convert to NaN first
    # avg[np.isclose(avg, 0.0, atol=1e-6)] = np.float32('nan')

    # Replace NaN with numeric NoData sentinel for safer downstream handling
    out_band_data = np.where(np.isnan(avg), nodata_value, avg).astype(np.float32)

    # --- write GeoTIFF with NoData metadata set ---
    driver = gdal.GetDriverByName("GTiff")
    # add tiling & compression for performance if desired
    ds: gdal.Dataset = driver.Create(
        fname, ncols, nrows, 1, gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES"]
    )
    if ds is None:
        raise RuntimeError(f"Failed to create output raster: {fname}")

    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)

    band = ds.GetRasterBand(1)
    band.WriteArray(out_band_data)
    band.SetNoDataValue(nodata_value)     # <- critical: advertise NoData to readers
    band.FlushCache()
    ds.FlushCache()

    # Clean close
    band = None
    ds = None

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

def create_bathymetry(E: np.ndarray, nrows: int, ncols: int, dem_geotransform: tuple, dem_projection: str, BathyFromARFileName: str, BathyWaterMaskFileName: str, Flood_Ensemble: np.ndarray, BathyOutputFileName: str, WeightBox: np.ndarray, TW_for_WeightBox_ElipseMask: int, Bathy_Use_Banks: bool, bathymetry_creation_options: list[str] = None):
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
                             bathymetry_creation_options: list[str] = None,):

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
    """
    if input_file:
        #Open the Input File
        with open(input_file,'r') as infile:
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
    Flood_File = ReadInputFile(lines,'OutFLD')
    OutDEP = ReadInputFile(lines,'OutDEP')
    OutWSE = ReadInputFile(lines,'OutWSE')
    FloodImpact_File = ReadInputFile(lines,'FloodImpact_File')
    FlowFileName: str = ReadInputFile(lines,'COMID_Flow_File') or ReadInputFile(lines,'Comid_Flow_File')
    VDTDatabaseFileName = ReadInputFile(lines,'Print_VDT_Database')
    CurveParamFileName = ReadInputFile(lines,'Print_Curve_File')
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
    Bathy_Use_Banks = bool(ReadInputFile(lines,'Bathy_Use_Banks'))

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
    
    Model_Start_Time = datetime.now()

    LOG.info('Opening ' + DEM_File)
    ds: gdal.Dataset = gdal.Open(DEM_File)
    dem_geotransform = ds.GetGeoTransform()
    dem_projection = ds.GetProjection()
    nrows = ds.RasterYSize
    ncols = ds.RasterXSize
    cellsize = dem_geotransform[1]
    yll = dem_geotransform[3] - nrows * abs(dem_geotransform[5])
    yur = dem_geotransform[3]
    
    E = np.full((nrows+2, ncols+2), -9999.0)  #Create an array that is slightly larger than the STRM Raster Array and fill it with -9999.0
    E[1:-1, 1:-1] = ds.ReadAsArray()
    ds = None  

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
    
    #Get Cellsize Information
    (dx, dy, dm) = convert_cell_size(cellsize, yll, yur)
    
    #Get list of Unique Stream IDs.  Also find where all the cell values are.
    B = np.zeros((nrows+2,ncols+2), dtype=np.int32)  #Create an array that is slightly larger than the STRM Raster Array
    B[1:-1, 1:-1] = S

    (RR,CC) = np.where(B > 0)

    COMID_Unique = np.unique(B[RR, CC]) # Always sorted
    COMID_Unique = COMID_Unique.astype(int) # Ensure it's treated as integers
    # COMID_Unique = np.delete(COMID_Unique, 0)  #We don't need the first entry of zero

    # Open the StrmShp_File if provided
    if StrmShp_File:
        # Read the shapefile
        LOG.info('Opening ' + StrmShp_File)
        Strm_gdf = gpd.read_file(StrmShp_File)
        # filter the Strm_gdf to only include the COMIDs in the COMID_Unique array
        Strm_gdf = Strm_gdf[Strm_gdf['LINKNO'].isin(COMID_Unique)]

        # change the TopWidthPlausibleLimit to be weighted by the stream order column in Strm_gdf
        if 'StrmOrder' in Strm_gdf.columns:
            Strm_gdf['TopWidthPlausibleLimit'] = (Strm_gdf['StrmOrder']/max(Strm_gdf['StrmOrder'].values)) * TopWidthPlausibleLimit
            # drop all columns in the GDF except for the LINKNO/COMID column and the TopWidthPlausibleLimit column
            Strm_gdf = Strm_gdf[['LINKNO', 'TopWidthPlausibleLimit']]
            # Build a lookup dictionary from the GDF
            linkno_to_twlimit = dict(zip(Strm_gdf['LINKNO'], Strm_gdf['TopWidthPlausibleLimit']))
            del Strm_gdf
        else:
            linkno_to_twlimit = None
            del Strm_gdf
    else:
        # If no shapefile is provided, set the linkno_to_twlimit to None
        linkno_to_twlimit = None
    
    # COMID Flow File Read-in
    LOG.info('Opening and Reading ' + FlowFileName)
    with open(FlowFileName, 'r') as infile:
        header = infile.readline()
        line = infile.readline()
    
    #Order from highest to lowest flow
    ls = line.split(',')
    num_flows = len(ls)-1
    LOG.info('Evaluating ' + str(num_flows) + ' Flow Events')
    
    #Creating the initial Weight and Eclipse Boxes
    LOG.info('Creating the Weight and Eclipse Boxes')
    TW = int( max( np.round(TopWidthPlausibleLimit/dx,0), np.round(TopWidthPlausibleLimit/dy,0) ) )  #This is how many cells we will be looking at surrounding our stream cell
    TW_for_WeightBox_ElipseMask = TW
    # WeightBox = CreateWeightAndElipseMask(TW_for_WeightBox_ElipseMask, dx, dy, TW_MultFact)  #3D Array with the same row/col dimensions as the WeightBox
    WeightBox = create_weightbox(TW_for_WeightBox_ElipseMask, dx, dy)

    #If you're setting a set-depth value for all streams, just need to simulate one flood event
    if Set_Depth>=0.0:
        num_flows = 1
    
    # Create comid to flow dict
    COMID_Unique_Flow = {}

    # Create initial rasters once, outside the loop
    T_Rast = np.empty((nrows,ncols), np.float32)
    W_Rast = np.empty((nrows,ncols), np.float32)

    #Go through all the Flow Events
    Flood_array_list = []
    Depth_array_list = []
    WSE_array_list = []
    for flow_event_num in range(num_flows):
        LOG.info('Working on Flow Event ' + str(flow_event_num))
        # clear out last event’s values
        T_Rast[:] = -1.0
        W_Rast[:] = -1.0
        #Get an Average Flow rate associated with each stream reach.
        if Set_Depth<=0.000000001:
            COMID_Unique_Flow = FindFlowRateForEachCOMID_Ensemble(FlowFileName, flow_event_num)
        Flood_array, Depth_array, WSE_array = Curve2Flood(E, B, RR, CC, nrows, ncols, dx, dy, COMID_Unique, COMID_Unique_Flow, CurveParamFileName, VDTDatabaseFileName, Q_Fraction, TopWidthPlausibleLimit, TW_MultFact, WeightBox, TW_for_WeightBox_ElipseMask, LocalFloodOption, Set_Depth, quiet, flood_vdt_cells, T_Rast, W_Rast, OutDEP, OutWSE, linkno_to_twlimit=linkno_to_twlimit)        
        Flood_array_list.append(Flood_array)
        Depth_array_list.append(Depth_array)
        WSE_array_list.append(WSE_array)

    # Combine all flow events into a single ensemble
    #Turn into a percentage
    if num_flows > 1:
        Flood_Ensemble = (100 * np.sum(Flood_array_list, axis=0) / num_flows).astype(np.uint8)
    else:
        Flood_Ensemble = Flood_array_list[0] * 100  # Much quicker than nansum

    # If selected, we can also flood cells based on the Land Cover and the Stream Raster
    if Flood_WaterLC_and_STRM_Cells:
        LOG.info('Flooding the Water-Related Land Cover and STRM cells')
        Flood_Ensemble = Flood_WaterLC_and_STRM_Cells_in_Flood_Map(Flood_Ensemble, S, LAND_File, LAND_WaterValue)

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

    if OutDEP:
        create_depth_or_wse(num_flows, Depth_array_list, Flood_Ensemble, S, dem_geotransform, dem_projection, ncols, nrows, OutDEP)

    if OutWSE:
        create_depth_or_wse(num_flows, WSE_array_list, Flood_Ensemble, S, dem_geotransform, dem_projection, ncols, nrows, OutWSE)

    if StrmShp_File:
        # convert the raster to a geodataframe
        flood_gdf = Write_Output_Raster_As_GeoDataFrame(Flood_Ensemble, ncols, nrows, dem_geotransform, dem_projection, gdal.GDT_Byte)
        
        # the name of our flood shapefile
        shp_output_filename = f"{Flood_File[:-4]}.gpkg"

        # save the geodataframe (do not specify the driver, it will be inferred from the file extension)
        flood_gdf.to_file(shp_output_filename)

    if BathyFromARFileName and os.path.exists(BathyFromARFileName) and BathyOutputFileName:
        create_bathymetry(E, nrows, ncols, dem_geotransform, dem_projection, BathyFromARFileName, BathyWaterMaskFileName, Flood_Ensemble, BathyOutputFileName, WeightBox, TW_for_WeightBox_ElipseMask, Bathy_Use_Banks, bathymetry_creation_options)

    # Example of simulated execution
    LOG.info("Flood mapping completed.")
    Model_Simulation_Time = datetime.now() - Model_Start_Time
    LOG.info(f'Simulation time (sec)= {Model_Simulation_Time.seconds}')

    return

