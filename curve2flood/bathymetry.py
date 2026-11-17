import numpy as np
from osgeo import gdal

from curve2flood._log import LOG
from curve2flood._raster import _STRIP_ROWS, _read_rows, open_raster, write_threads
from curve2flood.config import Config


def create_bathymetry(config: Config, E: np.ndarray | None, nrows: int, ncols: int, dem_geotransform: tuple,
                      dem_projection: str, bathymetry_creation_options: list[str] | None = None):
    """Write the topobathymetric DEM (``FSOutBATHY``): the DEM with ARC's
    bathymetry (``BATHY_Out_File``) burned in (see ``Create_Topobathy_Dataset``).
    ``E`` is the DEM padded by one cell, or None to read it from ``DEM_File``.
    Each cell is burned on its own, so the rasters are read, burned and
    written a strip of rows at a time.
    """
    LOG.info('Working on Bathymetry')
    bathy_ds: gdal.Dataset = open_raster(config.bathy_out_file)
    bathy_band: gdal.Band = bathy_ds.GetRasterBand(1)
    nodata = bathy_band.GetNoDataValue()
    if (bathy_ds.RasterYSize, bathy_ds.RasterXSize) != (nrows, ncols):
        raise ValueError(f"The bathymetry raster {config.bathy_out_file} is {bathy_ds.RasterYSize} by "
                         f"{bathy_ds.RasterXSize} cells, and the DEM {nrows} by {ncols}.")
    
    # The dataset is kept, or its band would outlive it
    dem_ds = None if E is not None else open_raster(config.dem_file)
    dem_band = None if dem_ds is None else dem_ds.GetRasterBand(1)

    creation_options = bathymetry_creation_options
    if creation_options is None:
        creation_options = [f"COMPRESS={config.compression}", 'PREDICTOR=2', write_threads()]
    out_ds: gdal.Dataset = gdal.GetDriverByName("GTiff").Create(config.fsoutbathy, xsize=ncols, ysize=nrows, bands=1,
                                                                eType=gdal.GDT_Float32, options=creation_options)
    out_ds.SetGeoTransform(dem_geotransform)
    out_ds.SetProjection(dem_projection)
    out_band = out_ds.GetRasterBand(1)
    for r0 in range(0, nrows, _STRIP_ROWS):
        n = min(_STRIP_ROWS, nrows - r0)
        bathy = _read_rows(bathy_band, r0, n, ncols)
        if nodata is not None:
            bathy[bathy == nodata] = np.nan
        if E is not None:
            dem = E[1 + r0:1 + r0 + n, 1:-1]
        else:
            dem = _read_rows(dem_band, r0, n, ncols)

        out_band.WriteArray(Create_Topobathy_Dataset(dem, bathy, config.bathy_use_banks), 0, r0)
        # Out of GDAL's block cache, which would hold every block until the file is closed
        out_band.FlushCache()
    # Closing the dataset writes it to disk
    out_band = out_ds = dem_band = dem_ds = None

def Create_Topobathy_Dataset(E: np.ndarray, Bathy: np.ndarray, Bathy_Use_Banks: bool) -> np.ndarray:
    """The DEM ``E`` with ARC's bathymetry ``Bathy`` burned in: ARC's bed
    elevation wherever it has one (finite), and the DEM elsewhere.  Unless
    ``Bathy_Use_Banks`` is set, the result is never above the DEM.

    ARC has already decided where its channel goes and filled the small gaps
    between its cross sections, so its raster is taken as it is: each rating
    curve's cell keeps the bed its curve was worked out on (the VDT's
    ``Elev``), and the channel keeps the banks its cross sections were carved
    with.
    """
    burned = np.where(np.isfinite(Bathy), Bathy, E).astype(np.float32, copy=False)
    if not Bathy_Use_Banks:
        np.minimum(burned, E, out=burned)

    return burned