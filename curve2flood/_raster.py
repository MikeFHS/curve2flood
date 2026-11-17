"""Reading rasters with GDAL, and the threads it compresses and decompresses them with."""
import contextlib
from dataclasses import dataclass

import numpy as np
from osgeo import gdal

from curve2flood._log import LOG

gdal.UseExceptions()

# Rows of a raster worked out and written at a time, where a whole raster's temporaries would add to the run's peak
# memory: a row of 256 x 256 tiles
_STRIP_ROWS = 256

def gdal_threads() -> str:
    """How many threads GDAL compresses and decompresses a raster's blocks with: GDAL_NUM_THREADS, as a configuration
    option or an environment variable, when it is set, and otherwise every CPU."""
    return gdal.GetConfigOption("GDAL_NUM_THREADS") or "ALL_CPUS"


def write_threads() -> str:
    """The GeoTIFF creation option that compresses with ``gdal_threads()``."""
    return f"NUM_THREADS={gdal_threads()}"


@contextlib.contextmanager
def decoding_threads():
    """Decompress the rasters opened and read inside with ``gdal_threads()``: GDAL decompresses a read's blocks in
    parallel only when asked.  Set for this thread alone, and only for the while."""
    with gdal.config_option("GDAL_NUM_THREADS", gdal_threads(), thread_local=True):
        yield


def open_raster(path: str) -> gdal.Dataset:
    """A raster opened to read, decompressing with ``gdal_threads()``."""
    with decoding_threads():
        return gdal.Open(str(path), gdal.GA_ReadOnly)


def read_into(ds: gdal.Dataset, arr: np.ndarray):
    """``arr[:] = ds.ReadAsArray()``, band 1, a strip of rows at a time.

    Reading the whole raster at once holds a second copy in the file's own data
    type (float64 for some filled DEMs) and keeps every block in GDAL's cache
    until the file is closed; a strip at a time holds only a strip of each.
    ``arr`` may be a view, such as the inside of a padded array; the values are
    cast as NumPy assigns them.
    """
    band = ds.GetRasterBand(1)
    block_rows = band.GetBlockSize()[1]
    step = block_rows * max(1, 512 // block_rows)
    with decoding_threads():
        for r in range(0, arr.shape[0], step):
            n = min(step, arr.shape[0] - r)
            arr[r:r + n] = band.ReadAsArray(0, r, arr.shape[1], n)
            band.FlushCache()


def read_array(path: str, dtype: np.dtype) -> np.ndarray:
    """Band 1 of a raster as ``dtype``."""
    ds = open_raster(path)
    arr = np.empty((ds.RasterYSize, ds.RasterXSize), dtype=dtype)
    read_into(ds, arr)
    return arr

def _read_rows(band: gdal.Band, r0: int, n: int, ncols: int) -> np.ndarray:
    """``n`` rows of a band from ``r0``, as float32, dropped from GDAL's block cache once read."""
    rows = band.ReadAsArray(0, r0, ncols, n).astype(np.float32, copy=False)
    band.FlushCache()
    return rows

def _check_shape(array: np.ndarray, shape: tuple, name: str):
    """Raise a ValueError if a raster's shape differs from the DEM's."""
    if array.shape != shape:
        raise ValueError(f"{name} dimensions ({array.shape[1]}x{array.shape[0]}) do not match the DEM ({shape[1]}x{shape[0]}).")


@dataclass(frozen=True)
class Grid:
    """Where a raster's cells are: its geotransform, its coordinate system (WKT) and its size."""
    geotransform: tuple[float, float, float, float, float, float]
    projection: str
    nrows: int
    ncols: int

    @classmethod
    def of(cls, path: str) -> "Grid":
        """A raster's grid, from its header alone."""
        ds = open_raster(path)
        return cls(tuple(ds.GetGeoTransform()), ds.GetProjection(), ds.RasterYSize, ds.RasterXSize)

    def check_matches(self, other: "Grid", name: str, other_name: str):
        """Raise a ValueError unless ``other`` has the same size and geotransform, to a millionth of a cell."""
        if (self.nrows, self.ncols) != (other.nrows, other.ncols):
            raise ValueError(f"{name} is {self.nrows} by {self.ncols} cells, but {other_name} is {other.nrows} by "
                             f"{other.ncols}.")
        tolerance = 1e-6 * max(abs(v) for v in self.geotransform[1:3] + self.geotransform[4:6])
        if not np.allclose(self.geotransform, other.geotransform, rtol=0.0, atol=tolerance):
            raise ValueError(f"{name}'s geotransform {self.geotransform} isn't {other_name}'s {other.geotransform}: "
                             f"they cover different ground.")


def Read_Raster_GDAL(InRAST_Name):
    """Read band 1 of a raster as ``(array, geotransform, projection)``.

    South-up rasters (positive pixel height) are flipped to north-up.
    """
    dataset = open_raster(InRAST_Name)
    geotransform = dataset.GetGeoTransform()
    RastArray = dataset.GetRasterBand(1).ReadAsArray()
    nrows = RastArray.shape[0]

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

    Rast_Projection = dataset.GetProjectionRef()
    dataset = None
    return RastArray, geotransform, Rast_Projection