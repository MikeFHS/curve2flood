import numpy as np


class StreamCellValues:
    """A raster's values at the stream cells alone: the curve-based mappers read their top widths, WSEs, slopes and
    stream ids nowhere else, so a value per stream cell takes the place of a raster of every cell.

    ``[rows, cols] = values`` sets the stream cells among those cells as it
    would set a raster's (the other cells are left out: nothing reads them),
    and ``[:] = value`` every stream cell.  ``[row, col]`` is a cell's value,
    as a raster's element, and ``value`` off the stream cells.  ``cells`` are
    the stream cells' flat indices, ``row * ncols + col``, ascending;
    ``values`` their values.
    """

    def __init__(self, cells: np.ndarray, ncols: int, value: float, dtype=np.float32):
        self.cells = cells
        self.ncols = ncols
        self.values = np.full(cells.size, value, dtype=dtype)
        self.fill = self.values.dtype.type(value)

    def __getitem__(self, index):
        row, col = index
        flat = int(row) * self.ncols + int(col)
        at = int(np.searchsorted(self.cells, flat))
        return self.values[at] if at < self.cells.size and self.cells[at] == flat else self.fill

    def __setitem__(self, index, values):
        if isinstance(index, slice):
            self.values[index] = values
            return
        if not self.cells.size:
            return
        rows, cols = index
        flat = np.asarray(rows, dtype=np.int64) * self.ncols + np.asarray(cols, dtype=np.int64)
        at = np.minimum(np.searchsorted(self.cells, flat), self.cells.size - 1)
        stream = self.cells[at] == flat
        self.values[at[stream]] = np.asarray(values)[stream]

    def raster(self, nrows: int) -> np.ndarray:
        """The raster, ``nrows`` by ``ncols``, NaN but at the cells."""
        raster = np.full((nrows, self.ncols), np.nan, dtype=self.values.dtype)
        raster.ravel()[self.cells] = self.values
        return raster