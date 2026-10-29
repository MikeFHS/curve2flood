import numpy as np
from numba import njit


class SparseFill:
    """A raster of mostly zeros, as its other cells by row: row ``r``'s are columns ``cols[starts[r]:starts[r + 1]]``,
    ascending, with ``values``.  ``[pixels]`` gives the values at flat pixel ids, as a raster indexed so would."""

    def __init__(self, starts: np.ndarray, cols: np.ndarray, values: np.ndarray, ncols: int):
        self.starts, self.cols, self.values, self.ncols = starts, cols, values, ncols

    def __getitem__(self, pixels: np.ndarray) -> np.ndarray:
        return _sparse_values(self.starts, self.cols, self.values, self.ncols, np.asarray(pixels))

@njit(cache=True, nogil=True)
def _sparse_values(starts: np.ndarray, cols: np.ndarray, values: np.ndarray, ncols: int,
                   pixels: np.ndarray) -> np.ndarray:
    """``SparseFill``'s values at ``pixels``: found by binary search among their rows' columns, or 0."""
    out = np.zeros(pixels.size, dtype=np.float32)
    for i in range(pixels.size):
        row = pixels[i] // ncols
        col = pixels[i] - row * ncols
        lo, hi = starts[row], starts[row + 1]
        end = hi
        while lo < hi:
            mid = (lo + hi) >> 1
            if cols[mid] < col:
                lo = mid + 1
            else:
                hi = mid
        if lo < end and cols[lo] == col:
            out[i] = values[lo]

    return out