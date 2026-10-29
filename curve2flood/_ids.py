"""Reach (stream) ids: int32 where every one fits, which halves a raster of them, and int64 otherwise."""
import numpy as np

_INT32 = np.iinfo(np.int32)


def id_dtype(ids: np.ndarray) -> np.dtype:
    """int32 if every one of ``ids`` fits in it, else int64: GEOGLOWS and TDX-Hydro ids fit, NHDPlus HR's 14 digits
    don't.  The -9999 of a cell without a stream fits either."""
    ids = np.asarray(ids)
    if ids.size == 0:
        return np.dtype(np.int32)
    
    if ids.dtype.kind in 'iu' and ids.dtype.itemsize <= 2:
        return np.dtype(np.int32)
    
    low, high = ids.min(), ids.max()
    return np.dtype(np.int32) if _INT32.min <= low and high <= _INT32.max else np.dtype(np.int64)


def as_ids(ids: np.ndarray) -> np.ndarray:
    """``ids`` as int32 or int64 (see ``id_dtype``), itself if it already is the one."""
    ids = np.asarray(ids)
    dtype = id_dtype(ids)
    return ids if ids.dtype == dtype else ids.astype(dtype)
