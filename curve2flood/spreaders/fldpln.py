from __future__ import annotations
from pathlib import Path
from multiprocessing import shared_memory

import tqdm
import numpy as np
import pandas as pd
import networkx as nx
from numba import njit
from osgeo import gdal
import geopandas as gpd
import multiprocessing as mp

_SHARED_MEMORYS = {}

# Neighbor order used in the MATLAB code:
# 1 2 3
# 4 x 5
# 6 7 8
NEIGHBOR_DELTAS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

# Whitebox D8
# 64  128 1
# 32  x   2
# 16  8   4

# ESRI D8
# 32  64 128
# 16  x   1
# 8   4   2

@njit(cache=True, nogil=True)
def _pixel_to_rc(pixel: int, ncols: int) -> tuple[int, int]:
    return pixel // ncols, pixel % ncols

@njit(cache=True, nogil=True)
def _rc_to_pixel(row: int, col: int, ncols: int) -> int:
    return np.int32(row * ncols + col)

@njit(cache=True, nogil=True)
def _valid_neighbors(pixel: int, nrows: int, ncols: int):
    row, col = _pixel_to_rc(pixel, ncols)
    out = []
    for pos, (dr, dc) in enumerate(NEIGHBOR_DELTAS):
        rr = row + dr
        cc = col + dc
        if 0 <= rr < nrows and 0 <= cc < ncols:
            out.append((pos, _rc_to_pixel(rr, cc, ncols)))
    return out

@njit(cache=True, nogil=True)
def _next_downstream(pixel: int, fdr: np.ndarray, nrows: int, ncols: int) -> int:
    # ESRI D8
    # OUTFLOW = {
    #     np.uint8(32): (-1, -1),
    #     np.uint8(64): (-1, 0),
    #     np.uint8(128): (-1, 1),
    #     np.uint8(16): (0, -1),
    #     np.uint8(1): (0, 1),
    #     np.uint8(8): (1, -1),
    #     np.uint8(4): (1, 0),
    #     np.uint8(2): (1, 1),
    # }

    # Whitebox D8
    OUTFLOW = {
        np.uint8(64): (-1, -1),
        np.uint8(128): (-1, 0),
        np.uint8(1): (-1, 1),
        np.uint8(32): (0, -1),
        np.uint8(2): (0, 1),
        np.uint8(16): (1, -1),
        np.uint8(8): (1, 0),
        np.uint8(4): (1, 1),
    }

    fd = fdr[pixel]
    if fd == 0:
        return -1

    row, col = _pixel_to_rc(pixel, ncols)
    dr, dc = OUTFLOW[fd]
    rr = row + dr
    cc = col + dc
    if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols:
        return -1
    return _rc_to_pixel(rr, cc, ncols)

def _segment_pixels_matlab(seg_id: int, seg_info: np.ndarray, fdr: np.ndarray,
                    nrows: int, ncols: int) -> list[int]:
    """Return stream pixels for a segment id. The seed-point path is not ported."""
    row_idx = int(seg_id)
    if row_idx < 0 or row_idx >= seg_info.shape[0]:
        raise IndexError("seg0 is outside seg_info.")

    start = round(seg_info[row_idx, 0])
    length = round(seg_info[row_idx, 4])

    if length <= 0:
        return []

    pixels = [start]
    current = start
    for _ in range(1, length):
        nxt = _next_downstream(current, fdr, nrows, ncols)
        if nxt == -1:
            break
        pixels.append(nxt)
        current = nxt
    return pixels


def _downstream_exclusion_matlab(seg_id: int, seg_info: np.ndarray, fdr: np.ndarray,
                          nrows: int, ncols: int) -> set[int]:
    """Pixels downstream of the segment end, excluded from spillover candidates."""
    row_idx = int(seg_id)
    end_pixel = int(seg_info[row_idx, 1])

    excluded: set[int] = set()
    current = end_pixel
    seen = {current}
    while True:
        nxt = _next_downstream(current, fdr, nrows, ncols)
        if nxt == -1 or nxt in seen:
            break
        excluded.add(np.int32(nxt))
        seen.add(np.int32(nxt))
        current = nxt

@njit(cache=True, nogil=True, parallel=True)
def _segment_pixels(stream_id: int, stream_info: np.ndarray, fdr: np.ndarray, nrows: int, ncols: int) -> list[int]:
    """Return stream pixels for a segment id. The seed-point path is not ported."""
    mask = stream_info[:, 3] == stream_id
    if not np.any(mask):
        raise ValueError("stream_id not found in stream_info.")
    
    start = stream_info[mask, 0][0]
    length = stream_info[mask, 2][0]

    pixels = []
    if length <= 0:
        return pixels

    pixels.append(start)
    current = start
    for _ in range(1, length):
        nxt = _next_downstream(current, fdr, nrows, ncols)
        if nxt == -1:
            break
        pixels.append(nxt)
        current = nxt
    return pixels

@njit(cache=True, nogil=True, parallel=True)
def _downstream_exclusion(stream_id: int, stream_info: np.ndarray, fdr: np.ndarray, nrows: int, ncols: int) -> set[int]:
    """Pixels downstream of the segment end, excluded from spillover candidates."""
    end_pixel = stream_info[stream_info[:, 3] == stream_id, 1][0]  # Assuming linkno is in the 4th column (index 3)

    excluded: set[int] = set()
    current = end_pixel
    seen = {current}
    while True:
        nxt = _next_downstream(current, fdr, nrows, ncols)
        if nxt == -1 or nxt in seen:
            break
        excluded.add(nxt)
        seen.add(nxt)
        current = nxt

    return excluded

@njit(cache=True, nogil=True)
def _backfill_from_source(source: int, base_dtf: float, max_wse: float,
                          fil: np.ndarray, fdr: np.ndarray, nrows: int, ncols: int,
                          bg: float, flood_members: set[int] | None = None,
                          excluded: set[int] | None = None) -> list[tuple[int, float]]:
    """
    Backfill opposite the D8 flow direction from `source`.

    Returned DTF values include `base_dtf`. When `flood_members` is supplied,
    pixels already in the floodplain are not returned; this matches the first
    backfill pass in the MATLAB code. Spillover backfill passes leave it as None
    so existing pixels can be overwritten when the new DTF is lower.
    """
    # Whitebox
    INFLOW = np.array([4, 8, 16, 2, 32, 1, 128, 64], dtype=np.int32)

    # MATLAB
    # INFLOW = np.array([2, 4, 8, 1, 16, 128, 64, 32], dtype=np.int32)

    source_elev = fil[source]
    result: list[tuple[int, float]] = []
    queue: list[int] = [source]
    visited = {source}
    if excluded is None:
        excluded = {np.int32(-1)} # This helps numba know what the set's type is

    while queue:
        center = queue.pop()
        for pos, nbr in _valid_neighbors(center, nrows, ncols):
            if nbr in visited or nbr in excluded:
                continue
            if flood_members is not None and nbr in flood_members:
                continue
            if fil[nbr] == bg or fil[nbr] > max_wse:
                continue
            if fdr[nbr] != INFLOW[pos]:
                continue
            dtf = base_dtf + max(0.0, fil[nbr] - source_elev)
            visited.add(np.int32(nbr))
            result.append((nbr, dtf))
            queue.append(nbr)

    return result

@njit(cache=True, nogil=True)
def sort_boundary(lst: list, fil: np.ndarray) -> list:
    # This is a numba-compatible sort, since numba cannot cache .sort(key=lambda x: (x[2], -fil[x[1]]
    def key_func(x):
        return x[2], -fil[x[1]]  # Sort by spillover, than elevation

    for i in range(1, len(lst)):
        current_item = lst[i]
        current_key = key_func(current_item)
        j = i - 1
        
        while j >= 0 and key_func(lst[j]) < current_key:
            lst[j + 1] = lst[j]
            j -= 1
        lst[j + 1] = current_item
        
    return lst

@njit(cache=True, nogil=True)
def sort_candidates(lst: list) -> list:
    # This is a numba-compatible sort, since numba cannot cache .sort(key=lambda x: (x[2], -x[4]))
    def key_func(x):
        return x[2], -x[4]  # Sort by spillover, than elevation

    for i in range(1, len(lst)):
        current_item = lst[i]
        current_key = key_func(current_item)
        j = i - 1
        
        while j >= 0 and key_func(lst[j]) < current_key:
            lst[j + 1] = lst[j]
            j -= 1
        lst[j + 1] = current_item
        
    return lst

@njit(cache=True, nogil=True)
def _initial_boundary(records: dict[int, tuple[int, float]], fil: np.ndarray,
              nrows: int, ncols: int, bg: float) -> list[tuple[int, int, float]]:
    out: list[tuple[int, int, float]] = []
    for pixel, (fsp, dtf) in records.items():
        if fil[pixel] == bg:
            continue
        for _, nbr in _valid_neighbors(pixel, nrows, ncols):
            if nbr not in records and fil[nbr] != bg:
                out.append((fsp, pixel, dtf))
                break
    out = sort_boundary(out, fil)
    return out

@njit(cache=True, nogil=True)
def _update_boundary(records: dict[int, tuple[int, float]], filled_dem: np.ndarray, new_boundary: set[int]) -> list[tuple[int, int, float]]:
    bdy = [
        (records[p][0], p, records[p][1])
        for p in new_boundary
    ]
    bdy = sort_boundary(bdy, filled_dem)
    return bdy

@njit(cache=True, nogil=True)
def _flood_map(records: dict[int, tuple[int, float]], flddat: np.ndarray, fldmn: float) -> None:
    for pixel, (_, dtf) in records.items():
        flddat[pixel] = max(fldmn, dtf)

@njit(cache=True, nogil=True)
def _spill_candidates(boundary: list[tuple[int, int, float]], records: dict[int, tuple[int, float]],
                      fil: np.ndarray, nrows: int, ncols: int,
                      fldht: float, mxht: float, bg: float, excluded: set[int]) -> list[tuple[int, float, int, float, float]]:
    best: dict[int, tuple[float, float, int, float, float]] = {}
    for fsp, bdy_pixel, bdy_dtf in boundary:
        bdy_elev = fil[bdy_pixel]
        if bdy_elev == bg:
            continue
        limit = min(mxht, bdy_elev + fldht - bdy_dtf)
        for _, nbr in _valid_neighbors(bdy_pixel, nrows, ncols):
            if nbr in records or nbr in excluded:
                continue
            nbr_elev = fil[nbr]
            if nbr_elev == bg or nbr_elev > limit:
                continue
            spill_dtf = bdy_dtf + max(0.0, nbr_elev - bdy_elev)
            available_depth = fldht - bdy_dtf - max(0.0, nbr_elev - bdy_elev)
            if nbr not in best:
                best[nbr] = (available_depth, bdy_elev, fsp, spill_dtf, nbr_elev)
            else:
                previous = best[nbr]
                old_available = previous[0]
                old_bdy_elev = previous[1]
                if available_depth > old_available or (
                    available_depth == old_available and bdy_elev > old_bdy_elev
                ):
                    best[nbr] = (available_depth, bdy_elev, fsp, spill_dtf, nbr_elev)

    candidates = [
        (fsp, spill_dtf, pixel, pixel_elev, bdy_elev)
        for pixel, (_, bdy_elev, fsp, spill_dtf, pixel_elev) in best.items()
    ]
    candidates = sort_candidates(candidates)
    return candidates

@njit(cache=True, nogil=True)
def _forward_path(start: int, spill_dtf: float, fil: np.ndarray, fdr: np.ndarray,
                  flddat: np.ndarray, nrows: int, ncols: int, bg: float,
                  excluded: set[int]) -> list[int]:
    path: list[int] = []
    seen: set[int] = set()
    current = start
    while True:
        if current in seen:
            break
        seen.add(np.int32(current))
        path.append(np.int32(current))
        nxt = _next_downstream(current, fdr, nrows, ncols)
        if nxt == -1:
            break
        if nxt in excluded:
            path.append(np.int32(nxt))
            break
        if fil[nxt] == bg:
            break
        if flddat[nxt] > 0 and spill_dtf >= flddat[nxt]:
            break
        current = nxt
    return path

@njit(cache=True, nogil=True)
def _assimilate(records: dict[int, tuple[int, float]], fsp: int, pixel: int, dtf: float) -> bool:
    if pixel not in records:
        records[pixel] = (fsp, dtf)
        return True
    
    if records[pixel][1] > dtf:
        records[pixel] = (fsp, dtf)
        return True
    
    return False

@njit(cache=True, nogil=True, parallel=True)
def _fldpln_library_for_segment(shape: tuple[int, int],
                               dem: np.ndarray,
                               filled_dem: np.ndarray, 
                               flow_direction: np.ndarray, 
                               stream_id: int,
                               stream_info: np.ndarray,
                               dh: float,
                               fldmn: float,
                               fldmx: float,
                               iterative_spill: bool,
                               global_max_wse: float = 0.0,
                               bg: float = -9999) -> pd.DataFrame:
    """
    Build an FLDPLN floodplain table for a stream segment.
    """
    nrows, ncols = shape
    stream_pixels = _segment_pixels(stream_id, stream_info, flow_direction, nrows, ncols)
    excluded = _downstream_exclusion(stream_id, stream_info, flow_direction, nrows, ncols)

    records: dict[int, tuple[int, float]] = {}
    for pixel in stream_pixels:
        if filled_dem[pixel] != bg:
            records[pixel] = (pixel, 0.0)

    fldht = 0.0
    iterations = int(np.ceil(fldmx / dh))

    flood_depths = np.zeros(filled_dem.size, dtype=np.float32)

    for _ in range(iterations):
        fldht += min(dh, fldmx - fldht)

        boundary = _initial_boundary(records, filled_dem, nrows, ncols, bg)
        new_boundary = set()
        _flood_map(records, flood_depths, fldmn)
        flood_members = set(records)

        for fsp, boundary_pixel, boundary_dtf in boundary:
            boundary_elev = filled_dem[boundary_pixel]
            max_wse = min(global_max_wse, boundary_elev + fldht - boundary_dtf)
            additions = _backfill_from_source(
                boundary_pixel, boundary_dtf, max_wse, filled_dem, flow_direction, nrows, ncols,
                bg, flood_members=flood_members
            )
            for pixel, dtf in additions:
                if _assimilate(records, fsp, pixel, dtf):
                    flood_depths[pixel] = max(fldmn, dtf)
                    flood_members.add(np.int32(pixel))
                    new_boundary.add(np.int32(pixel))

        boundary = _update_boundary(records, filled_dem, new_boundary)
        spill = True
        while spill:
            before = len(records)
            _flood_map(records, flood_depths, fldmn)
            candidates = _spill_candidates(
                boundary, records, filled_dem, nrows, ncols, fldht, global_max_wse, bg, excluded
            )

            new_boundary.clear()

            for fsp, spill_dtf, pixel, pixel_elev, _ in candidates:
                path = _forward_path(pixel, spill_dtf, filled_dem, flow_direction, flood_depths, nrows, ncols, bg, excluded)
                if not path:
                    continue
                path_set = set(path)
                path_stage = min(global_max_wse, pixel_elev + fldht - spill_dtf)
                path_depth = path_stage - pixel_elev

                pending: list[tuple[int, float]] = [(p, 0.0) for p in path]
                for source in path:
                    source_wse = filled_dem[source] + path_depth
                    pending.extend(
                        _backfill_from_source(
                            source, 0.0, source_wse, filled_dem, flow_direction, nrows, ncols,
                            bg, flood_members=None, excluded=path_set
                        )
                    )

                for new_pixel, local_dtf in pending:
                    if _assimilate(records, fsp, new_pixel, local_dtf + spill_dtf):
                        new_boundary.add(np.int32(new_pixel))

            boundary = _update_boundary(records, filled_dem, new_boundary)
            if iterative_spill:
                has_shallow_boundary = False

                for _, _, dtf in boundary:
                    if dtf < fldht:
                        has_shallow_boundary = True
                        break

                if len(records) > before and has_shallow_boundary:
                    spill = len([row for row in boundary if row[2] < fldht]) > 0
                else:
                    spill = False
            else:
                spill = False

    rows: list[list[float]] = []
    for pixel, (fsp, dtf) in records.items():
        out_dtf = max(fldmn, dtf)
        sink_fill_depth = filled_dem[pixel] - max(0.0, dem[pixel])
        rows.append([fsp, pixel, out_dtf, sink_fill_depth])

    return rows

def fldpln_library_for_segment(dem: np.ndarray,
                               filled_dem: np.ndarray, 
                               flow_direction: np.ndarray, 
                               stream_id: int,
                               stream_info: np.ndarray,
                               dh: float,
                               fldmn: float,
                               fldmx: float,
                               iterative_spill: bool,
                               global_max_wse: float = 0.0,
                               bg: float = -9999):
    """Build an FLDPLN floodplain table for a single stream segment.

    Parameters
    ----------
    dem : np.ndarray
        Original elevation raster.
    filled_dem : np.ndarray
        Filled elevation raster.
    flow_direction : np.ndarray
        D8 flow-direction raster, assumes whitebox conventions.
    stream_id : int
        Stream segment identifier.
    stream_info : np.ndarray
        Segment metadata, as a 2D array with one row per segment and columns including start pixel (0), end pixel (1), length (2), and linkno (3).
    dh : float
        Flood increment step; must be positive.
    fldmn : float
        Minimum flood depth.
    fldmx : float
        Maximum flood depth to evaluate.
    iterative_spill : bool
        If True, continue spilling while shallow boundary conditions exist.
    global_max_wse : float, optional
        Upper bound on water-surface elevation. 0 means no bound. 
    bg : float, optional
        Background/no-data value in DEM.

    Returns
    -------
    pandas.DataFrame
        Table with FSP, FPP, DTF, and fill depth columns.
    """
    if dh <= 0:
        raise ValueError("dh must be positive.")

    nrows, ncols = dem.shape
    if flow_direction.shape != (nrows, ncols):
        raise ValueError("flow_direction shape must match filled_dem shape.")
    if dem.shape != (nrows, ncols):
        raise ValueError("dem shape must match filled_dem shape.")

    dem = dem.astype(np.float32, copy=False).ravel()
    filled_dem = filled_dem.astype(np.float32, copy=False).ravel()
    flow_direction = flow_direction.astype(np.uint8, copy=False).ravel()

    global_max_wse = np.finfo(np.float32).max if not global_max_wse else 0.99999 * global_max_wse

    rows = _fldpln_library_for_segment(
        (np.int32(nrows), np.int32(ncols)), dem, filled_dem, flow_direction, stream_id, stream_info, dh, fldmn, fldmx, iterative_spill, global_max_wse, bg
    )
    header = [
        "FSP",
        "FPP",
        "DTF",
        "fill depth",
    ]
    df = pd.DataFrame(rows, columns=header)
    df["FSP"] = df["FSP"].astype(np.int32)
    df["FPP"] = df["FPP"].astype(np.int32)
    return df


def _set_shared(name: str, shm: shared_memory.SharedMemory):
    """
    We need the shared memory objects to persist somewhere; otherwise, the memory is freed and the numpy arrays point to invalid memory.
    These must last the lifetime of the program! Reason being, bathymetry is the last thing written, and we need the shared memory to persist until then.
    """
    global _SHARED_MEMORYS
    _SHARED_MEMORYS[name] = shm

def read_array_and_set_shared(file: str, dtype: np.dtype, set_shared: bool,
                              name: str = None):
    ds = gdal.Open(file)
    shape = (ds.RasterYSize, ds.RasterXSize)

    dtype = np.dtype(dtype)
    if not set_shared:
        arr = np.empty(
            shape, 
            dtype=dtype
        )
        arr[:] = ds.ReadAsArray()
        return arr
    
    size = int(dtype.itemsize * np.prod(shape))
    shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    arr[:] = ds.ReadAsArray()
    globals()[name] = arr
    _set_shared(name, shm)
    return arr

def run_parallel(args):
    return fldpln_library_for_segment(
            globals()['dem_array'],
            globals()['filled_dem_array'],
            globals()['flow_direction_array'],
            *args
        )

def close_shared_memory(names: list[str]):
    """
    Close and unlink shared memory segments.

    Parameters
    ----------
    names : list[str]
        Names of the shared memory segments to close.
    """
    for name in names:
        shm = globals().get(name)
        if shm is not None:
            shm.close()
            shm.unlink()
            del globals()[name]
            del _SHARED_MEMORYS[name]

def init_parallel(
    names: list[str],
    shapes: list[tuple],
    dtypes: list[np.dtype],
):
    """
    Worker initializer for multiprocessing.

    Attaches shared memory segments into NumPy arrays and stores them into
    module-level globals so the per-cell worker function can run without
    pickling large arrays.

    Parameters
    ----------
    names, shapes, dtypes
        Metadata produced by :func:`get_init_parallel_args`.
    """
    shms = [shared_memory.SharedMemory(name=name) for name in names]

    for shm, name, shape, dtype in zip(shms, names, shapes, dtypes):
        _set_shared(name, shm)
        globals()[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

def build_fldpln_library(
    dem: str,
    filled_dem: str,
    stream_info_file: str,
    flow_direction_file: str,
    library_file: str,
    dh: float,
    fldmn: float,
    fldmx: float,
    iterative_spill: bool,
    vdt_file: str = None,
    stream_ids: list[int] | None = None,
    global_max_wse: float = 0.0,
    bg: float = -9999,
    parallel: bool = False,
    pbar: bool = True,
    processes: int | None = None
):
    """
    Build floodplain library.

    Parameters
    ----------
    dem: str
        Path to the DEM raster.
    filled_dem: str
        Path to the filled DEM raster.
    stream_info_file: str
        Path to the stream info CSV file.
    flow_direction_file: str
        Path to the flow direction raster.
    dh: float
        Depth increment (meters).
    fldmn: float
        Minimum depth for a cell to be considered flooded (meters).
    fldmx: float
        Maximum floodplain depth to iterate up to (meters).
    iterative_spill: bool
        Whether to use iterative spill routing when generating outputs. Recommended if the DEM is high resolution (<= 10 m)
    vdt_file: str | None
        Optional VDT file used to derive per-stream maximum depths.
    stream_ids: list[int] | None
        Optional subset of COMIDs to process.
    global_max_wse: float
        Global maximum water-surface elevation override.
    bg: float
        Background/no-data value for output rasters.
    parallel: bool
        If True, process stream reaches using multiprocessing.
    pbar
        If True, display a progress bar.
    processes
        Number of worker processes to use when parallel is enabled.
    """
    stream_info = pd.read_csv(stream_info_file)
    assert stream_info.ndim == 2, "Stream info file must be a 2D table"
    assert stream_info.shape[1] == 4, "Stream info file must have 4 columns: start pixel (0), end pixel (1), length (2), and stream ID (3)"

    if stream_ids is not None:
        stream_info = stream_info[stream_info.iloc[:, 3].isin(stream_ids)]

    dem_array = read_array_and_set_shared(dem, np.float32, set_shared=parallel, name='dem_array')
    filled_dem_array = read_array_and_set_shared(filled_dem, np.float32, set_shared=parallel, name='filled_dem_array')
    flow_direction_array = read_array_and_set_shared(flow_direction_file, np.uint8, set_shared=parallel, name='flow_direction_array')

    stream_ids = stream_info.iloc[:, 3].unique()
    if vdt_file is None:
        max_depths = [fldmx] * len(stream_ids)
    else:
        if Path(vdt_file).suffix in {'parquet', 'pq'}:
            vdt_df = pd.read_parquet(vdt_file)
        else:
            vdt_df = pd.read_csv(vdt_file)

        vdt_df = vdt_df[vdt_df['COMID'].isin(stream_ids)]
        # Find last wse_* column
        wse_cols = [col for col in vdt_df.columns if col.startswith('wse_')]
        assert wse_cols, "No wse_* columns found in VDT file"
        max_wse_col = wse_cols[-1]
        vdt_df['depth'] = vdt_df[max_wse_col] - dem_array[vdt_df['Row'], vdt_df['Col']]
        ids_max_depths = vdt_df.groupby('COMID', sort=False, as_index=False)['depth'].max().values
        stream_ids = ids_max_depths[:, 0].astype(np.int32)
        max_depths = np.minimum(ids_max_depths[:, 1], fldmx)

    if pbar:
        pbar = tqdm.tqdm
    else:
        pbar = lambda x, **kwargs: x

    if processes is None:
        processes = max(min(mp.cpu_count(), len(stream_ids)), 1)
        if processes == 1:
            parallel = False

    args = [
        (
            stream_id,
            stream_info.values,
            dh,
            fldmn,
            max_depths[i],
            iterative_spill,
            global_max_wse, 
            bg
        )
        for i, stream_id in enumerate(stream_ids)
    ]
    # Sort args from largest max depth to smallest, (longest time to shortest)
    args.sort(key=lambda x: x[4], reverse=True)

    dfs: list[pd.DataFrame] = []
    if parallel:
        names = ['dem_array', 'filled_dem_array', 'flow_direction_array']
        shapes = [dem_array.shape, filled_dem_array.shape, flow_direction_array.shape]
        dtypes = [dem_array.dtype, filled_dem_array.dtype, flow_direction_array.dtype]
        with mp.Pool(processes, init_parallel, (names, shapes, dtypes)) as pool:
            for df in pbar(pool.imap_unordered(run_parallel, args), total=len(args), desc="Processing streams in parallel"):
                dfs.append(df)
    else:
        for i, _ in enumerate(pbar(stream_ids, desc="Processing streams")):
            dfs.append(fldpln_library_for_segment(
                dem_array,
                filled_dem_array,
                flow_direction_array,
                *args[i]
            ))

    close_shared_memory(['dem_array', 'filled_dem_array', 'flow_direction_array'])

    df = pd.concat(dfs, ignore_index=True)
    dfs = None
    df = df.sort_values(['FSP', 'FPP'], ignore_index=True)
    df = df.round(3)
    if Path(library_file).suffix in {'parquet', 'pq'}:
        df.to_parquet(library_file, compression='brotli', index=False, store_decimal_as_integer=True)
    else:
        df.to_csv(library_file, index=False)

def make_dtf_map(filled_dem_file: str, fldpln_library_file: str, output_file: str):
    filled_dem_array: np.ndarray = gdal.Open(filled_dem_file).ReadAsArray()
    nrows, ncols = filled_dem_array.shape

    if Path(fldpln_library_file).suffix == '.parquet':
        df = pd.read_parquet(fldpln_library_file)
    else:
        df = pd.read_csv(fldpln_library_file)

    df = df.groupby('FPP', as_index=False).agg({'DTF': 'min'})
    df['Row'] = (df['FPP'] // ncols).astype(int)
    df['Col'] = (df['FPP'] % ncols).astype(int)
    df['DTF'] = df['DTF'].clip(lower=0)

    dtf_array = np.full_like(filled_dem_array, np.nan, dtype=np.float32)
    dtf_array[df['Row'], df['Col']] = df['DTF']

    out_ds: gdal.Dataset = gdal.GetDriverByName('GTiff').Create(
        output_file,
        ncols,
        nrows,
        1,
        gdal.GDT_Float32,
        options=['COMPRESS=DEFLATE']
    )
    out_ds.SetGeoTransform(gdal.Open(filled_dem_file).GetGeoTransform())
    out_ds.SetProjection(gdal.Open(filled_dem_file).GetProjection())
    out_ds.WriteArray(dtf_array)
    out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    out_ds.FlushCache()
    out_ds = None

def make_flood_map(
        dem: np.ndarray,
        filled_dem: np.ndarray,
        vdt_df: pd.DataFrame,
        fdr: np.ndarray,
        fldpln_library: pd.DataFrame,
        stream_info_df: pd.DataFrame,
        stream_gdf: gpd.GeoDataFrame,
        dem_with_bathymetry: np.ndarray):
    nrows, ncols = filled_dem.shape
    vdt_df['DoF'] = vdt_df['WSE'] - filled_dem[vdt_df['Row'], vdt_df['Col']]
    vdt_df['FSP'] = (vdt_df['Row'] * ncols + vdt_df['Col']).astype(int)
    stream_ids = set(vdt_df['COMID'])
    existing_fsps = set(vdt_df['FSP'])

    stream_gdf = stream_gdf.sort_values('topological_order')
    stream_ids = stream_gdf['LINKNO']
    G = nx.from_pandas_edgelist(
        stream_gdf[stream_gdf['DSLINKNO'] > 0],
        source='LINKNO',
        target='DSLINKNO',
        create_using=nx.DiGraph
    )

    fdr = fdr.ravel()
    last_dof_dict = {}
    rows_to_add = []

    # We need to traverse each stream segment, and add missing FSPs to the vdt_df with interpolated DoF values.
    last_dof = np.inf
    for stream_id in stream_ids:
        row = stream_info_df.loc[stream_info_df['source_id_col'] == stream_id, ['start_pixel', 'length']]
        if row.empty:
            continue

        if stream_id not in vdt_df['COMID'].values:
            last_dof_dict[stream_id] = last_dof
            continue
        info = row.values[0]
        fsp = info[0]
        length = info[1]
        to_do = []
        last_fsp = None

        last_wse = np.inf
        # row, col = _pixel_to_rc(fsp, ncols)
        # for r, c in zip(*np.where(streams[row-1:row+2, col-1:col+2] > 0)):
        #     potential_upstream_id = streams[row-1+r, col-1+c]
        #     if potential_upstream_id == stream_id:
        #         continue
        #     upstream_dof = last_dof_dict.get(potential_upstream_id, np.inf)
        #     interped_wse = upstream_dof + filled_dem[_pixel_to_rc(fsp, ncols)]
        #     if interped_wse < last_wse:
        #         last_wse = interped_wse
        if last_wse == np.inf and (preds := list(G.predecessors(stream_id))):
            while preds:
                should_break = False
                for pred in preds:
                    if pred in last_dof_dict:
                        last_wse = last_dof_dict[pred] + filled_dem[_pixel_to_rc(fsp, ncols)]
                        should_break = last_wse != np.inf
                        if should_break:
                            break
                if should_break:
                    break
                preds = [p for pred in preds for p in G.predecessors(pred)]

        wses = []
        wses_raw = []
        xs = []
        elevs = []
        bathys = []
        for i in range(length):
            if fsp in existing_fsps:
                raw_wse = vdt_df.loc[vdt_df['FSP'] == fsp, 'WSE'].values[0]
                if to_do:
                    if last_fsp is None:
                        # No values at the beginning of a segment; just use a constant WSE. Worst case is we underestimate these pixels
                        current_wse = vdt_df.loc[vdt_df['FSP'] == fsp, 'WSE'].values[0]
                        if current_wse < last_wse:
                            last_wse = current_wse
                        elif current_wse > last_wse + 0.1: # ALlow downstream to rise 10 cm
                            # Reassign wse
                            current_wse = last_wse
                            vdt_df.loc[vdt_df['FSP'] == fsp, 'WSE'] = last_wse
                            vdt_df.loc[vdt_df['FSP'] == fsp, 'DoF'] = last_wse - filled_dem[_pixel_to_rc(fsp, ncols)]

                        for fsp_to_add in to_do:
                            rows_to_add.append((
                                fsp_to_add,
                                current_wse - filled_dem[_pixel_to_rc(fsp_to_add, ncols)],
                            ))
                            wses.append(current_wse)
                            xs.append(i)
                            elevs.append(filled_dem[_pixel_to_rc(fsp_to_add, ncols)])
                            bathys.append(dem_with_bathymetry[_pixel_to_rc(fsp_to_add, ncols)])
                            wses_raw.append(np.nan)
                    else:
                        last_wse_before_update = last_wse
                        # Interpolate DoF values for the missing FSPs between last and fsp
                        current_wse = vdt_df.loc[vdt_df['FSP'] == fsp, 'WSE'].values[0]
                        if current_wse < last_wse:
                            last_wse = current_wse
                        elif current_wse > last_wse + 0.1: # ALlow downstream to rise 10 cm
                            # Reassign wse
                            current_wse = last_wse
                            vdt_df.loc[vdt_df['FSP'] == fsp, 'WSE'] = last_wse
                            vdt_df.loc[vdt_df['FSP'] == fsp, 'DoF'] = last_wse - filled_dem[_pixel_to_rc(fsp, ncols)]

                        for j, fsp_to_add in enumerate(to_do, start=1):
                            WSE = np.interp(j, [0, len(to_do) + 1], [last_wse_before_update, current_wse])
                            rows_to_add.append((
                                fsp_to_add,
                                WSE - filled_dem[_pixel_to_rc(fsp_to_add, ncols)],
                            ))
                            wses.append(WSE)
                            xs.append(i)
                            elevs.append(filled_dem[_pixel_to_rc(fsp_to_add, ncols)])
                            bathys.append(dem_with_bathymetry[_pixel_to_rc(fsp_to_add, ncols)])
                            wses_raw.append(np.nan)
                    to_do = []
                    wses_raw.pop()
                    wses_raw.append(raw_wse)
                else:
                    current_wse = vdt_df.loc[vdt_df['FSP'] == fsp, 'WSE'].values[0]
                    wses_raw.append(current_wse)
                    if current_wse < last_wse:
                        last_wse = current_wse
                    elif current_wse > last_wse + 0.1: # ALlow upstream to rise 10 cm
                        current_wse = last_wse
                        # Reassign wse
                        vdt_df.loc[vdt_df['FSP'] == fsp, 'WSE'] = last_wse
                        vdt_df.loc[vdt_df['FSP'] == fsp, 'DoF'] = last_wse - filled_dem[_pixel_to_rc(fsp, ncols)]
                    wses.append(current_wse)
                    xs.append(i)
                    elevs.append(filled_dem[_pixel_to_rc(fsp, ncols)])
                    bathys.append(dem_with_bathymetry[_pixel_to_rc(fsp, ncols)])

                last_fsp = fsp
            else:
                to_do.append(fsp)

            fsp = _next_downstream(fsp, fdr, nrows, ncols)
            if fsp == -1:
                break

        if to_do and last_fsp is not None:
            # No values at the end of a segment; use the same DoF as the last known pixel, and ensure that it is no deeper than the last wse
                dof_last = vdt_df.loc[vdt_df['FSP'] == last_fsp, 'DoF'].values[0]
                for fsp_to_add in to_do:
                    elev = filled_dem[_pixel_to_rc(fsp_to_add, ncols)]
                    last_wse = min(elev + dof_last, last_wse)
                    rows_to_add.append((
                        fsp_to_add,
                        last_wse - elev,
                    ))
                    wses.append(last_wse)
                    xs.append(i)
                    elevs.append(elev)
                    wses_raw.append(np.nan)
                    bathys.append(dem_with_bathymetry[_pixel_to_rc(fsp_to_add, ncols)])

        last_dof = last_wse - filled_dem[_pixel_to_rc(last_fsp, ncols)]
        last_dof_dict[stream_id] = last_dof

        # import matplotlib.pyplot as plt
        # mask = ~np.isnan(wses_raw)

        # def butter_lowpass_filter(data, cutoff, fs, order=4):
        #     nyq = 0.5 * fs 
        #     normal_cutoff = cutoff / nyq
        #     b, a = butter(order, normal_cutoff, btype='low', analog=False)
        #     return filtfilt(b, a, data, padlen=min(3 * max(len(b), len(a)), len(data) - 1))

        # fs = 100        
        # cutoff = 2.0    

        # smoothed = butter_lowpass_filter(np.asarray(wses_raw)[mask], cutoff=cutoff, fs=fs)

        # plt.plot(xs, wses, label='Selected WSE')
        # plt.plot(xs, elevs, label='DEM Elevation', color='black')
        # plt.plot(xs, wses_raw, label='Raw WSE', color='red', linestyle='dashed')
        # plt.plot(xs, bathys, label='Bathymetry', color='blue', linestyle='solid')
        # plt.title(f"Water Surface Elevation, Thalweg, and Bathymetry Thalweg across {stream_id}")
        # plt.xlabel('Pixel Index (1 pixel = 30m)')
        # plt.ylabel('Elevation (m)')
        # plt.legend()
        # plt.show()

    vdt_df = vdt_df[['FSP', 'DoF']].copy()
    if rows_to_add:
        vdt_df = pd.concat([vdt_df, pd.DataFrame(rows_to_add, columns=['FSP', 'DoF'])], ignore_index=True)

    max_DoF = vdt_df['DoF'].max()
    fldpln_library = fldpln_library[fldpln_library['DTF'] < max_DoF]

    # merge the two dataframes on the row and column indices
    fldpln_library = pd.merge(fldpln_library, vdt_df, on='FSP', how='inner')
    fldpln_library['DTF'] = fldpln_library['DoF'] - fldpln_library['DTF']
    fldpln_library = fldpln_library.groupby(['FPP'], as_index=False).agg({'DTF': 'max', 'sink fill depth': 'first'})
    fldpln_library['DTF'] = fldpln_library['DTF'] + fldpln_library['sink fill depth']

    wse_array = np.full_like(dem, np.nan, dtype=np.float32)
    fsp = fldpln_library['FPP'].astype(int)
    rows = (fsp // ncols).astype(int)
    cols = (fsp % ncols).astype(int)
    wse_array[rows, cols] = fldpln_library['DTF'].values + dem[rows, cols]
    mask = (dem_with_bathymetry > -9998) & (wse_array > dem_with_bathymetry)
    wse_array[~mask] = np.nan

    return wse_array
