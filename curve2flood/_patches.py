"""Patches of cells: the ones that hold a seed, found a row's runs of cells at a time."""
import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def patches_holding(cells: np.ndarray, seeds: np.ndarray, eight: bool) -> np.ndarray:
    """Where ``cells`` is not 0, in the patches that hold a cell where ``seeds`` is above 0: as
    ``scipy.ndimage.label`` would find the patches, connected through 4 sides, or 8 neighbours with ``eight``.

    Each row's runs of cells are joined (union-find) to the runs of the next
    row that share a side with them, or a corner too, rather than every cell
    labelled: a raster of labels takes four bytes a cell.
    """
    nrows, ncols = cells.shape
    gap = 1 if eight else 0
    # Each row's runs: run k covers columns first[k] to last[k] of its row
    counts = np.zeros(nrows + 1, dtype=np.int64)
    for r in prange(nrows):
        n = 0
        inside = False
        for c in range(ncols):
            wet = cells[r, c] != 0
            if wet and not inside:
                n += 1
            inside = wet
        counts[r + 1] = n

    starts = np.cumsum(counts)
    first = np.empty(starts[-1], dtype=np.int32)
    last = np.empty(starts[-1], dtype=np.int32)
    seeded = np.zeros(starts[-1], dtype=np.bool_)
    for r in prange(nrows):
        k = starts[r]
        inside = False
        for c in range(ncols):
            wet = cells[r, c] != 0
            if wet and not inside:
                first[k] = c
            if wet and seeds[r, c] > 0:
                seeded[k] = True
            if inside and not wet:
                last[k] = c - 1
                k += 1
            inside = wet
        if inside:
            last[k] = ncols - 1

    # Runs of neighbouring rows that touch are one patch
    parent = np.arange(starts[-1]).astype(np.int32)
    for r in range(nrows - 1):
        i, i_end = starts[r], starts[r + 1]
        j, j_end = starts[r + 1], starts[r + 2]
        while i < i_end and j < j_end:
            if last[i] + gap < first[j]:
                i += 1
            elif last[j] + gap < first[i]:
                j += 1
            else:
                a, b = _root(parent, i), _root(parent, j)
                if a != b:
                    parent[max(a, b)] = min(a, b)
                if last[i] < last[j]:
                    i += 1
                else:
                    j += 1

    kept = np.zeros(starts[-1], dtype=np.bool_)
    for k in range(starts[-1]):
        if seeded[k]:
            kept[_root(parent, k)] = True

    for k in range(starts[-1]):
        kept[k] = kept[_root(parent, k)]

    out = np.zeros((nrows, ncols), dtype=np.bool_)
    for r in prange(nrows):
        for k in range(starts[r], starts[r + 1]):
            if kept[k]:
                out[r, first[k]:last[k] + 1] = True
                
    return out


@njit(cache=True, inline="always")
def _root(parent: np.ndarray, k: int) -> int:
    """The root of ``k``'s tree, halving the path to it on the way."""
    while parent[k] != k:
        parent[k] = parent[parent[k]]
        k = parent[k]
    return k
