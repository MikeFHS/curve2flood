import numpy as np
from numba import njit
from numba.parfors.parfor import register_jitable

# Neighbour order:
# 1 2 3
# 4 x 5
# 6 7 8
# INFLOW/OUTFLOW are indexed by this ordering, and the solvers' flat neighbour
# offsets mirror it as ``dr * ncols + dc``.
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

@register_jitable(cache=True, nogil=True, forceinline=True)
def _pixel_to_rc(pixel: int, ncols: int) -> tuple[int, int]:
    return pixel // ncols, pixel % ncols

@register_jitable(cache=True, nogil=True, forceinline=True)
def _rc_to_pixel(row: int, col: int, ncols: int) -> int:
    return np.int32(row * ncols + col)

@register_jitable(cache=True, nogil=True)
def _next_downstream(pixel: int, fdr: np.ndarray, nrows: int, ncols: int) -> int:
    """The pixel ``pixel`` drains into, or -1 at a sink or the raster edge."""
    fd = fdr[pixel]
    if fd == 0:
        return -1

    # Whitebox D8 code -> (row, col) step.  Branches, not a lookup table: a
    # table built on every call was most of the cost of walking the network.
    row, col = _pixel_to_rc(pixel, ncols)
    if fd == 1:
        row -= 1
        col += 1
    elif fd == 2:
        col += 1
    elif fd == 4:
        row += 1
        col += 1
    elif fd == 8:
        row += 1
    elif fd == 16:
        row += 1
        col -= 1
    elif fd == 32:
        col -= 1
    elif fd == 64:
        row -= 1
        col -= 1
    elif fd == 128:
        row -= 1
    else:
        raise ValueError("Not a WhiteboxTools D8 flow direction code.")
    if row < 0 or row >= nrows or col < 0 or col >= ncols:
        return -1
    return _rc_to_pixel(row, col, ncols)


# Whitebox D8
# 64  128 1
# 32  x   2
# 16  8   4

# ESRI D8
# 32  64 128
# 16  x   1
# 8   4   2

# Whitebox.  INFLOW[pos] is the code a neighbour at NEIGHBOR_DELTAS[pos] carries
# when it drains into the centre cell; OUTFLOW[pos] is the code the centre cell
# carries when it drains into that neighbour.
INFLOW = np.array([4, 8, 16, 2, 32, 1, 128, 64], dtype=np.int32)
OUTFLOW = np.array([64, 128, 1, 32, 2, 16, 8, 4], dtype=np.int32)

@njit(cache=True, nogil=True, inline="always")
def _neighbour_bits_of_row(fil, fdr, ncols, bg, row, inflow, has_data):
    for pixel in range(row * ncols + 1, (row + 1) * ncols - 1):
        drains, data = 0, 0
        for pos in range(8):
            dr, dc = NEIGHBOR_DELTAS[pos]
            neighbor = pixel + dr * ncols + dc
            if fdr[neighbor] == INFLOW[pos]:
                drains |= 1 << pos
            if fil[neighbor] != bg:
                data |= 1 << pos

        inflow[pixel] = drains
        has_data[pixel] = data


# The iterative solver's arrays, one value per cell of its window (``_iterative_arrays``), in this order:
#  0-4    dtf, fsp, in_flood, on_boundary, flood_depths: the floodplain, set only at the cells it floods; a cell's
#         DTF and FSP mean nothing until in_flood says it is flooded, and a dry cell's DTF is taken as _UNREACHED
#  5-13   dirty, boundary, added, stack, fill_pixels, fill_dtf, path, touched, flooded: lists of cells
#  14-17  cand_avail, cand_belev, cand_fsp, cand_dtf: spillover candidates, read only where cand_stamp is current
#  18-19  spilled_depth, spilled_dtf: the path depth and DTF a cell was last backfilled from, read only where
#         spilled_round is current
#  20-25  dirty_stamp, visited, seen, on_path, cand_stamp, spilled_round: generation stamps
#  26     wet: which neighbours are flooded (see _flood), set only next to the cells it floods
# ``counters`` holds the last stamp and spill round, so that stamps go on rising when the arrays are reused.

def _iterative_arrays(n: int) -> tuple[np.ndarray, ...]:
    """The iterative solver's arrays for a window of ``n`` cells, as it starts.

    Allocated by numpy, not numba, whose zeros are written out: numpy's are
    the operating system's, and take no memory until the solver writes them.
    So DTF and FSP start at 0, not filled with a mark of their own: a cell's
    are read only once it floods.
    """
    return (np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.int32),
            np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.float32),
            np.empty(n, dtype=np.int32), np.empty(n, dtype=np.int32), np.empty(n, dtype=np.int32),
            np.empty(n, dtype=np.int32), np.empty(n, dtype=np.int32), np.empty(n, dtype=np.float32),
            np.empty(n + 2, dtype=np.int32), np.empty(n, dtype=np.int32), np.empty(n, dtype=np.int32),
            np.empty(n, dtype=np.float32), np.empty(n, dtype=np.float32), np.empty(n, dtype=np.int32),
            np.empty(n, dtype=np.float32), np.empty(n, dtype=np.float64), np.empty(n, dtype=np.float32),
            np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32),
            np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32),
            np.zeros(n, dtype=np.uint8))