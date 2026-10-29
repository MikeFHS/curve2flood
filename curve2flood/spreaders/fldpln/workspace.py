import numpy as np
from numba import njit

from .solvers import _iterative_arrays, _solve_iterative_reusing

# Segments a Workspace solves before it starts again on new arrays: the cells segments flood take memory until then
_RENEW_EVERY = 8

class Workspace:
    """One thread's iterative solver, on the whole raster, kept from one segment to the next.

    The workspace solves every segment once, on
    the whole raster with the windows' one-cell halo, which a window that
    the floodplain doesn't reach gives the same result as.  Its arrays,
    about 100 bytes a cell, take memory only as segments flood cells, and
    each solve resets the cells it flooded.  Every ``_RENEW_EVERY``
    segments it starts again on new arrays, which hands back the memory of
    the cells the last ones flooded.

    ``fil`` and ``fdr`` are the filled DEM and flow directions with the halo
    (``_crop`` of the whole raster), flattened, and ``inflow`` and
    ``has_data`` their ``_neighbour_bits``; ``channel`` is
    ``_channel_lookup``'s, empty without a water mask.  The threads share
    them.
    """

    def __init__(self, fil: np.ndarray, fdr: np.ndarray, inflow: np.ndarray, has_data: np.ndarray,
                 channel: tuple[np.ndarray, np.ndarray, np.ndarray]):
        self.fil, self.fdr, self.inflow, self.has_data, self.channel = fil, fdr, inflow, has_data, channel
        self.excluded = np.zeros(fil.size, dtype=np.bool_)
        self.arrays = _iterative_arrays(fil.size)
        self.counters = np.zeros(2, dtype=np.int64)
        self.solved = 0

    def solve(self, sources: np.ndarray, downstream: np.ndarray, dh: float, fldmn: float, fldmx: float,
              iterative_spill: bool, global_max_wse: float, bg: float, spill_decay: float, ncols: int):
        """``_solve_iterative`` of the raster (``ncols`` wide), with ``sources`` as ids of the raster with its halo
        and ``downstream`` (TD(R)'s stream pixels, which the channel widens) as its own; the cells and sources
        come out as the raster's own ids."""
        if self.solved == _RENEW_EVERY or self.counters.max() > 2 ** 30:
            # Each solve leaves the arrays as it found them, but for the generation stamps, which are only ever
            # compared with the current one: new arrays, with the stamps started again from 0, solve the same, and
            # the old ones' memory goes back to the operating system (an array of zeros takes none until written)
            self.arrays = _iterative_arrays(self.fil.size)
            self.counters[:] = 0
            self.solved = 0
        self.solved += 1
        _mark_downstream(self.excluded, downstream, *self.channel, ncols, True)
        try:
            return _solve_iterative_reusing(self.fil, self.fdr, self.inflow, self.has_data, ncols + 2, sources,
                                            self.excluded, dh, fldmn, fldmx, iterative_spill, global_max_wse, bg,
                                            spill_decay, self.arrays, self.counters, 1, ncols)
        finally:
            _mark_downstream(self.excluded, downstream, *self.channel, ncols, False)


@njit(cache=True, nogil=True)
def _mark_downstream(excluded: np.ndarray, downstream: np.ndarray, keys: np.ndarray, starts: np.ndarray,
                     cells: np.ndarray, ncols: int, value: bool):
    """Set ``excluded`` (ids of the raster with its halo) to ``value`` at TD(R)'s pixels (``downstream``, raster
    ids, ``ncols`` wide) and at the channel cells of their groups (``_channel_lookup``): ``_widen_downstream``
    without building the list."""
    wc = ncols + 2
    for pixel in downstream:
        row = pixel // ncols
        excluded[(row + 1) * wc + pixel - row * ncols + 1] = value
        if keys.size:
            group = np.searchsorted(keys, pixel)
            if group < keys.size and keys[group] == pixel:
                for cell in cells[starts[group]:starts[group + 1]]:
                    row = cell // ncols
                    excluded[(row + 1) * wc + cell - row * ncols + 1] = value
