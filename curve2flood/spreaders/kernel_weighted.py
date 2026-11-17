"""Kernel-weighted flood mapper: spreads each stream cell's water surface over its neighbourhood."""
import numba
import numpy as np
from numba import njit, prange

# Cell states of FloodAllLocalAreas
_DRY, _WET, _REACHED = 0, 1, 2

@njit(cache=True)
def FloodAllLocalAreas(WSE, E, r_min, r_max, c_min, c_max, r_use, c_use, state):
    """Cells of ``E[r_min:r_max, c_min:c_max]`` at or below ``WSE`` that connect to the stream cell ``(r_use, c_use)``.

    Fills ``state``, at least ``(r_max - r_min + 2, c_max - c_min + 2)``, with
    ``_REACHED`` for those cells, offset by one row and column (a dry border).
    The connection is traced in a few sweeps, so a winding path can be
    missed.  The stream cell is always reached.
    """
    nrows_local = r_max - r_min + 2
    ncols_local = c_max - c_min + 2
    for r in range(nrows_local):
        for c in range(ncols_local):
            state[r, c] = _DRY
    for r in range(r_max - r_min):
        for c in range(c_max - c_min):
            if E[r_min + r, c_min + c] <= WSE:
                state[r + 1, c + 1] = _WET

    r_idx = r_use - r_min + 1
    c_idx = c_use - c_min + 1
    state[r_idx, c_idx] = _REACHED

    # Spread the mark to the wet neighbours of reached cells: down and right
    # of the stream cell, then up and left of it, then over the whole window
    for r in range(r_idx, nrows_local - 1):
        for c in range(c_idx, ncols_local - 1):
            if state[r, c] == _REACHED:
                _reach_neighbours(state, r, c)
    for r in range(r_idx, 0, -1):
        for c in range(c_idx, 0, -1):
            if state[r, c] == _REACHED:
                _reach_neighbours(state, r, c)
    for r in range(1, nrows_local - 1):
        for c in range(1, ncols_local - 1):
            if state[r, c] == _REACHED:
                _reach_neighbours(state, r, c)
    return state

@njit(cache=True, inline="always")
def _reach_neighbours(state, r, c):
    for a in range(r - 1, r + 2):
        for b in range(c - 1, c + 2):
            if state[a, b] == _WET:
                state[a, b] = _REACHED

# Bytes of FloodAllLocalAreas' windows worked out at once, in parallel, before they are spread
_LOCAL_BATCH_BYTES = 1 << 28

def create_kernel_weighted_spread_map(
    RR,
    CC,
    T_Rast,
    W_Rast,
    S_Rast,
    E,
    nrows,
    ncols,
    TW_m,
    dx,
    dy,
    LocalFloodOption,
    WeightBox,
    TW_for_WeightBox_ElipseMask,
    TopWidthPlausibleLimit,
    Set_Depth,
    V_Rast=None,
    channel=None,
):
    """Spread each stream cell's WSE over its neighbourhood.

    Stream cell ``(RR[i], CC[i])`` of the padded DEM ``E`` contributes its WSE
    (``W_Rast``) to the square window of cells up to one top width
    (``T_Rast``, capped at ``TW_m``) away, weighted by ``WeightBox``.  With
    ``LocalFloodOption`` it only reaches cells below its WSE that connect to it.  With ``Set_Depth > 0``
    each stream cell's WSE is its DEM elevation plus ``Set_Depth`` and its top
    width ``TopWidthPlausibleLimit``.

    Returns ``(WSE, total weight, slope times weight)`` on the padded grid;
    WSE is NaN where it is not above the DEM.  ``S_Rast`` (slopes) may be None,
    and the slope sums are then a 1 x 1 placeholder.  Given the stream cells'
    valley slopes ``V_Rast`` and the ``channel`` cells (unpadded, as
    ``S_Rast``), a cell outside the channel sums the valley slopes instead.

    The sums are added cell by cell, with the arithmetic of the array sums
    this replaced (a WSE times a weight in float64, a slope times a weight in
    float32, each added to a float32 sum), so the result is the same to the
    last bit without a temporary array per stream cell.  They are added in
    parallel, a band of rows each (``_row_bands``), but every cell still
    takes the stream cells' contributions in their order, so the sums, and
    their rounding, are those of one stream cell at a time.  The rasters are
    read at the stream cells alone (see ``spread_stream_cells``).
    """
    at = (RR - 1, CC - 1)
    return spread_stream_cells(RR, CC, T_Rast[at], W_Rast[at], None if S_Rast is None else S_Rast[at], E, nrows, ncols,
                               TW_m, dx, dy, LocalFloodOption, WeightBox, TW_for_WeightBox_ElipseMask,
                               TopWidthPlausibleLimit, Set_Depth, None if V_Rast is None else V_Rast[at], channel)

def spread_stream_cells(RR, CC, top_widths, water_surfaces, slopes, E, nrows, ncols, TW_m, dx, dy, LocalFloodOption,
                        WeightBox, TW_for_WeightBox_ElipseMask, TopWidthPlausibleLimit, Set_Depth, valley_slopes=None,
                        channel=None):
    """``create_kernel_weighted_spread_map`` with each stream cell's top width, WSE, slope and valley slope as
    vectors, ``top_widths[i]`` for the cell ``(RR[i], CC[i])``, rather than rasters of every cell."""
    return _spread(RR, CC, top_widths, water_surfaces, slopes, E, nrows, ncols, TW_m, dx, dy, LocalFloodOption,
                   WeightBox, TW_for_WeightBox_ElipseMask, TopWidthPlausibleLimit, Set_Depth, valley_slopes, channel,
                   4 * numba.get_num_threads())

@njit(cache=True, parallel=True)
def _spread(RR, CC, T_cells, W_cells, S_cells, E, nrows, ncols, TW_m, dx, dy, LocalFloodOption, WeightBox,
            TW_for_WeightBox_ElipseMask, TopWidthPlausibleLimit, Set_Depth, V_cells, channel, n_bands):
    """``spread_stream_cells``, its rows in ``n_bands`` bands."""
    WSE_Times_Weight = np.zeros((nrows + 2, ncols + 2), dtype=np.float32)
    Total_Weight = np.zeros((nrows + 2, ncols + 2), dtype=np.float32)
    # Each a test of an argument alone, so that numba drops the branches of one that isn't given
    has_slope = S_cells is not None
    has_valley = V_cells is not None
    has_channel = channel is not None
    Slope_Times_Weight = np.zeros((nrows + 2, ncols + 2) if has_slope else (1, 1), dtype=np.float32)

    # Each stream cell's water, slopes and window, or skip[i] where it spreads nothing
    num_nonzero = len(RR)
    skip = np.ones(num_nonzero, dtype=np.bool_)
    water = np.zeros(num_nonzero)
    slopes = np.zeros(num_nonzero)
    valley_slopes = np.zeros(num_nonzero)
    windows = np.zeros((num_nonzero, 6), dtype=np.int64)  # r_min, r_max, c_min, c_max, w_r_min, w_c_min
    for i in range(num_nonzero):
        r = RR[i]
        c = CC[i]
        if Set_Depth>0.0:
            WSE = float(E[r,c] + Set_Depth)
            COMID_TW_m = TopWidthPlausibleLimit
        else:
            # ARC results for this cell
            WSE = np.round(W_cells[i], 2)
            COMID_TW_m = T_cells[i]

        if COMID_TW_m < 0.00001 or (WSE - E[r,c]) < 0.001:
            continue

        SLOPE = 0.0
        if has_slope:
            SLOPE = float(S_cells[i])
        # Outside the channel, the valley's slope (the stream's without valley slopes)
        VALLEY_SLOPE = SLOPE
        if has_valley:
            VALLEY_SLOPE = float(V_cells[i])

        COMID_TW_m = min(COMID_TW_m, TW_m)

        # How many cells the top width reaches from the stream cell
        COMID_TW = int(max(np.round(COMID_TW_m / dx), np.round(COMID_TW_m / dy)))

        r_min = max(r - COMID_TW, 1)
        r_max = min(r + COMID_TW + 1, nrows + 1)
        c_min = max(c - COMID_TW, 1)
        c_max = min(c + COMID_TW + 1, ncols + 1)

        # The weight box is centred on TW_for_WeightBox_ElipseMask, the half
        # width it was built with, not on this cell's top width
        w_r_min = TW_for_WeightBox_ElipseMask - (r - r_min)
        w_c_min = TW_for_WeightBox_ElipseMask - (c - c_min)

        skip[i] = False
        water[i] = WSE
        slopes[i] = SLOPE
        valley_slopes[i] = VALLEY_SLOPE
        windows[i, 0], windows[i, 1], windows[i, 2], windows[i, 3] = r_min, r_max, c_min, c_max
        windows[i, 4], windows[i, 5] = w_r_min, w_c_min

    bands = _row_bands(skip, windows, nrows + 2, n_bands)
    if LocalFloodOption:
        # The windows' FloodAllLocalAreas, a batch at a time in parallel, then spread like the others.  Window i's
        # cells are offsets[i] to offsets[i + 1] of the stream cells' in a row.
        offsets = np.zeros(num_nonzero + 1, dtype=np.int64)
        for i in range(num_nonzero):
            size = 0
            if not skip[i]:
                size = (windows[i, 1] - windows[i, 0] + 2) * (windows[i, 3] - windows[i, 2] + 2)
            offsets[i + 1] = offsets[i] + size

        first = 0
        while first < num_nonzero:
            last = first + 1
            while last < num_nonzero and offsets[last + 1] - offsets[first] <= _LOCAL_BATCH_BYTES:
                last += 1
            states = np.empty(offsets[last] - offsets[first], dtype=np.uint8)
            for i in prange(first, last):
                if skip[i]:
                    continue
                r_min, r_max, c_min, c_max = windows[i, 0], windows[i, 1], windows[i, 2], windows[i, 3]
                state = states[offsets[i] - offsets[first]:offsets[i + 1] - offsets[first]].reshape(
                    (r_max - r_min + 2, c_max - c_min + 2))
                FloodAllLocalAreas(water[i], E, r_min, r_max, c_min, c_max, RR[i], CC[i], state)

            for k in prange(bands.size - 1):
                for i in range(first, last):
                    if skip[i]:
                        continue
                    r_min, r_max, c_min, c_max = windows[i, 0], windows[i, 1], windows[i, 2], windows[i, 3]
                    w_r_min, w_c_min = windows[i, 4], windows[i, 5]
                    state = states[offsets[i] - offsets[first]:offsets[i + 1] - offsets[first]].reshape(
                        (r_max - r_min + 2, c_max - c_min + 2))
                    WSE = water[i]
                    SLOPE = slopes[i]
                    VALLEY_SLOPE = valley_slopes[i]
                    # Every cell of the window adds its weight times 0 or 1, as the
                    # masked array sums did (a NaN slope reaches the unreached cells)
                    width = c_max - c_min
                    for a in range(max(r_min, bands[k]) - r_min, min(r_max, bands[k + 1]) - r_min):
                        rr = r_min + a
                        # The window's row of each, as a line of its own
                        weights = WeightBox[w_r_min + a, w_c_min:w_c_min + width]
                        state_row = state[a + 1, 1:width + 1]
                        wse_sums = WSE_Times_Weight[rr, c_min:c_max]
                        totals = Total_Weight[rr, c_min:c_max]
                        for b in range(width):
                            reached = 1.0 if state_row[b] == _REACHED else 0.0
                            weight = weights[b]
                            wse_sums[b] = np.float32(np.float64(wse_sums[b]) + WSE * np.float64(weight) * reached)
                            totals[b] = np.float32(np.float64(totals[b]) + np.float64(weight) * reached)
                        if has_slope:
                            slope_sums = Slope_Times_Weight[rr, c_min:c_max]
                            for b in range(width):
                                reached = 1.0 if state_row[b] == _REACHED else 0.0
                                slope = SLOPE
                                if has_channel and not channel[rr - 1, c_min + b - 1]:
                                    slope = VALLEY_SLOPE
                                slope_sums[b] = np.float32(np.float64(slope_sums[b])
                                                           + np.float64(np.float32(slope) * weights[b]) * reached)
            first = last
    else:
        for k in prange(bands.size - 1):
            for i in range(num_nonzero):
                if skip[i]:
                    continue
                r_min, r_max, c_min, c_max = windows[i, 0], windows[i, 1], windows[i, 2], windows[i, 3]
                w_r_min, w_c_min = windows[i, 4], windows[i, 5]
                WSE = water[i]
                SLOPE = slopes[i]
                VALLEY_SLOPE = valley_slopes[i]
                width = c_max - c_min
                for a in range(max(r_min, bands[k]) - r_min, min(r_max, bands[k + 1]) - r_min):
                    rr = r_min + a
                    # The window's row of each, as a line of its own: the loop along it vectorizes
                    weights = WeightBox[w_r_min + a, w_c_min:w_c_min + width]
                    wse_sums = WSE_Times_Weight[rr, c_min:c_max]
                    totals = Total_Weight[rr, c_min:c_max]
                    for b in range(width):
                        wse_sums[b] = np.float32(np.float64(wse_sums[b]) + WSE * np.float64(weights[b]))
                        totals[b] += weights[b]

                    # A weighted slope, for velocity: the valley's outside the channel, where there is one
                    if has_slope:
                        slope_sums = Slope_Times_Weight[rr, c_min:c_max]
                        for b in range(width):
                            slope = SLOPE
                            if has_channel and not channel[rr - 1, c_min + b - 1]:
                                    slope = VALLEY_SLOPE
                            slope_sums[b] += np.float32(slope) * weights[b]

    # Flood where the weighted WSE is above a valid elevation; the weighted
    # WSE takes the place of the WSE x weight sums
    nan = np.float32(np.nan)
    for a in prange(nrows + 2):
        for b in range(ncols + 2):
            elevation = E[a, b]
            total = Total_Weight[a, b]
            if elevation > -9998.0 and WSE_Times_Weight[a, b] > elevation * total:
                WSE_Times_Weight[a, b] = WSE_Times_Weight[a, b] / total
            else:
                WSE_Times_Weight[a, b] = nan

    return WSE_Times_Weight, Total_Weight, Slope_Times_Weight

@njit(cache=True)
def _row_bands(skip, windows, nrows, n_bands):
    """Rows ``bands[k]`` to ``bands[k + 1]`` (exclusive) of ``n_bands`` bands with about as many window cells each:
    the stream cells' windows (``windows[i, 0:4]``, rows ``r_min`` to ``r_max`` and columns ``c_min`` to ``c_max``,
    unless ``skip[i]``) cover cells of some rows far more than others."""
    work = np.zeros(nrows + 1)
    for i in range(skip.size):
        if not skip[i]:
            width = windows[i, 3] - windows[i, 2]
            work[windows[i, 0]] += width
            work[windows[i, 1]] -= width
            
    total = np.cumsum(np.cumsum(work)[:nrows])
    bands = np.empty(n_bands + 1, dtype=np.int64)
    bands[0] = 0
    for k in range(1, n_bands):
        bands[k] = np.searchsorted(total, total[-1] * k / n_bands) if total.size else 0
    bands[n_bands] = nrows
    for k in range(1, n_bands + 1):
        bands[k] = max(bands[k], bands[k - 1])
    return bands
