"""Small tests of the flood mappers' building blocks, on toy grids."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from numba import njit
from osgeo import gdal, ogr
from scipy import ndimage as ndi
from shapely.geometry import Point

from curve2flood import core
from curve2flood._patches import patches_holding
from curve2flood.spreaders import fldpln, kernel_weighted, multi_point
from curve2flood.spreaders.fldpln import library as fldpln_library
from curve2flood.spreaders.fldpln import library_writer
from curve2flood.spreaders.fldpln import mapping as fldpln_mapping
from curve2flood.spreaders.kernel_weighted import create_kernel_weighted_spread_map

# --------------------------------------------------------------------------
# Kernel-weighted mapper
# --------------------------------------------------------------------------

def _valley():
    """A flat 7x7 DEM (padded by one cell) at 100 m, with a ridge and low ground behind it on the east side."""
    E = np.full((9, 9), -9999.0, np.float32)
    E[1:-1, 1:-1] = 100.0
    E[1:-1, 6] = 100.5  # a ridge
    E[1:-1, 7] = 99.0   # low ground behind the ridge
    return E


def _spread(local, set_depth):
    """One stream cell in the middle of ``_valley``, with a 30 m top width and its WSE 0.2 m above the ground."""
    E = _valley()
    T = np.full((7, 7), -1.0, np.float32)
    W = np.full((7, 7), np.nan, np.float32)
    T[3, 3], W[3, 3] = 30.0, 100.2
    weights = core.create_weightbox(3, 10.0, 10.0)
    wse, _, _ = create_kernel_weighted_spread_map(
        np.array([4]), np.array([4]), T, W, None, E, 7, 7, 30.0, 10.0, 10.0, local, weights, 3, 30.0, set_depth)
    return wse


@pytest.mark.parametrize("set_depth", [-1.0, 0.2])
def test_kernel_floods_the_cells_below_the_water(set_depth):
    """The cell's own WSE and top width, or Set_Depth and TopWidthPlausibleLimit, give the same flood here."""
    wse = _spread(local=False, set_depth=set_depth)
    wet = np.isfinite(wse)
    assert wet[1:8, [1, 2, 3, 4, 5, 7]].all()  # everything within the top width but the ridge
    assert wet.sum() == 7 * 6
    assert np.allclose(wse[wet], 100.2)


def test_kernel_local_flood_stops_at_the_ridge():
    wet = np.isfinite(_spread(local=True, set_depth=-1.0))
    assert wet[1:8, 1:6].all() and wet.sum() == 7 * 5


@pytest.mark.parametrize("local", [False, True])
def test_kernel_slopes_are_the_valley_s_outside_the_channel(local):
    """_spread's stream cell, its slope 0.001 and its valley's 0.004: the weighted slope is 0.001 in the channel
    (its own column) and 0.004 beyond it, and 0.001 everywhere without valley slopes."""
    E = _valley()
    T = np.full((7, 7), -1.0, np.float32)
    W, S, V = (np.full((7, 7), np.nan, np.float32) for _ in range(3))
    T[3, 3], W[3, 3], S[3, 3], V[3, 3] = 30.0, 100.2, 0.001, 0.004
    channel = np.zeros((7, 7), np.bool_)
    channel[:, 3] = True
    weights = core.create_weightbox(3, 10.0, 10.0)
    args = (np.array([4]), np.array([4]), T, W, S, E, 7, 7, 30.0, 10.0, 10.0, local, weights, 3, 30.0, -1.0)

    wse, total, slope_sums = create_kernel_weighted_spread_map(*args, V, channel)
    plain = create_kernel_weighted_spread_map(*args)

    wet = np.isfinite(wse)[1:-1, 1:-1]
    assert wet[:, 3].all() and wet[:, [2, 4]].all()
    with np.errstate(invalid="ignore"):  # the cells it doesn't reach have no weight
        slopes = (slope_sums / total)[1:-1, 1:-1]
        plain_slopes = (plain[2] / plain[1])[1:-1, 1:-1]
    np.testing.assert_allclose(slopes[wet & channel], 0.001, rtol=1e-6)
    np.testing.assert_allclose(slopes[wet & ~channel], 0.004, rtol=1e-6)
    np.testing.assert_allclose(plain_slopes[wet], 0.001, rtol=1e-6)
    np.testing.assert_array_equal(plain[0], wse)  # the water is the same


@njit
def _local_mask(WSE, E_Box, r_min, r_max, c_min, c_max, r_use, c_use):
    """The kernel mapper's local flood as it was written, on window slices, for ``_slice_spread``."""
    FourMatrix = np.full((3, 3), 4)
    nrows_local = np.int32(r_max - r_min + 2)
    ncols_local = np.int32(c_max - c_min + 2)
    FloodLocal = np.zeros((nrows_local, ncols_local), dtype=np.float32)
    FloodLocal[1:nrows_local - 1, 1:ncols_local - 1] = np.where(E_Box <= WSE, 1, 0)
    FloodLocal[int(r_use - r_min + 1), int(c_use - c_min + 1)] = 4
    for r in range((r_use - r_min + 1), nrows_local - 1):
        for c in range((c_use - c_min + 1), ncols_local - 1):
            if FloodLocal[r, c] >= 3:
                FloodLocal[r - 1:r + 2, c - 1:c + 2] = FloodLocal[r - 1:r + 2, c - 1:c + 2] * FourMatrix
    for r in range((r_use - r_min + 1), 0, -1):
        for c in range((c_use - c_min + 1), 0, -1):
            if FloodLocal[r, c] >= 3:
                FloodLocal[r - 1:r + 2, c - 1:c + 2] = FloodLocal[r - 1:r + 2, c - 1:c + 2] * FourMatrix
    for r in range(1, nrows_local - 1):
        for c in range(1, ncols_local - 1):
            if FloodLocal[r, c] >= 3:
                FloodLocal[r - 1:r + 2, c - 1:c + 2] = FloodLocal[r - 1:r + 2, c - 1:c + 2] * FourMatrix
    return np.where(FloodLocal[1:nrows_local - 1, 1:ncols_local - 1] > 3.0, 1.0, 0.0)


@njit
def _slice_spread(RR, CC, T_Rast, W_Rast, S_Rast, E, nrows, ncols, TW_m, dx, dy, LocalFloodOption, WeightBox,
                  TW_for_WeightBox_ElipseMask, TopWidthPlausibleLimit, Set_Depth):
    """The kernel spread as it was written, adding a window slice per stream cell: the reference whose sums
    create_kernel_weighted_spread_map must reproduce to the last bit."""
    WSE_Times_Weight = np.zeros((nrows + 2, ncols + 2), dtype=np.float32)
    Slope_Times_Weight = np.zeros((nrows + 2, ncols + 2), dtype=np.float32)
    Total_Weight = np.zeros((nrows + 2, ncols + 2), dtype=np.float32)
    for i in range(len(RR)):
        r = RR[i]
        c = CC[i]
        if Set_Depth > 0.0:
            WSE = float(E[r, c] + Set_Depth)
            COMID_TW_m = TopWidthPlausibleLimit
        else:
            WSE = np.round(W_Rast[r - 1, c - 1], 2)
            COMID_TW_m = T_Rast[r - 1, c - 1]
        if COMID_TW_m < 0.00001 or (WSE - E[r, c]) < 0.001:
            continue
        if S_Rast is not None:
            SLOPE = float(S_Rast[r - 1, c - 1])
        COMID_TW_m = min(COMID_TW_m, TW_m)
        COMID_TW = int(max(np.round(COMID_TW_m / dx), np.round(COMID_TW_m / dy)))
        r_min = max(r - COMID_TW, 1)
        r_max = min(r + COMID_TW + 1, nrows + 1)
        c_min = max(c - COMID_TW, 1)
        c_max = min(c + COMID_TW + 1, ncols + 1)
        w_r_min = TW_for_WeightBox_ElipseMask - (r - r_min)
        w_r_max = TW_for_WeightBox_ElipseMask + (r_max - r)
        w_c_min = TW_for_WeightBox_ElipseMask - (c - c_min)
        w_c_max = TW_for_WeightBox_ElipseMask + (c_max - c)
        weight_slice = WeightBox[w_r_min:w_r_max, w_c_min:w_c_max]
        if LocalFloodOption:
            mask = _local_mask(WSE, E[r_min:r_max, c_min:c_max], r_min, r_max, c_min, c_max, r, c)
            WSE_Times_Weight[r_min:r_max, c_min:c_max] += (WSE * weight_slice * mask)
            Total_Weight[r_min:r_max, c_min:c_max] += weight_slice * mask
            if S_Rast is not None:
                Slope_Times_Weight[r_min:r_max, c_min:c_max] += (SLOPE * weight_slice * mask)
        else:
            cell_wse_weight = WSE * weight_slice
            WSE_Times_Weight[r_min:r_max, c_min:c_max] += cell_wse_weight
            Total_Weight[r_min:r_max, c_min:c_max] += weight_slice
            if S_Rast is not None:
                Slope_Times_Weight[r_min:r_max, c_min:c_max] += (SLOPE * weight_slice)
    valid_candidate = (E > -9998.0) & (WSE_Times_Weight > E * Total_Weight)
    WSE_array = np.where(valid_candidate, WSE_Times_Weight / Total_Weight, np.nan).astype(np.float32)
    return WSE_array, Total_Weight, Slope_Times_Weight


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("slopes", [False, True])
@pytest.mark.parametrize("set_depth", [-1.0, 0.3])
def test_kernel_sums_are_the_window_slice_sums_to_the_last_bit(local, slopes, set_depth):
    """Random ground with 40 stream cells, some top widths running off the grid, and a NaN slope."""
    rng = np.random.default_rng(7)
    nrows, ncols = 30, 25
    E = np.full((nrows + 2, ncols + 2), -9999.0, np.float32)
    E[1:-1, 1:-1] = 100.0 + 3.0 * rng.random((nrows, ncols), np.float32)
    E[5, 7] = -9999.0
    RR, CC = rng.integers(1, nrows + 1, 40), rng.integers(1, ncols + 1, 40)
    T = np.full((nrows, ncols), -1.0, np.float32)
    W = np.full((nrows, ncols), np.nan, np.float32)
    T[RR - 1, CC - 1] = rng.uniform(0.0, 90.0, 40)
    W[RR - 1, CC - 1] = E[RR, CC] + rng.uniform(-0.5, 2.0, 40)
    S = None
    if slopes:
        S = np.full((nrows, ncols), np.nan, np.float32)
        S[RR - 1, CC - 1] = rng.uniform(1e-4, 0.05, 40)
        S[RR[0] - 1, CC[0] - 1] = np.nan
    weights = core.create_weightbox(7, 10.0, 12.0)  # 70 m: 7 cells of 10 m, as Curve2Flood_MainFunction sizes it
    args = (RR, CC, T, W, S, E, nrows, ncols, 70.0, 10.0, 12.0, local, weights, 7, 70.0, set_depth)
    expected = _slice_spread(*args)
    actual = create_kernel_weighted_spread_map(*args)
    for i in range(3 if slopes else 2):
        assert np.array_equal(actual[i], expected[i], equal_nan=True)
    assert np.isfinite(actual[0]).sum() > 100
    assert not slopes or np.isnan(actual[2]).any()  # the NaN slope was spread


@pytest.mark.parametrize("local", [False, True])
def test_kernel_sums_are_the_same_in_any_number_of_bands(local):
    """Each band of rows adds the stream cells' weights in their order, so the sums are the same to the last bit
    however many bands there are, one or more than there are rows."""
    rng = np.random.default_rng(11)
    nrows, ncols = 40, 33
    E = np.full((nrows + 2, ncols + 2), -9999.0, np.float32)
    E[1:-1, 1:-1] = 100.0 + 3.0 * rng.random((nrows, ncols), np.float32)
    RR, CC = rng.integers(1, nrows + 1, 120), rng.integers(1, ncols + 1, 120)
    T = rng.uniform(0.0, 150.0, 120).astype(np.float32)
    W = (E[RR, CC] + rng.uniform(-0.5, 2.0, 120)).astype(np.float32)
    S = rng.uniform(1e-4, 0.05, 120).astype(np.float32)
    weights = core.create_weightbox(15, 10.0, 10.0)
    args = (RR, CC, T, W, S, E, nrows, ncols, 150.0, 10.0, 10.0, local, weights, 15, 150.0, -1.0, None, None)
    expected = kernel_weighted._spread(*args, 1)
    for n_bands in (2, 7, 64):
        for actual, single in zip(kernel_weighted._spread(*args, n_bands), expected):
            assert np.array_equal(actual, single, equal_nan=True)


# --------------------------------------------------------------------------
# FLDPLN mapper
# --------------------------------------------------------------------------

def _own_cells(library):
    """``library`` with a row for each stream pixel's own cell (pixels 5-9), as a real library has one for each of
    its sources, which gives the pixel's fill: 0 here, with too deep a DTF to flood anything, and a segment of its own,
    so it joins no segment's median."""
    df = library.collect()
    if "fill" not in df.columns:
        return library
    have = set(df.filter(pl.col("FSP") == pl.col("FPP"))["FSP"].to_list())
    pixels = [p for p in range(5, 10) if p not in have]
    own = {"FSP": pixels, "FPP": pixels, "DTF": [99.0] * len(pixels), "fill": [0.0] * len(pixels)}
    if "segment" in df.columns:
        own["segment"] = [100 + p for p in pixels]
    return pl.concat([df, pl.DataFrame(own, schema={c: df.schema[c] for c in own}).select(df.columns)]).lazy()


def _fldpln_map(library, wse=(101.0,) * 5, elev=(100.0,) * 5, reaches=(1,) * 5, own_cells=True, fill_below_dem=None,
                min_depth=0.0):
    """A 3x5 grid with a stream along the middle row, flowing east, 1 m deep by default (WSE 101 over a 100 m VDT
    Elev, with no fill). A NaN WSE is a stream pixel without a VDT value. ``reaches`` gives each stream pixel's
    reach: one reach by default. A LazyFrame library gets its stream pixels' own cells (``_own_cells``)."""
    fdr = np.zeros((3, 5), np.uint8)
    fdr[1, :4] = 2                # Whitebox east
    vdt = pl.DataFrame({"COMID": list(reaches), "Row": [1] * 5, "Col": list(range(5)), "Elev": list(elev),
                        "WSE": list(wse)})
    first = [i for i in range(5) if i == 0 or reaches[i] != reaches[i - 1]]
    last = [i - 1 for i in first[1:]] + [4]
    stream_info = pd.DataFrame({"start_pixel": [5 + i for i in first], "end_pixel": [5 + i for i in last],
                                "length": [j - i + 1 for i, j in zip(first, last)],
                                "stream_id": [reaches[i] for i in first]})
    if own_cells and isinstance(library, pl.LazyFrame):
        library = _own_cells(library)
    return fldpln.make_fldpln_flood_map(vdt, fdr, library, stream_info, fill_below_dem=fill_below_dem,
                                         min_depth=min_depth)


def test_fldpln_water_shallower_than_the_minimum_floods_nothing():
    """A 1 m stage whose water only just tops a 0.994 m rise leaves 6 mm over the cells beyond it (0 to 2), as down a
    fan from a stream on its side (reach 710777915, Salt Lake City): no flood with a 1 cm minimum, but the 0.9 m deep
    cell (14), and a filled pit's 6 mm plus its 0.5 m fill (3), still flood."""
    library = pl.LazyFrame({"FSP": [5, 5, 6, 9, 8], "FPP": [0, 1, 2, 14, 3], "DTF": [0.994, 0.994, 0.994, 0.1, 0.994],
                            "fill": [0.0, 0.0, 0.0, 0.0, 0.5]},
                           schema={"FSP": pl.Int32, "FPP": pl.Int32, "DTF": pl.Float32, "fill": pl.Float32})
    every = _fldpln_map(library).ravel()
    assert every[[0, 1, 2]] == pytest.approx([0.006] * 3, abs=1e-5)
    depth = _fldpln_map(library, min_depth=0.01).ravel()
    assert np.isnan(depth[[0, 1, 2]]).all()
    assert depth[14] == pytest.approx(0.9, abs=1e-5) and depth[3] == pytest.approx(0.506, abs=1e-5)
    assert np.isfinite(depth).sum() == 2


def test_fldpln_depth_is_the_water_left_plus_the_fill():
    library = pl.LazyFrame({
        "FSP": [5, 6, 7, 7, 8, 9, 9, 9],
        "FPP": [0, 6, 2, 12, 12, 14, 3, 4],
        "DTF": [0.1, 0.1, 0.5, 1.5, 0.8, 0.1, 1.3, 1.2],
        "fill": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0],
    }, schema={"FSP": pl.Int32, "FPP": pl.Int32, "DTF": pl.Float32, "fill": pl.Float32})
    depth = _fldpln_map(library).ravel()
    # Each library row whose water reaches the filled surface gives DoF - DTF + fill; a pixel keeps the deepest
    assert depth[[0, 6, 14]] == pytest.approx([0.9, 0.9, 0.9], abs=1e-5)
    assert depth[2] == pytest.approx(1.5, abs=1e-5)   # a pit 1 m below the filled DEM
    assert depth[12] == pytest.approx(0.2, abs=1e-5)  # from pixel 8 (DTF 0.8), not pixel 7 (DTF 1.5)
    assert np.isnan(depth[3])                         # the water stops 0.3 m short of the rim, so its 0.5 m fill stays dry
    assert np.isnan(depth[4])                         # DTF above the water
    assert np.isfinite(depth).sum() == 5


def test_fldpln_raised_bathymetry_takes_no_depth_away():
    """A negative fill, where bathymetry raises the DEM above the filled DEM, counts as 0: the water was routed across
    the filled surface, so it covers the cell at its depth above that surface, even where the raise (2 m) is deeper
    than the water (0.6 m), which used to leave a dry hole."""
    library = pl.LazyFrame({"FSP": [5, 6, 7], "FPP": [0, 1, 2], "DTF": [0.1, 0.4, 0.1], "fill": [-0.5, -2.0, 0.5]},
                           schema={"FSP": pl.Int32, "FPP": pl.Int32, "DTF": pl.Float32, "fill": pl.Float32})
    below = np.full((3, 5), np.nan, np.float32)
    depth = _fldpln_map(library, fill_below_dem=below)
    assert depth[0, [0, 1, 2]] == pytest.approx([0.9, 0.6, 1.4], abs=1e-5)
    # How far the filled DEM sits below the DEM, for the water surface: at every cell of the library
    assert below[0, :3].tolist() == pytest.approx([-0.5, -2.0, 0.0]) and below[1].tolist() == [0.0] * 5
    assert np.isnan(below[2]).all()


def test_fldpln_maps_every_library_row():
    """The mapper doesn't limit the drop below the stream: that is left to the library's build."""
    library = pl.LazyFrame({"FSP": [9], "FPP": [4], "DTF": [0.1], "fill": [0.0]},
                           schema={"FSP": pl.Int32, "FPP": pl.Int32, "DTF": pl.Float32, "fill": pl.Float32})
    assert _fldpln_map(library)[0, 4] == pytest.approx(0.9, abs=1e-5)  # 10 m below its stream pixel


def _segment_library(fsp, fpp, dtf, segment):
    return pl.LazyFrame({"FSP": fsp, "FPP": fpp, "DTF": dtf, "fill": [0.0] * len(fsp), "segment": segment},
                        schema={"FSP": pl.Int32, "FPP": pl.Int32, "DTF": pl.Float32, "fill": pl.Float32,
                                "segment": pl.Int32})


def test_fldpln_floods_each_segment_at_one_depth():
    """Stages 1, 3 and 2 m on segment 0 (pixels 5-7): the segment floods at their median, 2 m, though pixel 5 is the
    source of three of its five cells. Every cell of the segment takes that depth: pixel 6's cell with a DTF of 1.5 m
    floods 0.5 m deep, not 1.5 m. Segment 1 (pixels 8-9) floods at its own depth."""
    library = _segment_library(fsp=[5, 5, 5, 6, 7, 8, 9], fpp=[0, 1, 2, 11, 12, 13, 14],
                               dtf=[0.1, 0.4, 0.9, 1.5, 0.2, 0.5, 0.5], segment=[0, 0, 0, 0, 0, 1, 1])
    depth = _fldpln_map(library, wse=[101.0, 103.0, 102.0, 102.5, 102.5]).ravel()
    assert depth[[0, 1, 2]] == pytest.approx([1.9, 1.6, 1.1], abs=1e-5)   # not pixel 5's own 1 m
    assert depth[11] == pytest.approx(0.5, abs=1e-5)
    assert depth[12] == pytest.approx(1.8, abs=1e-5)
    assert depth[[13, 14]] == pytest.approx([2.0, 2.0], abs=1e-5)


def test_fldpln_without_segments_floods_each_reach_at_one_depth():
    """A library built before reaches were split has no segment column: the whole reach is one segment."""
    library = pl.LazyFrame({"FSP": [5, 9], "FPP": [0, 4], "DTF": [0.5, 0.5], "fill": [0.0, 0.0]},
                           schema={"FSP": pl.Int32, "FPP": pl.Int32, "DTF": pl.Float32, "fill": pl.Float32})
    depth = _fldpln_map(library, wse=[101.0, 101.0, 101.0, 101.0, 103.0])  # the median of the five pixels, 1 m
    assert depth[0, 0] == pytest.approx(0.5, abs=1e-5)
    assert depth[0, 4] == pytest.approx(0.5, abs=1e-5)


def test_fldpln_interpolates_the_depth_across_a_gap():
    """Pixel 7, down a cliff (bed 100 -> 94 -> 90), has no VDT value, but the other pixels of its segment (6 and 8)
    do. Its stage is the 0.3 m on either side, not the 1.3 m a straight water surface from 100.3 to 90.3 would stand
    above its bed, so the segment floods 0.3 m deep."""
    library = _segment_library(fsp=[6, 7, 7, 8], fpp=[1, 2, 12, 3], dtf=[9.0, 0.1, 0.5, 9.0], segment=[0, 0, 0, 0])
    depth = _fldpln_map(library, wse=[100.3, 100.3, np.nan, 90.3, 90.3], elev=[100.0, 100.0, 94.0, 90.0, 90.0])
    assert depth[0, 2] == pytest.approx(0.2, abs=1e-5)
    assert np.isnan(depth[2, 2])


def test_fldpln_unmeasured_pixels_take_their_own_segments_stages():
    """Pixels 6-8 have no VDT value (stages 1.75, 2.5 and 3.25 m along the stream, between 1 m and 4 m). In segment 0
    (pixels 5-7), past the segment's last VDT value they hold that pixel's 1 m, not stages leaning toward the next
    segment's 4 m, which would make the median 1.75 m: the cell with a DTF of 1.5 m stays dry. Segment 1 has only pixel
    8, without a VDT value, so it floods nothing."""
    library = _segment_library(fsp=[5, 6, 7, 7, 8, 9], fpp=[0, 1, 2, 12, 13, 14],
                               dtf=[0.5, 0.5, 0.5, 1.5, 0.5, 0.5], segment=[0, 0, 0, 0, 1, 2])
    depth = _fldpln_map(library, wse=[101.0, np.nan, np.nan, np.nan, 104.0]).ravel()
    assert depth[[0, 1, 2]] == pytest.approx([0.5] * 3, abs=1e-5)
    assert np.isnan(depth[12])
    assert np.isnan(depth[13])
    assert depth[14] == pytest.approx(3.5, abs=1e-5)


def test_fldpln_unmeasured_pixels_count_in_the_median():
    """Segment 0 is the whole stream. Pixels 6 and 7 have no VDT value; their stages, 0.47 and 0.73 m, are
    interpolated between pixel 5's 0.2 m and pixel 8's 1.0 m. The median of the five pixels is 0.73 m, not the 1.0 m of
    the three with VDT values, so the cell with a DTF of 0.8 m stays dry."""
    library = _segment_library(fsp=[5, 6, 7, 8, 9, 9], fpp=[0, 1, 2, 3, 4, 11],
                               dtf=[0.1, 0.1, 0.1, 0.1, 0.1, 0.8], segment=[0] * 6)
    depth = _fldpln_map(library, wse=[100.2, np.nan, np.nan, 101.0, 101.0]).ravel()
    assert depth[[0, 1, 2, 3, 4]] == pytest.approx([2.2 / 3 - 0.1] * 5, abs=1e-5)
    assert np.isnan(depth[11])


def test_fldpln_a_segment_without_vdt_values_floods_nothing():
    """Reach 1 is pixels 5-7, with a VDT value only at pixel 5 (1 m); reach 2, pixels 8-9, is 3 m deep. Segment 1,
    reach 1's tail, has no VDT value, so it floods nothing: not at reach 1's 1 m, nor at stages rising toward reach 2's
    3 m, though its cells' DTFs (0.5 and 1.5 m) are below both."""
    library = _segment_library(fsp=[5, 6, 7, 8, 9], fpp=[0, 1, 2, 3, 4], dtf=[0.5, 0.5, 1.5, 0.5, 0.5],
                               segment=[0, 1, 1, 2, 2])
    depth = _fldpln_map(library, wse=[101.0, np.nan, np.nan, 103.0, 103.0], reaches=[1, 1, 1, 2, 2]).ravel()
    assert depth[[0, 3, 4]] == pytest.approx([0.5, 2.5, 2.5], abs=1e-5)
    assert np.isnan(depth[[1, 2]]).all()


def test_fldpln_stage_is_the_water_over_elev_plus_the_pixel_s_own_fill():
    """The VDT's Elev is the DEM under the stream; the stream pixels sit in a filled pit 1 m deep (their own cells'
    fill), so the filled DEM there is 100 m and the stage 1 m, not the 2 m over Elev: the cell with a DTF of 0.5 m
    floods 0.5 m deep, and each stream pixel's own cell 1 - 0.1 m over its filled surface plus the 1 m pit."""
    library = _segment_library(fsp=[5, 6, 7, 7, 8, 9], fpp=[5, 6, 7, 2, 8, 9], dtf=[0.1, 0.1, 0.1, 0.5, 0.1, 0.1],
                               segment=[0] * 6).with_columns(fill=pl.Series([1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                                                                            dtype=pl.Float32))
    depth = _fldpln_map(library, elev=[99.0] * 5).ravel()
    assert depth[2] == pytest.approx(0.5, abs=1e-5)
    assert depth[5:10] == pytest.approx([1.9] * 5, abs=1e-5)


def test_fldpln_a_stream_pixel_without_its_own_row_has_no_vdt_value():
    """Pixel 7 has no row for its own cell, so the library gives no fill to measure its 109 m WSE from: it counts as
    having no VDT value, and segment 1, its alone, floods nothing, at 9 m or any other depth."""
    library = _segment_library(fsp=[5, 6, 7, 8, 9], fpp=[5, 6, 12, 8, 9], dtf=[0.1, 0.1, 0.5, 0.1, 0.1],
                               segment=[0, 0, 1, 2, 2])
    depth = _fldpln_map(library, wse=[101.0, 101.0, 109.0, 101.0, 101.0], own_cells=False).ravel()
    assert np.isnan(depth[12])
    assert depth[[5, 6, 8, 9]] == pytest.approx([0.9] * 4, abs=1e-5)


def test_interpolate_within_segments():
    """Linear between a segment's measured pixels, by position; held beyond its first and last; a segment without
    one keeps its stages; and nothing reaches across into the next segment."""
    segment = np.array([1, 1, 1, 1, 1, 2, 2, 3, 3])
    position = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9])
    measured = np.array([False, True, False, True, False, True, False, False, False])
    stage = np.array([9.0, 1.0, 9.0, 4.0, 9.0, 7.0, 9.0, 5.0, 6.0])
    out = fldpln_mapping._interpolate_within_segments(segment, position, measured, stage)
    assert out.tolist() == pytest.approx([1.0, 1.0, 2.0, 4.0, 4.0, 7.0, 7.0, 5.0, 6.0])


def test_median_by_group():
    group = np.array([2, 1, 1, 2, 1, 2, 2])
    value = np.array([5.0, 3.0, 1.0, 4.0, 2.0, 7.0, 6.0])
    groups, medians = fldpln_mapping._median_by_group(group, value)
    assert groups.tolist() == [1, 2]
    assert medians.tolist() == [2.0, 5.0]  # group 1: the middle of three; group 2: the lower middle of four


def test_fldpln_needs_the_fill_column():
    library = pl.LazyFrame({"FSP": [5], "FPP": [0], "DTF": [0.1]})
    with pytest.raises(ValueError, match="no fill column"):
        _fldpln_map(library)


@pytest.mark.parametrize("segments", [True, False])
def test_fldpln_maps_a_library_file_as_its_lazy_frame(tmp_path, monkeypatch, segments):
    """A Parquet library is read by its path, a part at a time, with or without a segment column."""
    monkeypatch.setattr(library_writer, "_LIBRARY_SLICE_ROWS", 3)
    monkeypatch.setattr(fldpln_mapping, "_READ_ROWS", 2)
    library = _segment_library(fsp=[5, 5, 5, 6, 7, 8, 9], fpp=[0, 1, 2, 11, 12, 13, 14],
                               dtf=[0.1, 0.4, 0.9, 1.5, 0.2, 0.5, 0.5], segment=[0, 0, 0, 0, 0, 1, 1])
    if not segments:
        library = library.drop("segment")
    library = _own_cells(library)
    path = tmp_path / "library.parquet"
    fldpln_library.save_library(library.collect().to_pandas(), path)
    wse = [101.0, 103.0, 102.0, 102.5, 102.5]
    from_file = _fldpln_map(str(path), wse=wse)
    assert np.isfinite(from_file).sum() >= 5
    assert np.array_equal(from_file, _fldpln_map(library, wse=wse), equal_nan=True)


def _random_library(n=400, cells=30, nan_fill=False):
    """A library of ``n`` rows, sorted by segment and source, over ``cells`` cells, each with one fill, DTF and fill
    in hundredths of a metre: ``(columns, fill of each cell)``.  Segment -1 has no depth, and segment 5 is outside
    the lookup of ``_random_dof``."""
    rng = np.random.default_rng(3)
    segment = np.sort(rng.integers(-1, 6, n)).astype(np.int32)
    fsp = (segment * 7 + rng.integers(0, 3, n)).astype(np.int32) % cells
    fpp = rng.integers(0, cells, n).astype(np.int32)
    fpp[::9] = fsp[::9]  # some sources' own cells
    cell_fill = rng.integers(-100, 200, cells) / 100.0
    if nan_fill:
        cell_fill[7] = np.nan
    columns = {"FSP": fsp, "FPP": fpp, "DTF": rng.integers(0, 300, n) / 100.0, "segment": segment,
               "fill": cell_fill[fpp]}
    return columns, cell_fill.astype(np.float32)


def _random_dof():
    return np.array([1.0, np.nan, 2.5, 0.3, 4.0], np.float32)


def _libraries(tmp_path, columns):
    """``columns`` as a LazyFrame of float32 metres, and saved as a Parquet library of decimals."""
    lazy = pl.LazyFrame({c: v.astype(np.float32) if c in ("DTF", "fill") else v for c, v in columns.items()})
    if np.isnan(columns["fill"]).any():
        return [lazy]
    path = tmp_path / "library.parquet"
    fldpln_library.save_library(pd.DataFrame(columns), path)
    return [lazy, str(path)]


@pytest.mark.parametrize("nan_fill", [False, True])
def test_deepest_heads_and_fill_are_numpy_s_maximum_at(tmp_path, monkeypatch, nan_fill):
    """Each cell's depth is the deepest of its rows' head plus its fill above 0, as every row's depth, the deepest
    kept by np.maximum.at, gives it: keys without a depth, or outside the lookup, are skipped, a NaN fill makes its
    cell NaN, and a library of decimals read a part at a time gives what its float32 metres do."""
    monkeypatch.setattr(fldpln_mapping, "_READ_ROWS", 64)
    monkeypatch.setattr(library_writer, "_LIBRARY_SLICE_ROWS", 64)
    columns, cell_fill = _random_library(nan_fill=nan_fill)
    dof = _random_dof()
    key, fpp = columns["segment"], columns["FPP"]
    dtf = columns["DTF"].astype(np.float32)
    known = (key >= 0) & (key < dof.size)
    head = np.where(known, dof[np.clip(key, 0, dof.size - 1)], np.float32(np.nan)) - dtf
    reaches = known & (head > 0)
    depth = head[reaches] + np.maximum(cell_fill[fpp[reaches]], np.float32(0.0))
    expected = np.full(cell_fill.size, -np.inf, np.float32)
    with np.errstate(invalid="ignore"):  # The NaN fill's depth
        np.maximum.at(expected, fpp[reaches], depth)
    assert np.isfinite(expected).sum() > 10 and np.isnan(expected).sum() == nan_fill
    for library in _libraries(tmp_path, columns):
        sources = fldpln_mapping.library_sources(library, cell_fill.size)
        best = np.full(cell_fill.size, -np.inf, np.float32)
        for part in fldpln_mapping._library_parts(library, ["segment", "FPP", "DTF"]):
            dtf_part = part["DTF"]
            fldpln_mapping._deepest_heads(best, part["segment"], part["FPP"], dtf_part,
                                          fldpln_mapping._scale(dtf_part), dof)
        fldpln_mapping._add_fill(best, sources)
        assert np.array_equal(best, expected, equal_nan=True)


def test_library_sources_are_each_source_once_with_its_own_fill(tmp_path, monkeypatch):
    """Each (segment, FSP) once, however the parts split their rows; the sources with a row for their own cell, and
    that cell's fill; the cells whose fill is above 0; and min(fill, 0) at every cell of the library, NaN off it."""
    monkeypatch.setattr(fldpln_mapping, "_READ_ROWS", 5)
    monkeypatch.setattr(library_writer, "_LIBRARY_SLICE_ROWS", 5)
    columns, cell_fill = _random_library(n=120, cells=40)
    fsp, fpp, segment = columns["FSP"], columns["FPP"], columns["segment"]
    in_library = np.zeros(cell_fill.size, bool)
    in_library[fpp] = True
    for library in _libraries(tmp_path, columns):
        below = np.full(cell_fill.size, np.nan, np.float32)
        sources = fldpln_mapping.library_sources(library, cell_fill.size, fill_below_dem=below)
        assert sources.key == "segment"
        pairs = sorted(set(zip(segment.tolist(), fsp.tolist())))
        assert sorted(sources.pixels.select("segment", "FSP").iter_rows()) == pairs
        own = np.unique(fsp[fsp == fpp])
        assert np.array_equal(sources.own_fsp, own) and np.array_equal(sources.own_fill, cell_fill[own])
        raised = np.flatnonzero(in_library & (cell_fill > 0))
        assert np.array_equal(sources.fill_cells, raised) and np.array_equal(sources.fill_values, cell_fill[raised])
        expected_below = np.where(in_library, np.minimum(cell_fill, np.float32(0.0)), np.float32(np.nan))
        assert np.array_equal(below, expected_below, equal_nan=True)


# --------------------------------------------------------------------------
# Multi-point mapper
# --------------------------------------------------------------------------

def test_filter_outliers():
    group = pd.DataFrame({"TopWidth": [10.0] * 10 + [100.0], "WSE": 5.0, "Velocity": 1.0})
    assert multi_point.filter_outliers(group)["TopWidth"].tolist() == [10.0] * 10
    one_row = group.iloc[:1]
    assert len(multi_point.filter_outliers(one_row)) == 1


def test_filter_outliers_is_each_reach_filtered_on_its_own():
    rng = np.random.default_rng(3)
    table = pd.DataFrame({"COMID": rng.integers(0, 12, 400), "TopWidth": rng.normal(50, 10, 400),
                          "WSE": rng.normal(100, 1, 400), "Velocity": rng.normal(1, 0.2, 400)})
    table.loc[rng.random(400) < 0.05, "WSE"] = np.nan
    table = pd.concat([table, pd.DataFrame({"COMID": [99], "TopWidth": [5.0], "WSE": [1.0], "Velocity": [0.1]})])
    expected = pd.concat([multi_point.filter_outliers(group.drop(columns="COMID")).assign(COMID=comid)
                          for comid, group in table.groupby("COMID")])
    assert sorted(multi_point.filter_outliers(table).index) == sorted(expected.index)


def test_clean_and_interp_one_row():
    q = np.array([30, 10, 20, -1.0])  # unsorted, with an invalid entry
    v = np.array([3, 1, 2, 9.0])
    t = np.array([30, 10, 20, 9.0])
    wse = np.array([103, 101, 102, 9.0])
    rows = [np.tile(a, (2, 1)) for a in (q, v, t, wse)]
    out = multi_point._clean_and_interp_rows(np.array([15.0, 50.0]), *rows, multi_point.NO_GROUND)
    assert [a[0] for a in out] == pytest.approx([1.5, 15.0, 101.5])
    assert [a[1] for a in out] == pytest.approx([3.0, 30.0, 103.0])  # held


def test_get_curve_columns_sorts_by_number():
    columns = ["q_10", "q_2", "v_10", "v_2", "t_10", "t_2", "wse_10", "wse_2", "COMID"]
    assert multi_point.get_curve_columns(columns)[0] == ["q_2", "q_10"]


@pytest.mark.parametrize("method, kept_of_all", [("mad", [0, 1, 2, 3, 5, 6]), ("iqr", list(range(8)))])
def test_hwm_outliers_are_dropped_within_each_reach(method, kept_of_all):
    depth = [1.0, 1.1, 0.9, 1.0, 9.0]  # the last is an outlier in its reach
    table = pd.DataFrame({"COMID": [1] * 5 + [2] * 3, "Elev": 100.0,
                          "WaterSurfaceElev_m": 100.0 + np.array(depth + [1.0, 1.0, 9.0])})
    kept = multi_point.filter_hwm_outliers_single_scenario(table, method=method, threshold=3.5, min_samples=5)
    assert kept.index.tolist() == [0, 1, 2, 3, 5, 6, 7]  # reach 2 has too few points to test
    # Over all points at once, both 9 m depths stand out by their MAD, but a quarter of the points widens the IQR
    kept = multi_point.filter_hwm_outliers_single_scenario(table, method=method, threshold=3.5, group_by="global")
    assert kept.index.tolist() == kept_of_all


def _gdal_buffer_mask(x, y, radius, shape, dx, dy):
    """The cells GDAL burns for shapely's buffers of the points, as the mapper burned them before."""
    driver = gdal.GetDriverByName("MEM")
    raster = driver.Create("", shape[1], shape[0], 1, gdal.GDT_Byte)
    raster.SetGeoTransform((0.0, dx, 0.0, 0.0, 0.0, -dy))
    vectors = (ogr.GetDriverByName("MEM") or ogr.GetDriverByName("Memory")).CreateDataSource("")
    layer = vectors.CreateLayer("buffers")
    for px, py, r in zip(x, y, radius):
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(ogr.CreateGeometryFromWkb(Point(float(px), float(py)).buffer(float(r), 16).wkb))
        layer.CreateFeature(feature)
    gdal.RasterizeLayer(raster, [1], layer, burn_values=[1])
    return raster.ReadAsArray().astype(bool)


def test_buffers_are_burned_as_gdal_burns_shapely_s_buffers():
    rng = np.random.default_rng(5)
    for dx, dy in [(10.0, 10.0), (27.8, 30.9)]:
        x = rng.uniform(-50, 40 * dx, 30).astype(np.float32)
        y = -rng.uniform(-50, 30 * dy, 30).astype(np.float32)
        width = rng.uniform(0, 12 * dx, 30).astype(np.float32)
        corridor, anchor = multi_point.build_variable_buffer_masks_from_points(
            x, y, width, (30, 40), (0.0, dx, 0.0, 0.0, 0.0, -dy), 50.0, 5.0, corridor_topwidth_factor=1.0,
            anchor_topwidth_factor=0.3)
        assert np.array_equal(corridor, _gdal_buffer_mask(x, y, width.astype(np.float64), (30, 40), dx, dy))
        radius = (width * 0.3).astype(np.float64)
        assert np.array_equal(anchor, _gdal_buffer_mask(x, y, radius, (30, 40), dx, dy))


def _smoothed_by_scipy(arr, sigma):
    """smooth_nan as it was written, with SciPy's gaussian_filter."""
    valid = np.isfinite(arr)
    num = ndi.gaussian_filter(np.where(valid, arr, 0.0).astype(np.float32), sigma)
    den = ndi.gaussian_filter(valid.astype(np.float32), sigma)
    out = np.full_like(arr, np.nan, dtype=np.float32)
    out[den > 1.0e-6] = num[den > 1.0e-6] / den[den > 1.0e-6]
    return out


@pytest.mark.parametrize("sigma", [0.1, 0.25, 0.6, 1.3])
def test_smoothing_is_scipy_s_gaussian_filter_to_the_last_bit(sigma):
    rng = np.random.default_rng(2)
    for shape in [(30, 41), (5, 3), (1, 9)]:
        wse = (100 + 20 * rng.random(shape)).astype(np.float32)
        wse[rng.random(shape) < 0.4] = np.nan
        assert np.array_equal(multi_point.smooth_nan(wse, sigma), _smoothed_by_scipy(wse, sigma), equal_nan=True)


@pytest.mark.parametrize("connectivity", [4, 8])
def test_propagating_from_the_anchors_is_scipy_s_binary_propagation(connectivity):
    rng = np.random.default_rng(connectivity)
    structure = ndi.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    for density in (0.3, 0.55, 0.8):
        cells = rng.random((60, 50)) < density
        anchor = cells & (rng.random(cells.shape) < 0.01)
        expected = ndi.binary_propagation(anchor, mask=cells, structure=structure)
        assert np.array_equal(patches_holding(cells, anchor, connectivity == 8), expected)


def test_flood_from_wse_must_connect_to_the_anchor():
    wse = np.full((1, 5), 100.5, np.float32)
    dem = np.array([[100, 100, 101, 100, 100]], np.float32)
    anchor = np.array([[True, False, False, False, False]])
    flood, water = multi_point.compute_fhs_flood_from_wse(wse, dem, np.ones((1, 5), bool), anchor, 4, 0)
    assert flood.tolist() == [[1, 1, 0, 0, 0]]  # the higher ground cuts off the last two cells
    assert np.isnan(water[0, 2])


def test_flood_map_from_one_point():
    dem = np.full((5, 5), 100.0, np.float32)
    kwargs = {"point_rows": [2], "point_cols": [2], "point_topwidth": [25.0], "point_ids": [1], "dx": 10.0,
              "corridor_topwidth_factor": 1.0, "anchor_topwidth_factor": 0.1, "maxdist_topwidth_factor": 1.0,
              "smooth_sigma_pixels": 0}
    result = multi_point.create_fhs_flood_map_from_points(dem, point_wse=[100.5], **kwargs)
    # Every cell within the 25 m corridor around the point floods: all but the four corners
    assert result["has_valid_points"] and result["flood"].sum() == 21 and result["flood"][0, 0] == 0
    assert np.nanmax(np.abs(result["depth"] - 0.5)) < 1e-4

    # A point below its own ground is dropped
    result = multi_point.create_fhs_flood_map_from_points(dem, point_wse=[99.0], **kwargs)
    assert not result["has_valid_points"] and result["flood"].sum() == 0


def test_a_flood_map_interpolated_a_few_cells_at_a_time_is_the_whole_s(monkeypatch):
    """The corridor's cells interpolated seven at a time give what they give all at once."""
    rng = np.random.default_rng(4)
    dem = (100.0 + rng.random((30, 40))).astype(np.float32)
    kwargs = {"point_rows": rng.integers(0, 30, 25), "point_cols": rng.integers(0, 40, 25),
              "point_topwidth": rng.uniform(20.0, 90.0, 25), "point_ids": np.arange(25) % 3, "dx": 10.0,
              "point_wse": 101.5 + rng.random(25), "point_slope": rng.uniform(1e-4, 1e-2, 25)}
    whole = multi_point.create_fhs_flood_map_from_points(dem, **kwargs)
    monkeypatch.setattr(multi_point, "_IDW_TARGETS", 7)
    parts = multi_point.create_fhs_flood_map_from_points(dem, **kwargs)
    assert whole["flood"].sum() > 100
    for name in ("flood", "wse", "depth", "slope"):
        assert np.array_equal(parts[name], whole[name], equal_nan=True)
