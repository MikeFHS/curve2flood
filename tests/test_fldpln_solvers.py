"""Tests for the FLDPLN library solvers.

The important ones pin the iterative solver bit-for-bit against
``fldpln_model_v5ram.py``, the Python port of the MATLAB model the spreader
descends from: :func:`test_iterative_matches_reference_model` one segment at a
time, and :func:`test_the_library_holds_the_reference_model_s_rows` through a
whole library build.  The port runs as it is, on the inputs converted to its
conventions (ESRI D8 codes for this package's WhiteboxTools ones, and its own
stream table layout).  Where MATLAB is installed,
:func:`test_reference_model_matches_the_matlab_original` pins the port to the
MATLAB model itself (``fldpln_matlab/``).
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from curve2flood.spreaders import fldpln as F
from curve2flood.spreaders.fldpln import library as L
from curve2flood.spreaders.fldpln import library_writer as LW
from curve2flood.spreaders.fldpln import mapping as M
from curve2flood.spreaders.fldpln import solvers as S
from curve2flood.spreaders.fldpln import workspace as W
from curve2flood.spreaders.fldpln.solvers import solver_exact as SE
from curve2flood.spreaders.fldpln.solvers import solver_iterative as SI

DATA_DIR = Path(__file__).parent / "data" / "scottsbluff"
REFERENCE_MODEL = Path(__file__).parent / "fldpln_model_v5ram.py"
MATLAB_MODEL = Path(__file__).parent / "fldpln_matlab"

BG = -9999.0
# A few segments that don't take long in the pure-Python reference.  Most run
# across filled flats: up to 3 m deep, only the first floods a cell above the
# smallest DTF, and none floods differently spilling once a step.
SEGMENTS = [760363429, 760418328, 760308499, 760395048, 760388378, 760536470]
# Three whose floodplains climb, to 48-106 distinct DTFs up to 3 m deep; the
# first two flood two to four times as many cells spilling to a steady state as
# spilling once a step.
CLIMBING_SEGMENTS = [760448294, 760494878, 760541470]
# One whose spill runs onto the raster's edge, where the flow directions are 0.
# The MATLAB model stops with an error there (its spill paths expect an edge cell
# to point off the raster), so only the port and the solver are compared on it.
EDGE_SEGMENTS = [760082219]
MATLAB_SEGMENTS = SEGMENTS + CLIMBING_SEGMENTS
REFERENCE_SEGMENTS = MATLAB_SEGMENTS + EDGE_SEGMENTS


def _opposite(pos: int) -> int:
    dr, dc = S.NEIGHBOR_DELTAS[pos]
    return S.NEIGHBOR_DELTAS.index((-dr, -dc))


@pytest.fixture(scope="module")
def rasters():
    if not DATA_DIR.exists():
        pytest.skip("The Scottsbluff rasters are not available.")
    from osgeo import gdal

    gdal.UseExceptions()
    fil = gdal.Open(str(DATA_DIR / "fabdem_filled.tif")).ReadAsArray().astype(np.float32)
    fdr = gdal.Open(str(DATA_DIR / "fabdem_flowdir.tif")).ReadAsArray().astype(np.uint8)
    info = pd.read_parquet(DATA_DIR / "fabdem_stream_info.parquet")
    return fil, fdr, info


def _segment(rasters, stream_id, solver, dh, fldmx, iterative_spill=True, margins=None, max_cells=0):
    fil, fdr, info = rasters
    previous = L._WINDOW_MARGINS
    if margins is not None:
        L._WINDOW_MARGINS = margins
    try:
        table = L.fldpln_library_for_segment(
            fil, fdr, int(stream_id), info.values, dh, 0.1, fldmx,
            iterative_spill, 0.0, BG, 0.0, solver, max_cells)
    finally:
        L._WINDOW_MARGINS = previous
    return table.sort_values("FPP").reset_index(drop=True)


def _library(tmp_path, name, **kwargs):
    """A library of the Scottsbluff fixture, mapped on the filled DEM itself (so every fill is 0)."""
    path = tmp_path / f"{name}.parquet"
    F.build_fldpln_library(
        str(DATA_DIR / "fabdem_filled.tif"),
        str(DATA_DIR / "fabdem_filled.tif"), str(DATA_DIR / "fabdem_stream_info.parquet"),
        str(DATA_DIR / "fabdem_flowdir.tif"), str(path), pbar=False,
        **{"dh": 0.5, "fldmn": 0.1, "iterative_spill": False, **kwargs})
    library = pd.read_parquet(path)
    library["DTF"] = library["DTF"].astype(float)
    return library


# --------------------------------------------------------------------------
# D8 conventions
# --------------------------------------------------------------------------

def test_inflow_is_outflow_reversed():
    """A neighbour drains into the centre exactly when it points back at it."""
    for pos in range(8):
        assert S.INFLOW[pos] == S.OUTFLOW[_opposite(pos)]


def test_outflow_matches_next_downstream():
    """OUTFLOW[pos] must route a cell to NEIGHBOR_DELTAS[pos], which is what the
    solvers' flat offsets (``dr * ncols + dc``) assume."""
    nrows = ncols = 5
    centre = 2 * ncols + 2
    for pos, (dr, dc) in enumerate(S.NEIGHBOR_DELTAS):
        fdr = np.zeros(nrows * ncols, dtype=np.uint8)
        fdr[centre] = S.OUTFLOW[pos]
        assert S._next_downstream(centre, fdr, nrows, ncols) == (2 + dr) * ncols + (2 + dc)


def test_next_downstream_stops_at_a_sink_and_the_edge_and_rejects_other_codes():
    fdr = np.array([0, 128, 3], dtype=np.uint8)  # a sink, north off the raster, not a D8 code
    assert S._next_downstream(0, fdr, 1, 3) == -1
    assert S._next_downstream(1, fdr, 1, 3) == -1
    with pytest.raises(ValueError, match="D8"):
        S._next_downstream(2, fdr, 1, 3)


def _walk_with_a_set(end_pixel, flat_fdr, nrows, ncols):
    """TD(R) walked step by step, remembering every pixel."""
    excluded, seen, current = set(), {end_pixel}, end_pixel
    while True:
        nxt = S._next_downstream(current, flat_fdr, nrows, ncols)
        if nxt == -1 or nxt in seen:
            return excluded
        excluded.add(nxt)
        seen.add(nxt)
        current = nxt


@pytest.mark.parametrize("end_pixel", range(12))
def test_downstream_exclusion_walks_each_pixel_below_the_end_once(end_pixel):
    """To the raster's edge, a sink, back to the end pixel, or round a loop below it, which is walked once."""
    nrows, ncols = 3, 4
    fdr = np.array([[2, 2, 8, 0],      # east, east, south, a sink
                    [8, 2, 4, 128],    # south (a loop with the cell below), east, south-east, north to the sink
                    [128, 2, 2, 64]],  # north, east, east, north-west (a loop with the cell it came from)
                   dtype=np.uint8).ravel()
    info = np.array([[0, end_pixel, 1, 5]], dtype=np.int64)
    excluded = L._downstream_exclusion(5, info, fdr, nrows, ncols)
    assert len(excluded) == len(set(excluded.tolist()))
    assert set(excluded.tolist()) == _walk_with_a_set(end_pixel, fdr, nrows, ncols)


def test_walk_reaches_matches_walking_each_reach():
    fil, fdr, info, _ = _channel()
    nrows, ncols = fil.shape
    flat_fdr = fdr.ravel()
    lengths = np.array([10, 15, 40, 0])  # the last two run off the raster's edge, and walk nothing
    starts = np.array([info[0, 0], info[1, 0], info[1, 0], info[0, 0]])
    pixels, offsets = M._walk_reaches(starts, lengths, flat_fdr, nrows, ncols)
    for i in range(len(starts)):
        expected, pixel = [], starts[i]
        for _ in range(lengths[i]):
            expected.append(pixel)
            pixel = L._next_downstream(pixel, flat_fdr, nrows, ncols)
            if pixel == -1:
                break
        assert pixels[offsets[i]:offsets[i + 1]].tolist() == expected


# --------------------------------------------------------------------------
# Fidelity to the MATLAB model and its Python port
# --------------------------------------------------------------------------

# The reference runs' fldmn, fldmx and dh (m)
REFERENCE_DEPTHS = (0.1, 3.0, 0.5)


def _reference_seg_info(info):
    """The stream table in the reference's layout: a row per segment, its start and end pixels in columns 0 and 1,
    and its length in column 4."""
    seg_info = np.zeros((len(info), 5))
    seg_info[:, 0] = info["start_pixel"].values
    seg_info[:, 1] = info["end_pixel"].values
    seg_info[:, 4] = info["length"].values
    return seg_info


def _segment_row(info, stream_id):
    return int(np.flatnonzero(info["stream_id"].values == stream_id)[0])


def _esri_d8(whitebox):
    """WhiteboxTools D8 codes as the MATLAB model's (ESRI's): each direction's code halved, north-east's 1 to 128."""
    return np.where(whitebox == 1, 128, whitebox >> 1).astype(whitebox.dtype)


@pytest.fixture(scope="module")
def reference(rasters):
    """``reference(stream_id, ssflg)``: the Python port's table for a segment, each worked out once, with spillover
    repeated to a steady state (``ssflg``) or once a step.

    The port runs as it is, the code the MATLAB test checks, on the inputs converted to its conventions: ESRI D8 codes
    and its stream table, a row per segment.
    """
    if not REFERENCE_MODEL.exists():
        pytest.skip("fldpln_model_v5ram.py is not available.")
    spec = importlib.util.spec_from_file_location("_v5ram", REFERENCE_MODEL)
    port = importlib.util.module_from_spec(spec)
    sys.modules["_v5ram"] = port
    spec.loader.exec_module(port)

    fil, fdr, info = rasters
    seg_info, esri = _reference_seg_info(info), _esri_d8(fdr).astype(np.int32)
    fldmn, fldmx, dh = REFERENCE_DEPTHS
    tables = {}

    def table(stream_id, ssflg=True):
        key = (stream_id, bool(ssflg))
        if key not in tables:
            result = port.fldpln_model_v5ram(_segment_row(info, stream_id), {"seg": seg_info}, fil, esri, fldmn,
                                             fldmx, dh, 0.0, ssflg, BG, shape=fil.shape)
            tables[key] = np.asarray(result.fldpln)
        return tables[key]

    return table


@pytest.mark.parametrize("iterative_spill", [True, False])
@pytest.mark.parametrize("stream_id", REFERENCE_SEGMENTS)
def test_iterative_matches_reference_model(rasters, reference, stream_id, iterative_spill):
    """The iterative solver gives the port's floodplain pixels, flood source pixels and DTFs (the port's float64 DTFs
    as float32) bit for bit, spilling to a steady state or once a step. The port raises DTFs below fldmn to it, and
    the solver keeps them as found, so those are compared raised."""
    fldmn, fldmx, dh = REFERENCE_DEPTHS
    table = reference(stream_id, iterative_spill)
    expected = table[np.argsort(table[:, 1])]

    actual = _segment(rasters, stream_id, "iterative", dh, fldmx, iterative_spill)

    assert np.array_equal(actual["FPP"].values, expected[:, 1])
    assert np.array_equal(actual["FSP"].values, expected[:, 0])
    assert np.array_equal(np.maximum(actual["DTF"].values, np.float32(fldmn)), expected[:, 2].astype(np.float32))
    assert (actual["DTF"].values[actual["FPP"].values == actual["FSP"].values] == 0).all()  # a stream pixel's own


@pytest.mark.parametrize("iterative_spill", [True, False])
def test_the_library_holds_the_reference_model_s_rows(rasters, reference, tmp_path, iterative_spill):
    """A library built as Curve2Flood builds one (every segment solved on the whole raster in one workspace, and
    written to Parquet) holds the port's rows for every segment, with the DTFs to the centimetre it keeps (raised to
    fldmn, as the port raises them). The reaches are kept whole, as the port solves them."""
    fldmn, fldmx, dh = REFERENCE_DEPTHS
    library = _library(tmp_path, "whole", stream_ids=REFERENCE_SEGMENTS, fldmn=fldmn, fldmx=fldmx, dh=dh,
                       iterative_spill=iterative_spill, max_reach_length=None, max_profile_deviation=None)
    expected = np.concatenate([reference(stream_id, iterative_spill) for stream_id in REFERENCE_SEGMENTS])
    expected = expected[np.lexsort((expected[:, 1], expected[:, 0]))]
    actual = library.sort_values(["FSP", "FPP"])

    assert library["segment"].nunique() == len(REFERENCE_SEGMENTS)
    assert np.array_equal(actual["FSP"].values, expected[:, 0])
    assert np.array_equal(actual["FPP"].values, expected[:, 1])
    assert np.array_equal(np.maximum(actual["DTF"].values.astype(np.float32), np.float32(fldmn)),
                          np.round(expected[:, 2].astype(np.float32), 2))


@pytest.fixture(scope="module")
def matlab_tables(rasters, tmp_path_factory):
    """The MATLAB model's table for each of MATLAB_SEGMENTS, spilling to a steady state or once a step,
    ``{(stream_id, ssflg): table}``, from one MATLAB session.

    The inputs are as the University of Kansas's scripts give them: the filled DEM as a float32 vector (single in
    MATLAB) with a BIL header, ESRI D8 codes, and pixels and segments counted from 1.
    """
    matlab = shutil.which("matlab")
    if matlab is None:
        pytest.skip("MATLAB is not on the PATH.")
    from scipy.io import loadmat, savemat

    fil, fdr, info = rasters
    work = tmp_path_factory.mktemp("matlab")
    fil.astype("<f4").tofile(work / "fil.bil")
    (work / "fil.hdr").write_text(f"BYTEORDER I\nLAYOUT BIL\nNROWS {fil.shape[0]}\nNCOLS {fil.shape[1]}\nNBANDS 1\n"
                                  f"NBITS 32\nPIXELTYPE FLOAT\n")
    seg_info = _reference_seg_info(info)
    seg_info[:, :2] += 1
    rows = {stream_id: _segment_row(info, stream_id) + 1 for stream_id in MATLAB_SEGMENTS}
    fldmn, fldmx, dh = REFERENCE_DEPTHS
    savemat(work / "inputs.mat", {
        "fildat": fil.astype(np.float32).reshape(-1, 1), "fdrdat": _esri_d8(fdr).reshape(-1, 1),
        "seg_info": seg_info, "segs": np.array(list(rows.values()), dtype=float),
        "fldmn": fldmn, "fldmx": fldmx, "dh": dh, "ssflgs": np.array([1.0, 0.0]), "bg": BG})

    command = f"addpath('{MATLAB_MODEL.as_posix()}'); run_segments('{work.as_posix()}')"
    run = subprocess.run([matlab, "-batch", command], capture_output=True, text=True, timeout=600, check=False)
    if run.returncode != 0:
        pytest.fail(f"MATLAB failed:\n{run.stdout[-3000:]}\n{run.stderr[-3000:]}")
    return {(stream_id, ssflg): loadmat(work / f"matlab_{row}_ss{int(ssflg)}.mat")["fldpln"]
            for stream_id, row in rows.items() for ssflg in (True, False)}


@pytest.mark.parametrize("ssflg", [True, False])
@pytest.mark.parametrize("stream_id", MATLAB_SEGMENTS)
def test_reference_model_matches_the_matlab_original(reference, matlab_tables, stream_id, ssflg):
    """The Python port gives the MATLAB model's flood source pixels, floodplain pixels and DTFs, row for row and bit
    for bit, spilling to a steady state or once a step. MATLAB counts pixels from 1, and its fourth column is the DTF
    plus the fill depth, where the port's is the fill depth alone, so only the first three columns are compared."""
    python = reference(stream_id, ssflg)[:, :3]
    matlab = matlab_tables[stream_id, ssflg][:, :3] - [1, 1, 0]
    assert len(python) > 100
    assert np.array_equal(python[np.argsort(python[:, 1])], matlab[np.argsort(matlab[:, 1])])


# --------------------------------------------------------------------------
# Exact solver
# --------------------------------------------------------------------------

@pytest.mark.parametrize("stream_id", SEGMENTS)
def test_exact_is_the_shrinking_dh_limit(rasters, stream_id):
    """Shrinking dh must move the depth-stepped result monotonically toward the
    exact field, never past it: the exact solver is a lower bound on DTF and an
    upper bound on extent."""
    exact = _segment(rasters, stream_id, "exact", 0.1, 3.0).set_index("FPP")["DTF"]
    previous_error = None
    for dh in (1.0, 0.5, 0.25):
        stepped = _segment(rasters, stream_id, "iterative", dh, 3.0).set_index("FPP")["DTF"]
        assert stepped.index.isin(exact.index).all(), "stepped found cells the exact solver missed"
        shared = exact.loc[stepped.index]
        assert (stepped.values >= shared.values - 1e-4).all(), "stepped beat the exact minimum"
        error = float((stepped.values - shared.values).mean())
        if previous_error is not None:
            assert error <= previous_error + 1e-6, "error grew as dh shrank"
        previous_error = error


def test_exact_dtf_is_at_least_the_net_climb(rasters):
    """DTF is the summed positive rise along the cheapest path, so it is never
    less than the net elevation gain from its own flood source -- a path that
    dips and climbs back pays for the climb twice, which is what makes FLDPLN
    differ from a plain height-above-nearest-drainage."""
    fil, _, _ = rasters
    flat = fil.ravel()
    table = _segment(rasters, 760363429, "exact", 0.1, 5.0)
    rise = flat[table["FPP"].values] - flat[table["FSP"].values]
    assert (table["DTF"].values >= -1e-6).all()
    # Tolerance covers accumulated float32 elevation differencing plus the exact
    # solver's fixed-point step; measured worst case over this fixture is ~6 mm.
    assert (table["DTF"].values >= rise - 0.02).all()


# --------------------------------------------------------------------------
# Spill decay
# --------------------------------------------------------------------------

def _spill_valley(spill_decay):
    """The iterative solver's rows, as ``{(row, col): DTF}``, for a one-pixel stream at (1, 1), 10 m high, beside a
    flat valley at 9 m that runs east along row 2 from column 1 to a sink at column 58.  Side cells at 9.2 m drain
    into it at columns 10 and 40, and everything else is a 100 m wall.  One depth step of 0.5 m: the stream's water
    spills into the valley at its own depth, 0, from (2, 1) and from (2, 2)."""
    ncols = 60
    fil = np.full((5, ncols), 100.0, dtype=np.float32)
    fdr = np.zeros((5, ncols), dtype=np.uint8)
    fil[1, 1] = 10.0
    fil[2, 1:59] = 9.0
    fdr[2, 1:58] = 2       # east
    for col in (10, 40):
        fil[3, col] = 9.2
        fdr[3, col] = 128  # north
    info = np.array([[ncols + 1, ncols + 1, 1, 7]])
    table = L.fldpln_library_for_segment(fil, fdr, 7, info, 0.5, 0.1, 0.5, False, 0.0, BG, spill_decay)
    assert (table["FSP"] == ncols + 1).all()
    return {divmod(int(p), ncols): float(d) for p, d in zip(table["FPP"], table["DTF"])}


def test_without_a_decay_a_spill_floods_its_whole_path_at_its_own_depth():
    rows = _spill_valley(0.0)
    assert set(rows) == {(1, 1), (3, 10), (3, 40)} | {(2, col) for col in range(1, 59)}
    assert all(rows[2, col] == 0.0 for col in range(1, 59))  # as found, not raised to fldmn
    assert rows[3, 10] == pytest.approx(0.2) and rows[3, 40] == pytest.approx(0.2)


def test_a_decaying_spill_loses_head_down_its_path():
    """Each step down the valley costs 0.012 m of DTF and of the water left to backfill with, so the spill from
    (2, 2) stops 41 steps on, where its 0.5 m runs out, and backfills the side cell at column 10 (0.2 m up, from
    0.096 m) but not the one at column 40 (from 0.456 m)."""
    rows = _spill_valley(0.012)
    assert set(rows) == {(1, 1), (3, 10)} | {(2, col) for col in range(1, 44)}
    assert rows[2, 1] == 0.0
    for col in range(2, 44):
        assert rows[2, col] == pytest.approx((col - 2) * 0.012, abs=1e-6)
    assert rows[3, 10] == pytest.approx(0.096 + 0.2, abs=1e-6)


def _down_a_plane(solver, spill_decay):
    """A solver's rows, as ``{(row, col): DTF}``, for a plane that falls 0.1 m a row to the south, every cell flowing
    south, and a one-pixel stream at (2, 10) that drains nowhere (so no TD(R)); 0.5 m deep, in one depth step."""
    nrows, ncols = 24, 21
    fil = np.repeat(100.0 - 0.1 * np.arange(nrows)[:, None], ncols, axis=1).astype(np.float32)
    fdr = np.full((nrows, ncols), 8, dtype=np.uint8)  # south
    fdr[2, 10] = 0
    source = 2 * ncols + 10
    info = np.array([[source, source, 1, 7]])
    table = L.fldpln_library_for_segment(fil, fdr, 7, info, 0.5, 0.1, 0.5, False, 0.0, BG, spill_decay, solver)
    assert (table["FSP"] == source).all()
    return {divmod(int(p), ncols): float(d) for p, d in zip(table["FPP"], table["DTF"])}


def test_the_exact_solver_s_water_loses_head_on_every_step_down():
    """Each row down the plane costs 0.05 m of head, whichever way the water steps, not only down the flow
    directions: once, water stepped down diagonally for free, and the whole plane flooded at 0.  Steps across a row,
    and up the plane (0.1 m a row), lose none."""
    rows = _down_a_plane("exact", 0.05)
    assert set(rows) == {(row, col) for row in range(13) for col in range(21)}
    for (row, col), dtf in rows.items():
        expected = (row - 2) * 0.05 if row >= 2 else (2 - row) * 0.1
        assert dtf == pytest.approx(expected, abs=1e-6), (row, col)


def test_the_exact_solver_s_head_loss_is_the_depth_stepped_solver_s_down_a_spill():
    """The depth-stepped solver spills from the stream into the cells around it, at its own depth, and from each runs
    down the flow directions losing 0.05 m a pixel, so 10 rows on its water is gone.  The exact solver's water loses
    as much a row, and one step more: the spill's first step down, onto the cell it spills into, costs it nothing."""
    exact = _down_a_plane("exact", 0.05)
    stepped = _down_a_plane("iterative", 0.05)
    spilled = {(row, col) for row in range(3, 14) for col in (9, 10, 11)}
    assert spilled <= set(stepped)
    for row, col in spilled:
        assert stepped[row, col] == pytest.approx((row - 3) * 0.05, abs=1e-6)
    for cell in set(exact) & set(stepped):
        assert stepped[cell] - 1e-6 <= exact[cell] <= stepped[cell] + 0.05 + 1e-6, cell


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("solver,dh", [("exact", 0.1), ("iterative", 0.5)])
def test_window_size_does_not_change_the_result(rasters, solver, dh):
    fil, _, _ = rasters
    whole = (max(fil.shape),)
    for stream_id in SEGMENTS:
        windowed = _segment(rasters, stream_id, solver, dh, 7.2)
        unwindowed = _segment(rasters, stream_id, solver, dh, 7.2, margins=whole)
        assert np.array_equal(windowed["FPP"].values, unwindowed["FPP"].values)
        assert np.array_equal(windowed["DTF"].values, unwindowed["DTF"].values)
        assert np.array_equal(windowed["FSP"].values, unwindowed["FSP"].values)


def _workspace(fil, fdr, channel=None):
    halo_fil = L._crop(fil, 0, fil.shape[0], 0, fil.shape[1], np.float32(BG)).ravel()
    halo_fdr = L._crop(fdr, 0, fil.shape[0], 0, fil.shape[1], np.uint8(0)).ravel()
    return L.Workspace(halo_fil, halo_fdr, *SI._neighbour_bits(halo_fil, halo_fdr, fil.shape[1] + 2, BG),
                        L._channel_lookup(channel, fil.size))


@pytest.mark.parametrize("fldmx", [0.6, 7.2])
def test_a_workspace_solves_as_windows_do_and_is_left_as_found(rasters, fldmx):
    """One workspace solves every segment, on the whole raster, to the rows the windows give, and resets what it
    flooded; solving them all twice over the same arrays changes nothing."""
    fil, fdr, info = rasters
    workspace = _workspace(fil, fdr)
    fresh = S._iterative_arrays(workspace.fil.size)
    for _ in range(2):
        for stream_id in SEGMENTS:
            args = (fil, fdr, int(stream_id), info.values, 0.5, 0.1, fldmx, False, 0.0, BG, 0.022)
            windowed = L._segment_rows(*args)
            whole = L._segment_rows(*args, workspace=workspace)
            for a, b in zip(windowed[:3], whole[:3]):
                assert np.array_equal(a, b)
    for kept, start in zip(workspace.arrays[:5] + workspace.arrays[26:], fresh[:5] + fresh[26:]):
        assert np.array_equal(kept, start)
    assert not workspace.excluded.any()

@pytest.mark.parametrize("fldmx", [0.6, 7.2])
def test_a_workspace_started_again_on_new_arrays_solves_the_same(rasters, monkeypatch, fldmx):
    """New arrays after every segment, their stamps from 0 again, solve as one set kept throughout."""
    fil, fdr, info = rasters
    kept = _workspace(fil, fdr)
    monkeypatch.setattr(W, "_RENEW_EVERY", 1)
    renewed = _workspace(fil, fdr)
    for stream_id in SEGMENTS * 2:
        args = (fil, fdr, int(stream_id), info.values, 0.5, 0.1, fldmx, False, 0.0, BG, 0.022)
        monkeypatch.setattr(W, "_RENEW_EVERY", 10 ** 9)
        expected = L._segment_rows(*args, workspace=kept)
        monkeypatch.setattr(W, "_RENEW_EVERY", 1)
        for a, b in zip(L._segment_rows(*args, workspace=renewed)[:3], expected[:3]):
            assert np.array_equal(a, b)


@pytest.mark.parametrize("n", [5, 600, 5000])
def test_the_radix_order_is_the_two_merge_sorts(n):
    """Ascending depth, then descending elevation, then the keys' own order; -0.0 ties with 0.0."""
    rng = np.random.default_rng(n)
    m = n + 10
    depth = rng.choice(np.array([0.0, -0.0, 0.1, 0.5, 7.5, 3.4e38], np.float32), m)
    elevation = rng.choice(np.array([0.0, -0.0, -5.0, 12.5, 99.99, 100.0, -9999.0], np.float32), m)
    keys = rng.permutation(m)[:n].astype(np.int32)
    by_elevation = keys[np.argsort(-elevation[keys], kind="stable")]
    expected = by_elevation[np.argsort(depth[by_elevation], kind="stable")]
    assert np.array_equal(SI._order_by_depth_then_elevation(keys, depth, elevation), expected)


def test_tiny_window_still_grows_to_the_right_answer(rasters):
    """A window far too small must be detected and grown, not silently truncated."""
    fil, _, _ = rasters
    reference = _segment(rasters, 760363429, "exact", 0.1, 5.0, margins=(max(fil.shape),))
    grown = _segment(rasters, 760363429, "exact", 0.1, 5.0, margins=(1, 4, 16, max(fil.shape)))
    assert np.array_equal(grown["FPP"].values, reference["FPP"].values)
    assert np.array_equal(grown["DTF"].values, reference["DTF"].values)


# --------------------------------------------------------------------------
# Argument handling
# --------------------------------------------------------------------------

def test_unknown_solver_is_rejected(rasters):
    with pytest.raises(ValueError, match="solver must be one of"):
        _segment(rasters, SEGMENTS[0], "dijkstra", 0.5, 1.0)


def test_iterative_requires_positive_dh(rasters):
    with pytest.raises(ValueError, match="dh must be positive"):
        _segment(rasters, SEGMENTS[0], "iterative", 0.0, 1.0)


def test_exact_ignores_dh(rasters):
    """dh is not a parameter of the exact solver, so it must not be consulted."""
    a = _segment(rasters, SEGMENTS[0], "exact", 0.01, 3.0)
    b = _segment(rasters, SEGMENTS[0], "exact", 5.00, 3.0)
    assert np.array_equal(a["FPP"].values, b["FPP"].values)
    assert np.array_equal(a["DTF"].values, b["DTF"].values)


# --------------------------------------------------------------------------
# max_cells: a floodplain budget per segment
# --------------------------------------------------------------------------

@pytest.mark.parametrize("stream_id", SEGMENTS)
def test_max_cells_is_a_lower_max_depth(rasters, stream_id):
    """A segment cut by max_cells is the segment solved with fldmx just below
    the depth it stopped at, so the cap finds a max depth rather than dropping
    arbitrary cells."""
    full = _segment(rasters, stream_id, "exact", 0.1, 5.0)
    zero_depth = len(_segment(rasters, stream_id, "exact", 0.1, 0.0))
    if len(full) - zero_depth < 2:
        pytest.skip("the segment floods (almost) nothing above zero depth, so there is no depth to cut at")
    max_cells = zero_depth + (len(full) - zero_depth) // 2
    capped = _segment(rasters, stream_id, "exact", 0.1, 5.0, max_cells=max_cells)
    limit = capped.attrs["depth_limit"]
    assert limit is not None and len(capped) <= max_cells
    assert capped["DTF"].max() <= limit + 1e-6
    lowered = _segment(rasters, stream_id, "exact", 0.1, limit - 1 / SE._DTF_UNITS_PER_METRE)
    assert np.array_equal(capped["FPP"].values, lowered["FPP"].values)
    assert np.array_equal(capped["DTF"].values, lowered["DTF"].values)
    assert np.array_equal(capped["FSP"].values, lowered["FSP"].values)


def test_max_cells_keeps_the_cells_flooded_at_no_depth(rasters):
    """There is no depth to cut the zero-depth floodplain back to, so a budget
    smaller than it keeps exactly that."""
    zero_depth = _segment(rasters, SEGMENTS[0], "exact", 0.1, 0.0)
    capped = _segment(rasters, SEGMENTS[0], "exact", 0.1, 5.0, max_cells=1)
    assert capped.attrs["depth_limit"] > 0
    assert np.array_equal(capped["FPP"].values, zero_depth["FPP"].values)
    assert np.array_equal(capped["FSP"].values, zero_depth["FSP"].values)


def test_max_cells_above_the_floodplain_changes_nothing(rasters):
    full = _segment(rasters, SEGMENTS[0], "exact", 0.1, 5.0)
    capped = _segment(rasters, SEGMENTS[0], "exact", 0.1, 5.0, max_cells=len(full))
    assert capped.attrs["depth_limit"] is None
    assert np.array_equal(capped["FPP"].values, full["FPP"].values)
    assert np.array_equal(capped["DTF"].values, full["DTF"].values)


def test_max_cells_needs_the_exact_solver(rasters):
    with pytest.raises(ValueError, match="max_cells needs solver='exact'"):
        _segment(rasters, SEGMENTS[0], "iterative", 0.5, 1.0, max_cells=10)


def test_library_caps_each_segment(rasters, tmp_path):
    """The capped library is the full one with each segment cut at its own depth."""
    full = _library(tmp_path, "full", fldmx=5.0, stream_ids=SEGMENTS[:2], solver="exact")
    capped = _library(tmp_path, "capped", fldmx=5.0, stream_ids=SEGMENTS[:2], max_cells_per_segment=len(full) // 4,
                      solver="exact")
    assert 0 < len(capped) < len(full)
    both = capped.merge(full, on=["FSP", "FPP"], how="left", suffixes=("", "_full"))
    assert both["DTF_full"].notna().all() and (both["DTF"] == both["DTF_full"]).all()


def test_library_is_depth_stepped_by_default(rasters, tmp_path):
    """Without solver settings the library is the depth-stepped one: dh 0.5, fldmn 0.1, fldmx 25, one spill per
    step (FLDPLN's advice for coarse DEMs)."""
    path = tmp_path / "default.parquet"
    F.build_fldpln_library(str(DATA_DIR / "fabdem_filled.tif"), str(DATA_DIR / "fabdem_filled.tif"),
                           str(DATA_DIR / "fabdem_stream_info.parquet"), str(DATA_DIR / "fabdem_flowdir.tif"), str(path),
                           stream_ids=SEGMENTS[:1], pbar=False)
    default = pd.read_parquet(path)
    default["DTF"] = default["DTF"].astype(float)
    stepped = _library(tmp_path, "stepped", fldmx=25.0, stream_ids=SEGMENTS[:1], solver="iterative")
    exact = _library(tmp_path, "exact", fldmx=25.0, stream_ids=SEGMENTS[:1], solver="exact")
    assert len(default) and default.equals(stepped)
    assert not default.equals(exact)


@pytest.mark.parametrize("bad", [0, -5, 2.5])
def test_library_rejects_bad_max_cells(rasters, tmp_path, bad):
    with pytest.raises(ValueError, match="max_cells_per_segment must be"):
        _library(tmp_path, "bad", fldmx=1.0, stream_ids=SEGMENTS[:1], max_cells_per_segment=bad)


def test_library_max_cells_needs_the_exact_solver(rasters, tmp_path):
    with pytest.raises(ValueError, match="needs solver='exact'"):
        _library(tmp_path, "bad", fldmx=1.0, stream_ids=SEGMENTS[:1], max_cells_per_segment=10, solver="iterative")


# --------------------------------------------------------------------------
# The fill column: how far the filled DEM sits above the DEM
# --------------------------------------------------------------------------

def _slope(nrows=40, ncols=40, drop_per_row=0.5):
    """A plane falling south with a stream running east along row 5.  A downhill
    step costs no depth, so the stream reaches every cell south of it at DTF 0."""
    rows = np.arange(nrows, dtype=np.float32)[:, None]
    fil = np.repeat(100.0 - drop_per_row * rows, ncols, axis=1).astype(np.float32)
    fdr = np.full((nrows, ncols), 8, dtype=np.uint8)  # Whitebox D8: south
    fdr[5, 10:20] = 2                                   # east, along the stream
    info = np.array([[5 * ncols + 10, 5 * ncols + 20, 11, 1]], dtype=np.int64)
    return fil, fdr, info


def _slope_library(tmp_path, name, dem=None, filled=None, **kwargs):
    """The library of ``_slope`` (with ``filled`` for its filled DEM, if given), mapped on ``dem`` (the filled DEM
    when None)."""
    from osgeo import gdal
    fil, fdr, info = _slope()
    fil = fil if filled is None else filled
    rasters = {"filled": fil, "flowdir": fdr, "dem": fil if dem is None else dem}
    for raster, array in rasters.items():
        dtype = gdal.GDT_Byte if array.dtype == np.uint8 else gdal.GDT_Float32
        ds = gdal.GetDriverByName("GTiff").Create(str(tmp_path / f"{raster}.tif"), array.shape[1], array.shape[0], 1, dtype)
        ds.GetRasterBand(1).WriteArray(array)
        ds = None
    pd.DataFrame(info, columns=["start_pixel", "end_pixel", "length", "stream_id"]).to_parquet(tmp_path / "info.parquet")
    F.build_fldpln_library(str(tmp_path / "dem.tif"), str(tmp_path / "filled.tif"), str(tmp_path / "info.parquet"),
                           str(tmp_path / "flowdir.tif"), str(tmp_path / f"{name}.parquet"), dh=0.1, fldmn=0.1,
                           fldmx=1.0, iterative_spill=False, pbar=False, **kwargs)
    return pd.read_parquet(tmp_path / f"{name}.parquet")


def test_library_fill_is_the_filled_dem_above_the_dem(tmp_path):
    fil = _slope()[0]
    ncols = fil.shape[1]
    dem = fil - 0.25
    dem[8, 12] -= 2.75     # a pit 3 m below the filled DEM
    dem[9, 15] = -9999.0   # no data
    library = _slope_library(tmp_path, "fill", dem=dem)
    fpp, fill = library["FPP"].to_numpy(), library["fill"].astype(float).to_numpy()
    assert len(library) and (fpp != 9 * ncols + 15).all()  # a cell without a DEM value never floods
    pit = fpp == 8 * ncols + 12
    assert pit.any() and np.allclose(fill[pit], 3.0)
    assert np.allclose(fill[~pit], 0.25)


def test_library_needs_the_dem_on_the_same_grid(tmp_path):
    with pytest.raises(ValueError, match="The filled DEM is .* cells, but the DEM is 5 by 5"):
        _slope_library(tmp_path, "bad", dem=np.zeros((5, 5), np.float32))


def test_library_says_where_its_grid_is(tmp_path):
    """A library records its grid, so that a flood map needs no DEM to place it, and how it was built."""
    _slope_library(tmp_path, "described")
    metadata = F.library_metadata(tmp_path / "described.parquet")
    fil = _slope()[0]
    assert (metadata["nrows"], metadata["ncols"]) == fil.shape
    assert metadata["geotransform"] == [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # GDAL's for a raster without one
    assert metadata["cell_size_m"] == [1.0, 1.0] and metadata["version"] == 1
    assert metadata["build"]["fldmx"] == 1.0 and metadata["build"]["solver"] == "iterative"
    # A CSV library has nowhere to keep it
    assert F.library_metadata(tmp_path / "library.csv") is None


# --------------------------------------------------------------------------
# Background mask
# --------------------------------------------------------------------------

def test_library_masks_with_an_integer_bg_mask(rasters, tmp_path):
    """A 0/1 integer mask, like a lake raster read from disk, masks the same
    cells as a boolean one -- indexed as is, it would mask rows 0 and 1."""
    fil, _, _ = rasters
    floodplain = _segment(rasters, SEGMENTS[0], "exact", 0.1, 3.0)["FPP"].values
    mask = np.zeros(fil.shape, dtype=np.uint8)
    mask.ravel()[floodplain[::2]] = 1
    as_int = _library(tmp_path, "int", fldmx=3.0, stream_ids=SEGMENTS[:1], bg_mask=mask)
    as_bool = _library(tmp_path, "bool", fldmx=3.0, stream_ids=SEGMENTS[:1], bg_mask=mask.astype(bool))
    assert len(as_int) and not mask.ravel()[as_int["FPP"].values].any()
    assert as_int.equals(as_bool)


def test_masked_vdt_cells_do_not_set_the_max_depth(rasters, tmp_path):
    """A VDT cell under bg_mask (a lake) is never flooded from, so it must not
    count toward its segment's max depth: measured from bg it would be thousands
    of metres, and the segment would flood to fldmx."""
    fil, fdr, info = rasters
    stream_id = SEGMENTS[0]
    pixels = np.asarray(L._segment_pixels(stream_id, info.values, fdr.ravel(), *fil.shape))
    rows, cols = np.divmod(pixels, fil.shape[1])
    vdt_file = tmp_path / "vdt.parquet"
    pd.DataFrame({"COMID": stream_id, "Row": rows, "Col": cols, "wse_1": fil[rows, cols] + 1.0}).to_parquet(vdt_file)
    mask = np.zeros(fil.shape, dtype=bool)
    mask[rows[0], cols[0]] = True
    library = _library(tmp_path, "vdt", fldmx=5.0, vdt_file=str(vdt_file), bg_mask=mask, stream_ids=[stream_id])
    assert len(library) and library["DTF"].max() <= 1.0 + 0.01


# --------------------------------------------------------------------------
# Reaches longer than max_reach_length are cut into segments
# --------------------------------------------------------------------------

def test_step_lengths_follow_the_flow_direction():
    flat_fdr = np.array([2, 8, 4, 0], dtype=np.uint8)  # east, south, south-east, no outflow
    steps = L._step_lengths(np.arange(4), flat_fdr, 30.0, 40.0)
    assert steps.tolist() == pytest.approx([30.0, 40.0, 50.0, 35.0])


def test_split_reach_cuts_equal_lengths_and_keeps_the_reach_end():
    pixels = np.arange(100, 110)
    pieces = L._split_reach(np.array([100, 999, 10, 7]), pixels, np.full(10, 100.0), 300.0)
    assert [p[2] for p in pieces] == [3, 2, 3, 2]          # ceil(1000 / 300) = 4 pieces, 250 m each
    assert np.concatenate([p[3] for p in pieces]).tolist() == pixels.tolist()
    assert [p[0] for p in pieces] == [100, 103, 105, 108]
    assert [p[1] for p in pieces] == [102, 104, 107, 999]  # the last keeps the reach's own end pixel


@pytest.mark.parametrize("max_length", [None, 1000.0, 5000.0])
def test_split_reach_keeps_a_short_reach_whole(max_length):
    row = np.array([100, 999, 10, 7])
    pieces = L._split_reach(row, np.arange(100, 110), np.full(10, 100.0), max_length)
    assert len(pieces) == 1 and pieces[0][:3] == (100, 999, 10)


def test_library_splits_long_reaches_into_segments(tmp_path):
    """The slope's 11-pixel stream (1 m cells: its rasters have no coordinate system) cut at 4 m makes 3 segments.
    Every stream pixel belongs to one, and each cell's smallest DTF over the segments is its DTF in the whole reach."""
    _, _, _ = _slope()
    whole = _slope_library(tmp_path, "whole", max_reach_length=None)
    split = _slope_library(tmp_path, "split", max_reach_length=4.0)
    assert whole["segment"].unique().tolist() == [0]
    assert sorted(split["segment"].unique()) == [0, 1, 2]
    assert split.groupby("FSP")["segment"].nunique().max() == 1
    assert set(split["FSP"]) == set(whole["FSP"])
    best = split.assign(DTF=split["DTF"].astype(float)).groupby("FPP")["DTF"].min()
    reference = whole.assign(DTF=whole["DTF"].astype(float)).set_index("FPP")["DTF"]
    assert best.index.sort_values().equals(reference.index.sort_values())
    assert np.allclose(best.loc[reference.index], reference)


def test_library_rejects_a_bad_max_reach_length(tmp_path):
    with pytest.raises(ValueError, match="max_reach_length must be"):
        _slope_library(tmp_path, "bad", max_reach_length=0)


# --------------------------------------------------------------------------
# Reaches are cut where the filled DEM under them bends or steps
# --------------------------------------------------------------------------

def test_profile_cuts_at_a_step_wider_than_twice_the_deviation():
    """A 1 m step between two flats is 0.5 m off the chord over both, so it is a cut at 0.25 m but not at 0.5 m."""
    distance = np.arange(100) * 30.0
    elevation = np.r_[np.full(50, 100.0), np.full(50, 99.0)]
    assert L._profile_cuts(distance, elevation, 0.25).tolist() == [50]
    assert L._profile_cuts(distance, elevation, 0.5).tolist() == []


def test_profile_cuts_put_a_bend_with_the_gentler_piece():
    """A slope that levels into a filled flat, then drops 1 m: the pixel where the flat begins lies on both lines and
    goes with the flat, and the drop is a cut between its two pixels."""
    distance = np.arange(100) * 30.0
    elevation = np.r_[100 - 0.002 * distance[:40], np.full(40, 100 - 0.002 * distance[39]), 99 - 0.002 * distance[80:]]
    assert L._profile_cuts(distance, elevation, 0.25).tolist() == [39, 80]


def test_profile_cuts_keep_a_one_pixel_terrace_alone():
    elevation = np.r_[np.full(10, 10.0), 9.0, np.full(10, 8.0)]
    assert L._profile_cuts(np.arange(21) * 30.0, elevation, 0.25).tolist() == [10, 11]


@pytest.mark.parametrize("elevation", [np.linspace(110.0, 100.0, 50), np.full(50, 100.0), np.array([5.0, 1.0])])
def test_profile_cuts_leave_a_straight_profile_whole(elevation):
    assert L._profile_cuts(np.arange(len(elevation)) * 30.0, elevation, 0.01).tolist() == []


def test_split_reach_cuts_at_the_profile_then_by_length():
    """A step after pixel 3 cuts the 1 km reach into 400 and 600 m runs, and 300 m cuts each of them in two."""
    pixels = np.arange(100, 110)
    elevation = np.r_[np.full(4, 50.0), np.full(6, 48.0)]
    pieces = L._split_reach(np.array([100, 999, 10, 7]), pixels, np.full(10, 100.0), 300.0, elevation, 0.5)
    assert [p[3].tolist() for p in pieces] == [[100, 101], [102, 103], [104, 105, 106], [107, 108, 109]]
    assert [p[1] for p in pieces] == [101, 103, 106, 999]
    whole = L._split_reach(np.array([100, 999, 10, 7]), pixels, np.full(10, 100.0), None, elevation, 5.0)
    assert len(whole) == 1 and whole[0][:3] == (100, 999, 10)


def _stepped_slope():
    """``_slope`` with everything east of column 15 lowered 1 m: a step on the stream between its 5th and 6th pixel."""
    fil = _slope()[0].copy()
    fil[:, 15:] -= 1.0
    return fil


def test_library_cuts_reaches_at_a_step(tmp_path):
    library = _slope_library(tmp_path, "step", filled=_stepped_slope(), max_profile_deviation=0.25)
    segment_of = library.groupby("FSP")["segment"].agg(["min", "nunique"])
    assert (segment_of["nunique"] == 1).all()
    columns = segment_of.index.to_numpy() % 40
    assert sorted(set(segment_of["min"][columns < 15])) == [0] and sorted(set(segment_of["min"][columns >= 15])) == [1]


def test_library_profile_ignores_a_masked_lake(tmp_path):
    """A masked stream pixel is -9999 to the solver, but not to the profile: still the one cut at the step."""
    mask = np.zeros((40, 40), dtype=bool)
    mask[5, 12] = True
    library = _slope_library(tmp_path, "lake", filled=_stepped_slope(), bg_mask=mask, max_profile_deviation=0.25)
    assert library["segment"].nunique() == 2


def test_library_with_a_level_stream_is_not_cut(tmp_path):
    """The slope's stream runs level along row 5, so its profile is straight at any deviation."""
    whole = _slope_library(tmp_path, "whole", max_profile_deviation=None)
    assert _slope_library(tmp_path, "cut", max_profile_deviation=0.01).equals(whole)


def test_library_rejects_a_bad_max_profile_deviation(tmp_path):
    with pytest.raises(ValueError, match="max_profile_deviation must be"):
        _slope_library(tmp_path, "bad", max_profile_deviation=0)


# --------------------------------------------------------------------------
# The channel below a segment joins its TD(R), given a water mask
# --------------------------------------------------------------------------

def _channel(nrows=30, ncols=13):
    """A valley falling 0.5 m a row, with a channel three cells wide (columns 5-7, the water) whose middle column is
    the stream: reach 1 on rows 2-11 and reach 2 on rows 12-26.  The channel's sides drain into the stream, the
    banks (2 m up) into the channel."""
    rows, cols = np.mgrid[0:nrows, 0:ncols]
    side = np.abs(cols - 6)
    base = 100.0 - 0.5 * rows
    fil = np.where(side <= 1, base + 0.1 * side, base + 2.0 + 0.5 * (side - 2)).astype(np.float32)
    fdr = np.select([cols == 6, cols == 5, cols == 7, cols < 5], [8, 4, 16, 2], 32).astype(np.uint8)
    info = np.array([[2 * ncols + 6, 11 * ncols + 6, 10, 1], [12 * ncols + 6, 26 * ncols + 6, 15, 2]], dtype=np.int64)
    return fil, fdr, info, side <= 1


def _channel_library(tmp_path, name, **kwargs):
    from osgeo import gdal
    fil, fdr, info, _ = _channel()
    for raster, array, dtype in (("filled", fil, gdal.GDT_Float32), ("flowdir", fdr, gdal.GDT_Byte)):
        ds = gdal.GetDriverByName("GTiff").Create(str(tmp_path / f"{raster}.tif"), array.shape[1], array.shape[0], 1, dtype)
        ds.GetRasterBand(1).WriteArray(array)
        ds = None
    pd.DataFrame(info, columns=["start_pixel", "end_pixel", "length", "stream_id"]).to_parquet(tmp_path / "info.parquet")
    F.build_fldpln_library(str(tmp_path / "filled.tif"), str(tmp_path / "filled.tif"), str(tmp_path / "info.parquet"),
                           str(tmp_path / "flowdir.tif"), str(tmp_path / f"{name}.parquet"), dh=0.1, fldmn=0.1,
                           fldmx=1.0, iterative_spill=False, pbar=False, **kwargs)
    return pd.read_parquet(tmp_path / f"{name}.parquet")


def _walk_to_network(pixel, flat_fdr, on_network, nrows, ncols):
    """The first network pixel on ``pixel``'s D8 path, walked step by step, or -1."""
    seen = set()
    while pixel != -1 and not on_network[pixel] and pixel not in seen:
        seen.add(pixel)
        pixel = L._next_downstream(pixel, flat_fdr, nrows, ncols)
    return pixel if pixel != -1 and on_network[pixel] else -1


def test_network_pixels_are_each_reach_walked_down_the_flow_directions():
    fil, fdr, info, _ = _channel()
    flat_fdr = fdr.ravel()
    on_network = L._network_pixels(info, flat_fdr, *fil.shape)
    expected = np.concatenate([L._segment_pixels(int(r[3]), info, flat_fdr, *fil.shape) for r in info])
    assert sorted(np.flatnonzero(on_network)) == sorted(expected)


def test_each_cell_finds_the_first_network_pixel_on_its_path():
    """Paths are followed whatever order the starts come in; one ending at a sink, the raster's edge or in a loop
    meets nothing."""
    nrows, ncols = 3, 4
    fdr = np.array([[2, 2, 8, 0],      # east, east, south to the network pixel, a sink
                    [8, 2, 8, 32],     # south (a loop with the cell below), east, south, west to the network pixel
                    [128, 2, 32, 2]],  # north, east and west (a loop), east off the raster
                   dtype=np.uint8).ravel()
    on_network = np.zeros(nrows * ncols, dtype=bool)
    on_network[1 * ncols + 2] = True
    starts = np.arange(nrows * ncols)
    found = L._first_network_pixel(starts[::-1].copy(), fdr, on_network, nrows, ncols)[::-1]
    assert found.tolist() == [_walk_to_network(int(p), fdr, on_network, nrows, ncols) for p in starts]
    assert found[[0, 1, 5, 6, 7]].tolist() == [6] * 5
    assert found[[3, 4, 8, 9, 10, 11]].tolist() == [-1] * 6


def test_channel_cells_are_grouped_by_the_stream_pixel_they_drain_to():
    """Every water cell off the network is in the group of the network pixel its path first meets; the water below
    the network's last pixel drains off the raster without meeting one, so it is in no group."""
    fil, fdr, info, water = _channel()
    nrows, ncols = fil.shape
    flat_fdr = fdr.ravel()
    on_network = L._network_pixels(info, flat_fdr, nrows, ncols)
    keys, starts, cells = L._channel_cells(fdr, info, water)
    assert keys.tolist() == sorted(set(keys.tolist())) and np.all(on_network[keys])
    for i, key in enumerate(keys):
        for cell in cells[starts[i]:starts[i + 1]]:
            assert water.ravel()[cell] and not on_network[cell]
            assert _walk_to_network(int(cell), flat_fdr, on_network, nrows, ncols) == key
    grouped = set(cells.tolist())
    for cell in np.flatnonzero(water.ravel() & ~on_network):
        assert (cell in grouped) == (_walk_to_network(int(cell), flat_fdr, on_network, nrows, ncols) != -1)
    assert not grouped & {28 * ncols + 5, 28 * ncols + 6, 28 * ncols + 7}


def test_marking_td_r_marks_the_widened_cells_and_unmarking_clears_them():
    fil, fdr, info, water = _channel()
    nrows, ncols = fil.shape
    channel = L._channel_cells(fdr, info, water)
    downstream = L._downstream_exclusion(1, info, fdr.ravel(), nrows, ncols)
    excluded = np.zeros((nrows + 2) * (ncols + 2), dtype=bool)
    W._mark_downstream(excluded, downstream, *L._channel_lookup(channel, fil.size), ncols, True)
    widened = L._widen_downstream(downstream, channel)
    assert sorted(np.flatnonzero(excluded)) == sorted(L._to_local(widened, ncols, 0, 0, ncols + 2).tolist())
    W._mark_downstream(excluded, downstream, *L._channel_lookup(channel, fil.size), ncols, False)
    assert not excluded.any()


def test_a_workspace_widens_td_r_with_its_channel():
    fil, fdr, info, water = _channel()
    channel = L._channel_cells(fdr, info, water)
    workspace = _workspace(fil, fdr, channel)
    for stream_id in (1, 2):
        args = (fil, fdr, stream_id, info, 0.1, 0.1, 1.0, False)
        windowed = L._segment_rows(*args, channel=channel)
        whole = L._segment_rows(*args, channel=channel, workspace=workspace)
        assert all(np.array_equal(a, b) for a, b in zip(windowed[:3], whole[:3]))


def test_widening_adds_the_groups_of_the_td_pixels_only():
    keys, starts, cells = np.array([3, 7, 9]), np.array([0, 2, 3, 5]), np.array([30, 31, 70, 90, 91])
    widened = L._widen_downstream(np.array([7, 8, 9]), (keys, starts, cells))
    assert sorted(widened.tolist()) == [7, 8, 9, 70, 90, 91]
    assert L._widen_downstream(np.array([4]), (keys, starts, cells)).tolist() == [4]
    assert L._widen_downstream(np.empty(0, np.int64), (keys, starts, cells)).size == 0


def test_with_a_water_mask_a_segment_s_water_stays_out_of_the_channel_below_it(tmp_path):
    """Reach 1's water runs down the channel beside reach 2's stream pixels, which alone are its TD(R), to the edge of
    the raster (and from the channel's sides, a step into the stream). With the water mask, the channel below reach 1
    joins its TD(R), and its water stops at row 12: as in the reference, water descending from outside TD(R) may end
    its path on it, one step past reach 1's last pixel, but not run along it."""
    fil, _, _, water = _channel()
    ncols = fil.shape[1]
    line = _channel_library(tmp_path, "line")
    channel = _channel_library(tmp_path, "channel", water_mask=water.astype(np.uint8))

    def reach_1_rows(library):
        fpp = library.loc[library["segment"] == 0, "FPP"].to_numpy()
        return fpp // ncols, fpp % ncols

    rows, cols = reach_1_rows(line)
    assert rows.max() == fil.shape[0] - 1 and set(cols[rows > 12]) == {5, 6, 7}
    rows, cols = reach_1_rows(channel)
    assert rows.max() == 12 and cols[rows == 12].tolist() == [6] and set(cols) <= {5, 6, 7}
    # A wider TD(R) only takes paths away: no segment floods a cell it didn't, or floods one shallower
    merged = channel.merge(line, on=["segment", "FPP"], how="left", suffixes=("", "_line"))
    assert merged["DTF_line"].notna().all()
    assert (merged["DTF"].astype(float) >= merged["DTF_line"].astype(float)).all()
    assert len(channel) < len(line)


def test_a_parallel_build_is_the_serial_one(tmp_path):
    """The threads share the rasters and the channel cells, and the rows come out in the same order."""
    water = _channel()[3].astype(np.uint8)
    serial = _channel_library(tmp_path, "serial", water_mask=water)
    parallel = _channel_library(tmp_path, "parallel", water_mask=water, parallel=True, processes=2)
    assert len(serial) and parallel.equals(serial)


# --------------------------------------------------------------------------
# The library file
# --------------------------------------------------------------------------

def _toy_library(n=11):
    rng = np.random.default_rng(0)
    return {"FSP": np.sort(rng.integers(0, 50, n)).astype(np.int32), "FPP": rng.integers(0, 500, n).astype(np.int32),
            "DTF": rng.uniform(0, 25, n).astype(np.float32), "segment": rng.integers(0, 3, n).astype(np.int32),
            "fill": rng.uniform(-3, 30, n).astype(np.float32)}


def test_save_library_writes_row_groups_of_decimals(tmp_path, monkeypatch):
    """A row group at a time, zstd-compressed, the DTF and fill as 2-decimal numbers; a table or its columns'
    arrays save the same."""
    import pyarrow.parquet as pq
    monkeypatch.setattr(LW, "_LIBRARY_SLICE_ROWS", 4)
    columns = _toy_library()
    L.save_library(columns, tmp_path / "arrays.parquet")
    L.save_library(pd.DataFrame(columns), tmp_path / "table.parquet")
    file = pq.ParquetFile(tmp_path / "arrays.parquet")
    assert file.metadata.num_row_groups == 3
    assert file.metadata.row_group(0).column(0).compression == "ZSTD"
    saved = pd.read_parquet(tmp_path / "arrays.parquet")
    assert saved.equals(pd.read_parquet(tmp_path / "table.parquet"))
    assert saved["FPP"].tolist() == columns["FPP"].tolist()
    for column in ("DTF", "fill"):
        assert np.allclose(saved[column].astype(float), columns[column].round(2), rtol=0, atol=1e-5)


def test_library_writer_writes_whole_row_groups_whatever_the_parts(tmp_path, monkeypatch):
    """Parts of any size make the row groups save_library makes of all the rows at once; a CSV library has the same
    rows."""
    import pyarrow.parquet as pq
    monkeypatch.setattr(LW, "_LIBRARY_SLICE_ROWS", 4)
    columns = _toy_library()
    for suffix in ("parquet", "csv"):
        with LW.LibraryWriter(tmp_path / f"parts.{suffix}") as writer:
            for start, stop in ((0, 0), (0, 1), (1, 6), (6, 7), (7, 11)):
                writer.write({c: v[start:stop] for c, v in columns.items()})
    L.save_library(columns, tmp_path / "whole.parquet")
    metadata = pq.ParquetFile(tmp_path / "parts.parquet").metadata
    assert [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)] == [4, 4, 3]
    assert pd.read_parquet(tmp_path / "parts.parquet").equals(pd.read_parquet(tmp_path / "whole.parquet"))
    csv = pd.read_csv(tmp_path / "parts.csv")
    assert list(csv) == list(columns) and csv["FPP"].tolist() == columns["FPP"].tolist()
    assert np.allclose(csv["DTF"], columns["DTF"])


def test_a_background_writer_writes_what_a_plain_one_does(tmp_path, monkeypatch):
    monkeypatch.setattr(LW, "_LIBRARY_SLICE_ROWS", 4)
    monkeypatch.setattr(LW, "_WRITES_AHEAD", 2)
    columns = _toy_library()
    take = lambda start, stop: {c: v[start:stop] for c, v in columns.items()}
    with LW.LibraryWriter(tmp_path / "background.parquet", background=True) as writer:
        for start, stop in ((0, 1), (1, 6), (6, 7), (7, 11)):
            writer.write_later(take, start, stop)
    L.save_library(columns, tmp_path / "plain.parquet")
    assert pd.read_parquet(tmp_path / "background.parquet").equals(pd.read_parquet(tmp_path / "plain.parquet"))


def test_a_failure_in_the_writer_thread_is_raised_and_leaves_no_library(tmp_path):
    def fail():
        raise ValueError("bad rows")
    with pytest.raises(ValueError, match="bad rows"), LW.LibraryWriter(tmp_path / "library.parquet", background=True) as writer:
            writer.write_later(lambda: _toy_library())
            writer.write_later(fail)
    assert not (tmp_path / "library.parquet").exists() and not (tmp_path / "library.parquet.partial").exists()


def test_library_writer_without_rows_still_writes_the_columns(tmp_path):
    with LW.LibraryWriter(tmp_path / "empty.parquet") as writer:
        writer.write({c: v[:0] for c, v in _toy_library().items()})
    empty = pd.read_parquet(tmp_path / "empty.parquet")
    assert empty.empty and list(empty) == list(_toy_library())


def test_a_failed_build_leaves_no_library(tmp_path, monkeypatch):
    """The segments solved before the failure were written, but neither they nor the partial file are left: a rerun
    that skips finished libraries must not take them for one."""
    solve, calls = L._segment_rows, []

    def fail_on_the_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise MemoryError
        return solve(*args, **kwargs)
    monkeypatch.setattr(L, "_segment_rows", fail_on_the_second)
    with pytest.raises(MemoryError):
        _slope_library(tmp_path, "failed", max_reach_length=4.0)
    assert not (tmp_path / "failed.parquet").exists() and not (tmp_path / "failed.parquet.partial").exists()


def test_library_rows_are_sorted_and_skip_cells_without_a_dem_value():
    fsp = np.array([7, 3, 7, 3, 3], dtype=np.int32)
    fpp = np.array([5, 9, 1, 2, 4], dtype=np.int32)
    fill_depth = np.zeros(10, dtype=np.float32)
    fill_depth[4], fill_depth[9] = np.nan, 2.5
    rows = L._library_rows(11, fsp, fpp, np.arange(5, dtype=np.float32), fill_depth)
    assert list(rows) == ["FSP", "FPP", "DTF", "segment", "fill"]
    assert rows["FSP"].tolist() == [3, 3, 7, 7] and rows["FPP"].tolist() == [2, 9, 1, 5]
    assert rows["DTF"].tolist() == [3, 1, 2, 0] and rows["fill"].tolist() == [0, 2.5, 0, 0]
    assert rows["segment"].tolist() == [11] * 4 and rows["segment"].dtype == np.int32


def test_ordered_map_keeps_the_order_and_few_tasks_ahead():
    """The results come in the tasks' order, though the first finishes last, and no more than ``ahead`` tasks are
    ever submitted but not yet taken."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    submitted, taken, most = [0], [0], [0]

    def first_is_slow(task):
        time.sleep(0.05 if task == 0 else 0.0)
        return 10 * task
    with ThreadPoolExecutor(3) as pool:
        submit = pool.submit

        def counted(fn, *args):
            submitted[0] += 1
            most[0] = max(most[0], submitted[0] - taken[0])
            return submit(fn, *args)
        pool.submit = counted
        results = []
        for result in L._ordered_map(pool, first_is_slow, range(20), 4):
            taken[0] += 1
            results.append(result)
    assert results == [10 * task for task in range(20)] and most[0] == 4


def test_library_parts_read_a_file_as_a_lazy_frame_would(tmp_path, monkeypatch):
    """A file read through pyarrow a batch at a time gives the rows a LazyFrame of it does: its decimals as the
    integer hundredths whose ``_metres`` are the LazyFrame's Float32."""
    import polars as pl
    monkeypatch.setattr(LW, "_LIBRARY_SLICE_ROWS", 4)
    monkeypatch.setattr(M, "_READ_ROWS", 4)
    path = tmp_path / "library.parquet"
    L.save_library(_toy_library(), path)
    lazy = pl.scan_parquet(path)
    lazy = lazy.with_columns([pl.col(c).cast(pl.Float32) for c, d in lazy.collect_schema().items() if d == pl.Decimal])
    columns = ["segment", "FPP", "DTF", "fill"]
    from_file = list(M._library_parts(str(path), columns))
    from_lazy = list(M._library_parts(lazy, columns))
    assert [len(part["FPP"]) for part in from_file] == [4, 4, 3]
    for column in columns:
        values = np.concatenate([part[column] for part in from_file])
        expected = np.concatenate([part[column] for part in from_lazy])
        if column in ("DTF", "fill"):
            assert values.dtype == np.int32 and expected.dtype == np.float32
            values = M._metres(values)
        assert values.dtype == expected.dtype and np.array_equal(values, expected)
    assert M._library_columns(str(path)) == M._library_columns(lazy) == list(_toy_library())


def test_hundredths_are_polars_decimal_casts():
    """float32(integer / 100) is what polars' cast of a decimal32(7, 2) gives, over the whole of its range (every
    seventh value, and its ends)."""
    import polars as pl
    import pyarrow as pa
    integers = np.r_[np.arange(-9_999_999, 10_000_000, 7), 9_999_999, -1, 0, 1].astype(np.int32)
    column = pa.Array.from_buffers(pa.decimal32(7, 2), integers.size, [None, pa.py_buffer(integers.tobytes())])
    assert np.array_equal(M._metres(M._column_values(column)), pl.from_arrow(column).cast(pl.Float32).to_numpy())


def test_decimals_are_written_as_arrow_casts_them():
    """The library's decimals are Arrow's cast of the values as float32 rounded to two places, over the whole of
    decimal32(7, 2) (every seventh value, and its ends); a value that doesn't fit, or isn't a number, is refused."""
    import pyarrow as pa
    hundredths = np.r_[np.arange(-9_999_999, 10_000_000, 7), 9_999_999, -1, 0, 1]
    values = (hundredths / 100.0).astype(np.float32)
    kind = pa.decimal32(7, 2)
    assert LW._as_decimal(values, kind).equals(pa.array(values.round(2)).cast(kind))
    for bad in (1e5, np.nan):
        with pytest.raises(ValueError, match="doesn't fit"):
            LW._as_decimal(np.array([1.0, bad], np.float32), kind)


def test_column_values_read_numbers_in_place_as_pyarrow_converts_them():
    """Numbers without nulls are read from the column's buffer, a slice of one too, as pyarrow's to_numpy gives them;
    a column with nulls is converted by pyarrow."""
    import pyarrow as pa
    for values in (np.arange(-5, 7, dtype=np.int32), np.arange(9, dtype=np.uint16), np.linspace(0, 1, 6)):
        column = pa.array(values)[2:]
        read = M._column_values(column)
        assert read.dtype == values.dtype and np.array_equal(read, column.to_numpy())
    with_null = pa.array([1.5, None, 2.0])
    assert np.array_equal(M._column_values(with_null), [1.5, np.nan, 2.0], equal_nan=True)


def test_a_sparse_fill_reads_as_the_raster_and_a_dense_one_is_the_raster(tmp_path):
    """Filling raises a few cells: those alone are kept, and read back as the raster would give them, NaN where the
    DEM has no data; with most cells raised, the raster is kept whole."""
    from osgeo import gdal
    rng = np.random.default_rng(6)
    dem = (100.0 + rng.random((1100, 9))).astype(np.float32)  # rows enough for several strips
    dem[3, 4] = -9999.0
    path = str(tmp_path / "dem.tif")
    ds = gdal.GetDriverByName("GTiff").Create(path, 9, 1100, 1, gdal.GDT_Float32)
    ds.GetRasterBand(1).WriteArray(dem)
    ds = None
    pixels = np.arange(dem.size)
    for raised in (0.02, 0.9):
        filled = dem.copy()
        up = rng.random(dem.shape) < raised
        filled[up] += np.float32(0.5)
        expected = (filled - dem).ravel()
        expected[3 * 9 + 4] = np.nan
        fill = L._fill_depth(path, filled, BG)
        assert isinstance(fill, L.SparseFill if raised < 0.1 else np.ndarray)
        assert np.array_equal(fill[pixels], expected, equal_nan=True)


def test_the_water_mask_must_be_on_the_dem_s_grid(tmp_path):
    with pytest.raises(ValueError, match="water_mask has shape"):
        _channel_library(tmp_path, "bad", water_mask=np.ones((5, 5), np.uint8))
