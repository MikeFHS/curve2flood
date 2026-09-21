"""Tests for the FLDPLN library solvers.

The important one is :func:`test_iterative_matches_reference_model`, which pins
the iterative solver bit-for-bit against ``fldpln_model_v5ram.py`` -- the port of
the MATLAB model the spreader descends from.  It is run with only the two
conventions this package deliberately changed patched in: WhiteboxTools D8 codes
instead of the MATLAB ones, and the stream table rebuilt in the reference's own
column layout.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from curve2flood.spreaders import fldpln as F

DATA_DIR = Path(__file__).parent / "data" / "scottsbluff_fldpln" / "inputs"
REFERENCE_MODEL = Path(__file__).resolve().parents[1] / "fldpln_model_v5ram.py"

BG = -9999.0
# A few segments that exercise backfill, spillover and the steady-state loop
# without taking long in the pure-Python reference.
SEGMENTS = [760363429, 760418328, 760308499, 760395048, 760388378, 760536470]


def _opposite(pos: int) -> int:
    dr, dc = F.NEIGHBOR_DELTAS[pos]
    return F.NEIGHBOR_DELTAS.index((-dr, -dc))


@pytest.fixture(scope="module")
def rasters():
    if not DATA_DIR.exists():
        pytest.skip("Scottsbluff FLDPLN fixture is not available.")
    from osgeo import gdal

    gdal.UseExceptions()
    dem = gdal.Open(str(DATA_DIR / "fabdem_fldpln_bathymetry.tif")).ReadAsArray().astype(np.float32)
    fil = gdal.Open(str(DATA_DIR / "fabdem_filled.tif")).ReadAsArray().astype(np.float32)
    fdr = gdal.Open(str(DATA_DIR / "fabdem_flowdir.tif")).ReadAsArray().astype(np.uint8)
    info = pd.read_parquet(DATA_DIR / "fabdem_stream_info.parquet")
    return dem, fil, fdr, info


def _segment(rasters, stream_id, solver, dh, fldmx, iterative_spill=True, margins=None):
    dem, fil, fdr, info = rasters
    previous = F._WINDOW_MARGINS
    if margins is not None:
        F._WINDOW_MARGINS = margins
    try:
        table = F.fldpln_library_for_segment(
            dem, fil, fdr, int(stream_id), info.values, dh, 0.1, fldmx,
            iterative_spill, 0.0, BG, 0.0, solver)
    finally:
        F._WINDOW_MARGINS = previous
    return table.sort_values("FPP").reset_index(drop=True)


# --------------------------------------------------------------------------
# D8 conventions
# --------------------------------------------------------------------------

def test_inflow_is_outflow_reversed():
    """A neighbour drains into the centre exactly when it points back at it."""
    for pos in range(8):
        assert F.INFLOW[pos] == F.OUTFLOW[_opposite(pos)]


def test_outflow_matches_next_downstream():
    """OUTFLOW[pos] must route a cell to NEIGHBOR_DELTAS[pos], which is what the
    solvers' flat offsets (``dr * ncols + dc``) assume."""
    nrows = ncols = 5
    centre = 2 * ncols + 2
    for pos, (dr, dc) in enumerate(F.NEIGHBOR_DELTAS):
        fdr = np.zeros(nrows * ncols, dtype=np.uint8)
        fdr[centre] = F.OUTFLOW[pos]
        assert F._next_downstream(centre, fdr, nrows, ncols) == (2 + dr) * ncols + (2 + dc)


# --------------------------------------------------------------------------
# Fidelity to the MATLAB-derived reference
# --------------------------------------------------------------------------

@pytest.mark.parametrize("stream_id", SEGMENTS)
def test_iterative_matches_reference_model(rasters, stream_id):
    if not REFERENCE_MODEL.exists():
        pytest.skip("fldpln_model_v5ram.py is not available.")
    spec = importlib.util.spec_from_file_location("_v5ram", REFERENCE_MODEL)
    reference = importlib.util.module_from_spec(spec)
    sys.modules["_v5ram"] = reference
    spec.loader.exec_module(reference)

    # Convention 1: WhiteboxTools D8 codes instead of the MATLAB/ESRI ones.
    reference.INFLOW = np.asarray(F.INFLOW, dtype=np.int32)
    reference.OUTFLOW = np.asarray(F.OUTFLOW, dtype=np.int32)

    dem, fil, fdr, info = rasters
    # Convention 2: the reference indexes seg_info by row, with length in col 4.
    seg_info = np.zeros((len(info), 5))
    seg_info[:, 0] = info["start_pixel"].values
    seg_info[:, 1] = info["end_pixel"].values
    seg_info[:, 4] = info["length"].values
    row = int(np.flatnonzero(info["stream_id"].values == stream_id)[0])

    dh, fldmx = 0.5, 3.0
    result = reference.fldpln_model_v5ram(
        row, {"seg": seg_info}, fil, fdr.astype(np.int32), 0.1, fldmx, dh, 0.0,
        True, BG, shape=fil.shape, demdat=dem)
    table = np.asarray(result.fldpln)
    expected = pd.DataFrame({
        "FSP": table[:, 0].astype(np.int64),
        "FPP": table[:, 1].astype(np.int64),
        "DTF": table[:, 2].astype(np.float32),
    }).sort_values("FPP").reset_index(drop=True)

    actual = _segment(rasters, stream_id, "iterative", dh, fldmx)

    assert np.array_equal(actual["FPP"].values.astype(np.int64), expected["FPP"].values)
    assert np.array_equal(actual["DTF"].values, expected["DTF"].values)
    assert np.array_equal(actual["FSP"].values.astype(np.int64), expected["FSP"].values)


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
    _, fil, _, _ = rasters
    flat = fil.ravel()
    table = _segment(rasters, 760363429, "exact", 0.1, 5.0)
    rise = flat[table["FPP"].values] - flat[table["FSP"].values]
    assert (table["DTF"].values >= -1e-6).all()
    # Tolerance covers accumulated float32 elevation differencing plus the exact
    # solver's fixed-point step; measured worst case over this fixture is ~6 mm.
    assert (table["DTF"].values >= rise - 0.02).all()


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("solver,dh", [("exact", 0.1), ("iterative", 0.5)])
def test_window_size_does_not_change_the_result(rasters, solver, dh):
    _, fil, _, _ = rasters
    whole = (max(fil.shape),)
    for stream_id in SEGMENTS:
        windowed = _segment(rasters, stream_id, solver, dh, 7.2)
        unwindowed = _segment(rasters, stream_id, solver, dh, 7.2, margins=whole)
        assert np.array_equal(windowed["FPP"].values, unwindowed["FPP"].values)
        assert np.array_equal(windowed["DTF"].values, unwindowed["DTF"].values)
        assert np.array_equal(windowed["FSP"].values, unwindowed["FSP"].values)


def test_tiny_window_still_grows_to_the_right_answer(rasters):
    """A window far too small must be detected and grown, not silently truncated."""
    _, fil, _, _ = rasters
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
