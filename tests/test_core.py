"""Small tests of the Curve2Flood driver: running it, input files and raster helpers."""

import logging
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from osgeo import gdal, osr
from scipy.ndimage import generate_binary_structure, label, uniform_filter

from curve2flood import (
    LOG,
    Config,
    Curve2Flood_MainFunction,
    _ids,
    _raster,
    bathymetry,
    cli,
    core,
    outputs,
    rating_curves,
    remove_cells_not_connected,
)
from curve2flood.spreaders import fldpln, multi_point

gdal.UseExceptions()


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

def test_main_function_needs_settings():
    with pytest.raises(ValueError, match="input file, a dict of settings or a Config"):
        Curve2Flood_MainFunction()


def test_main_function_reports_a_missing_dem(tmp_path):
    with pytest.raises(FileNotFoundError, match="DEM_File .*missing.tif doesn't exist"):
        Curve2Flood_MainFunction(args={"DEM_File": str(tmp_path / "missing.tif")})
    with pytest.raises(FileNotFoundError, match="DEM_File"):
        Curve2Flood_MainFunction(Config(dem_file=str(tmp_path / "missing.tif")))


# --------------------------------------------------------------------------
# Input files
# --------------------------------------------------------------------------

@pytest.mark.parametrize("suffix", [".csv", ".parquet", ".pq"])
def test_flow_file_events(tmp_path, suffix):
    flows = pd.DataFrame({"COMID": [11, 12], "q1": [5.0, np.nan], "q2": [7.0, 8.0]})
    path = tmp_path / f"flows{suffix}"
    if suffix == ".csv":
        flows.to_csv(path, index=False)
    else:
        flows.to_parquet(path)
    assert core.check_flow_file(str(path)) == 2
    assert core.FindFlowRateForEachCOMID_Ensemble(str(path), 0) == {11: 5.0, 12: 0.0}  # missing flows are zero
    assert core.FindFlowRateForEachCOMID_Ensemble(str(path), 1) == {11: 7.0, 12: 8.0}


def test_set_stream_locations(tmp_path):
    path = tmp_path / "vdt.csv"
    pd.DataFrame({"COMID": [7, 8], "Row": [0, 2], "Col": [1, 0]}).to_csv(path, index=False)
    assert core.Set_Stream_Locations(3, 2, str(path)).tolist() == [[-9999, 7], [-9999, -9999], [8, -9999]]


def _manning_n(path, land_cover):
    """Each cell's Manning's n from the table at ``path``, warning of the land cover it doesn't list."""
    unlisted = set()
    n = outputs._manning_n(land_cover, outputs._manning_lookup(str(path)), unlisted)
    outputs._warn_unlisted_land_cover(unlisted, str(path))
    return n


def test_read_manning_table(tmp_path):
    path = tmp_path / "manning.txt"
    path.write_text("LC_ID\tDescription\tManning_n\n10\tTrees\t0.12\n80\tWater\t0.03\n")
    land_cover = np.array([[10, 80], [80, 10]])
    assert np.allclose(_manning_n(path, land_cover), [[0.12, 0.03], [0.03, 0.12]])
    # A float land cover raster is truncated to its codes, and the values are the float64 table's, in float32
    n = _manning_n(path, np.array([[10.0, 80.7]], np.float32))
    assert n.dtype == np.float32 and n.tolist() == [[np.float32(0.12), np.float32(0.03)]]


def test_read_raster_flips_south_up_rasters(tmp_path):
    path = str(tmp_path / "south_up.tif")
    ds = gdal.GetDriverByName("GTiff").Create(path, 2, 2, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((0, 1, 0, 0, 0, 1))  # positive pixel height: row 0 is the southern edge
    ds.GetRasterBand(1).WriteArray(np.array([[1, 2], [3, 4]], np.uint8))
    ds = None
    array, geotransform, _ = core.Read_Raster_GDAL(path)
    assert array.tolist() == [[3, 4], [1, 2]]
    assert geotransform == (0, 1, 0, 2, 0, -1)


def test_curve_file_fills_the_rasters(tmp_path):
    path = tmp_path / "curves.csv"
    pd.DataFrame({
        "COMID": [1, 1, 2, 2], "Row": [0, 1, 1, 2], "Col": [0, 0, 1, 1], "BaseElev": [100.0, 100.0, 50.0, 50.0],
        "depth_a": [0.5, 0.5, 1.0, 1.0], "depth_b": 0.5, "tw_a": [2.0, 2.0, 1.0, 1.0], "tw_b": 0.5,
        "vel_a": 1.0, "vel_b": 0.0, "Slope": [0.001, 0.002, 0.003, 0.004],
    }).to_csv(path, index=False)
    T = np.full((3, 2), -1.0, np.float32)
    W = np.full((3, 2), np.nan, np.float32)
    S = np.full((3, 2), np.nan, np.float32)
    _, depth, tw_max, T, W, S = rating_curves.Calculate_TW_D_V_ForEachCOMID_CurveFile(
        str(path), {1: 4.0, 2: 16.0}, np.array([1, 2]), T, W, S, 1.5)
    # Depth and top width are a * Q**b; WSE adds the base elevation; top widths are scaled by TW_MultFact
    assert W[0, 0] == pytest.approx(101.0) and W[2, 1] == pytest.approx(54.0)
    assert T[0, 0] == pytest.approx(6.0) and T[2, 1] == pytest.approx(6.0)
    assert S[1, 0] == pytest.approx(0.002)
    assert depth == {1: pytest.approx(1.0), 2: pytest.approx(4.0)}
    assert tw_max == pytest.approx(6.0)


# --------------------------------------------------------------------------
# Rating curves
# --------------------------------------------------------------------------

def test_interp1d_numba_interpolates_and_extrapolates():
    x = np.array([1, 2, 4], np.float32)
    y = np.array([10, 20, 40], np.float32)
    assert rating_curves.interp1d_numba(x, y, 3.0) == pytest.approx(30.0)
    assert rating_curves.interp1d_numba(x, y, 0.0) == pytest.approx(0.0)
    assert rating_curves.interp1d_numba(x, y, 5.0) == pytest.approx(50.0)


def test_top_width_at_baseflow():
    flows = np.array([1, 5, 10], np.float32)
    top_widths = np.array([3, 7, 9], np.float32)
    find = rating_curves.Find_TopWidth_at_Baseflow_when_using_VDT
    assert find(np.float32(4), flows, top_widths) == 7  # the first flow at or above the baseflow
    assert find(np.float32(5), flows, top_widths) == 7
    assert find(np.float32(20), flows, top_widths) == 9  # above every flow: the last


def test_vdt_interpolate():
    q = np.array([[10, 20, 30]] * 2, np.float32)
    t = np.array([[5, 10, 15]] * 2, np.float32)
    wse = np.array([[101, 102, 103]] * 2, np.float32)
    v = np.array([[1, 2, 3]] * 2, np.float32)
    flow = np.array([15, 50], np.float32)  # inside the table, and above it
    baseflow = np.array([20, 0], np.float32)
    dem = np.array([100, 100], np.float32)
    top_width, depth, water, velocity = rating_curves.vdt_interpolate(flow, baseflow, q, t, wse, v, dem, np.array([2, 1], np.float32))
    assert water.tolist() == pytest.approx([101.5, 103.0])
    assert depth.tolist() == pytest.approx([1.5, 3.0])
    assert velocity.tolist() == pytest.approx([1.5, 3.0])
    # Row 0's top width (7.5) is raised to its top width at baseflow (10), then doubled
    assert top_width.tolist() == pytest.approx([20.0, 15.0])


def test_calculate_interpolated_vdt(tmp_path):
    curves = {"q_1": 10.0, "q_2": 20.0, "q_3": 30.0, "t_1": 5.0, "t_2": 10.0, "t_3": 15.0,
              "wse_1": 101.0, "wse_2": 102.0, "wse_3": 103.0, "v_1": 1.0, "v_2": 2.0, "v_3": 3.0}
    path = tmp_path / "vdt.parquet"
    vdt = pd.DataFrame({"COMID": [1, 1, 2, 3], "Row": [0, 1, 0, 1], "Col": [0, 0, 1, 1],
                        "Elev": [100.0, 100.0, 101.8, 100.0], "QBaseflow": 0.0, "Slope": 0.01, **curves})
    vdt.to_parquet(path)
    out = core.calculate_interpolated_vdt(str(path), {1: 15.0, 2: 15.0}, 2.0).sort("COMID", "Row")
    assert out["COMID"].to_list() == [1, 1, 2]  # reach 3 has no flow
    # Measured from the VDT's Elev, and kept at or above it
    assert out["WSE"].to_list() == pytest.approx([101.5, 101.5, 101.8])
    assert out["Depth"].to_list() == pytest.approx([1.5, 1.5, 0.001], abs=1e-5)
    assert out["TopWidth"].to_list() == pytest.approx([15.0, 15.0, 15.0])  # 7.5 m times TW_MultFact

    vdt.drop(columns="Elev").to_parquet(path)
    with pytest.raises(ValueError, match="no Elev column"):
        core.calculate_interpolated_vdt(str(path), {1: 15.0}, 2.0)


def test_calculate_interpolated_vdt_keeps_a_reach_s_ends_but_drops_an_outlying_width(tmp_path):
    """The WSE falls along the reach, so its 1st and 99th percentiles are the reach's two ends: both are kept. A cell
    whose top width is far beyond the others' is still dropped."""
    n = 101
    elev = 100.0 - 0.01 * np.arange(n)
    vdt = pd.DataFrame({"COMID": 1, "Row": np.arange(n), "Col": 0, "Elev": elev, "QBaseflow": 0.0, "Slope": 0.01,
                        "q_1": 10.0, "q_2": 20.0, "t_1": 5.0, "t_2": 10.0, "wse_1": elev + 1.0, "wse_2": elev + 2.0,
                        "v_1": 1.0, "v_2": 2.0})
    vdt.loc[50, ["t_1", "t_2"]] = [500.0, 1000.0]
    path = tmp_path / "vdt.parquet"
    vdt.to_parquet(path)

    out = core.calculate_interpolated_vdt(str(path), {1: 15.0}, 1.0).sort("Row")

    assert out["Row"].to_list() == [row for row in range(n) if row != 50]
    assert out["WSE"][0] == pytest.approx(101.5)
    assert out["WSE"][-1] == pytest.approx(100.5)


def test_top_widths_are_scaled_by_tw_multfact_alone(tmp_path):
    """Not also by the flow over the baseflow, as they once were: at ten times its baseflow a cell's 7.5 m top width
    is 15 m with a TW_MultFact of 2, not 45 m, from the VDT database and from a curve file alike."""
    vdt = pd.DataFrame({"COMID": 1, "Row": [0, 1], "Col": 0, "Elev": 100.0, "QBaseflow": 1.5, "Slope": 0.01,
                        "q_1": 10.0, "q_2": 20.0, "t_1": 5.0, "t_2": 10.0, "wse_1": 101.0, "wse_2": 102.0,
                        "v_1": 1.0, "v_2": 2.0})
    vdt.to_parquet(tmp_path / "vdt.parquet")
    out = core.calculate_interpolated_vdt(str(tmp_path / "vdt.parquet"), {1: 15.0}, 2.0)
    assert out["TopWidth"].to_list() == pytest.approx([15.0, 15.0])

    pd.DataFrame({"COMID": 1, "Row": [0, 1, 2], "Col": 0, "BaseElev": 100.0, "QBaseflow": 1.5, "depth_a": 0.5,
                  "depth_b": 0.5, "tw_a": 2.0, "tw_b": 0.5, "vel_a": 1.0, "vel_b": 0.0, "Slope": 0.01}).to_csv(
        tmp_path / "curve.csv", index=False)
    curve = rating_curves.calculate_interpolated_curvefile(str(tmp_path / "curve.csv"), {1: 16.0}, 2.0)
    assert curve["TopWidth"].tolist() == pytest.approx([16.0] * 3)  # 2 * 16 ** 0.5, times TW_MultFact


# --------------------------------------------------------------------------
# Rasters
# --------------------------------------------------------------------------

def test_convert_cell_size():
    utm = osr.SpatialReference()
    utm.ImportFromEPSG(32613)
    assert core.convert_cell_size(30.0, -30.0, 0, 0, utm.ExportToWkt()) == (30.0, 30.0)

    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    dx, dy = core.convert_cell_size(1 / 3600, 1 / 3600, -0.5, 0.5, wgs84.ExportToWkt())
    assert dx == pytest.approx(30.92, abs=0.01)  # one arc-second at the equator
    assert dy == pytest.approx(30.72, abs=0.01)


def test_create_weightbox_is_inverse_squared_distance():
    box = core.create_weightbox(2, 10.0, 20.0)
    assert box.shape == (5, 5) and box.dtype == np.float32
    assert box[2, 2] == np.float32(1 / 0.0001)  # the centre is capped
    assert box[2, 3] == pytest.approx(1 / 10.0**2)
    assert box[3, 2] == pytest.approx(1 / 20.0**2)
    assert box[0, 4] == pytest.approx(1 / (40.0**2 + 20.0**2))


def test_remove_cells_not_connected_keeps_patches_touching_a_stream():
    flood = np.array([
        [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1],  # (1, 2) touches (0, 1) diagonally
        [0, 0, 0, 0, 1],
    ], dtype=np.uint8)
    streams = np.zeros_like(flood)
    streams[0, 0] = 7
    expected = flood.copy()
    expected[1:, 4] = 0
    assert np.array_equal(remove_cells_not_connected(flood, streams), expected)


def test_flood_flooded_cells_in_map():
    values = np.array([[1.0, np.nan, np.nan, np.nan, 5.0]], np.float32)
    flood = np.array([[1, 1, 0, 1, 0]], np.uint8)
    # Gaps take the nearest flooded value; cells outside the flood are NaN
    out = outputs.Flood_Flooded_Cells_in_Map(values, flood, eps=0.01)
    assert np.allclose(out, [[1.0, 1.0, np.nan, 1.0, np.nan]], equal_nan=True)
    # A flood with no values at all gets eps
    out = outputs.Flood_Flooded_Cells_in_Map(np.full((1, 3), np.nan, np.float32), np.array([[0, 1, 1]]), eps=0.01)
    assert np.allclose(out, [[np.nan, 0.01, 0.01]], equal_nan=True)

def test_flooded_cells_are_matched_in_place_as_in_a_copy():
    """In place, the raster itself becomes what a copy would: its sources are read as they were."""
    rng = np.random.default_rng(3)
    values = rng.random((30, 40)).astype(np.float32)
    values[rng.random(values.shape) < 0.7] = np.nan
    flood = (rng.random(values.shape) < 0.8).astype(np.uint8)
    copy = outputs.Flood_Flooded_Cells_in_Map(values, flood, eps=0.01)
    in_place = values.copy()
    assert outputs.Flood_Flooded_Cells_in_Map(in_place, flood, eps=0.01, in_place=True) is in_place
    assert np.array_equal(in_place, copy, equal_nan=True)
    with pytest.raises(TypeError):
        outputs.Flood_Flooded_Cells_in_Map(values.astype(np.float64), flood, in_place=True)


def _labelled_patches(flood, streams):
    """remove_cells_not_connected by SciPy's labels, as it was written."""
    labels, _ = label(flood, generate_binary_structure(2, 2))
    return flood * np.isin(labels, np.unique(labels[(streams > 0) & (labels > 0)]))


@pytest.mark.parametrize("dtype", [np.uint8, np.bool_, np.float32, np.int32])
def test_the_patches_kept_are_scipy_s_labels_touching_a_stream(dtype):
    """Random patches of every density, NaN and negative values among them, a checkerboard (all corners) and a
    diagonal: the patches SciPy labels, those touching a stream kept, in the flood's type."""
    rng = np.random.default_rng(5)
    floods = [rng.random((50, 60)) < p for p in (0.2, 0.45, 0.6)]
    floods += [np.add.outer(np.arange(9), np.arange(11)) % 2 == 0, np.eye(12, 15, dtype=bool)]
    for flood in floods:
        flood = np.where(flood, rng.normal(size=flood.shape), 0).astype(dtype)
        if dtype == np.float32:
            flood[rng.random(flood.shape) < 0.02] = np.nan
        streams = np.where(rng.random(flood.shape) < 0.01, 7, -9999)
        kept = remove_cells_not_connected(flood, streams)
        expected = _labelled_patches(flood, streams)
        assert kept.dtype == expected.dtype and np.array_equal(kept, expected, equal_nan=True)


def test_create_positive_max_array():
    a = np.array([[1.0, np.nan, -2.0]], np.float32)
    b = np.array([[3.0, np.nan, 0.0]], np.float32)
    maximum = outputs.accumulate_positive_max(outputs.accumulate_positive_max(None, a), b)
    assert np.allclose(outputs.finish_positive_max(maximum), [[3.0, np.nan, np.nan]], equal_nan=True)


def test_positive_max_is_folded_in_an_event_at_a_time():
    """No event (None) changes nothing, and the running maximum is the maximum of all the events at once."""
    rng = np.random.default_rng(0)
    events = [rng.normal(size=(5, 6)).astype(np.float32) for _ in range(4)]
    events[1][0, 0] = np.nan
    maximum = None
    for event in events + [None]:
        maximum = core.accumulate_positive_max(maximum, event)
    stacked = np.stack(events)
    expected = np.nanmax(stacked, where=stacked > 0, axis=0, initial=-np.inf)
    expected[expected == -np.inf] = np.nan
    assert np.array_equal(outputs.finish_positive_max(maximum), expected, equal_nan=True)
    assert outputs.accumulate_positive_max(None, None) is None


def test_flood_water_land_cover_and_stream_cells():
    flood = np.zeros((2, 3), np.uint8)
    streams = np.array([[0, 9, 0], [0, 0, -9999]])
    land_cover = np.array([[80, 10, 10], [10, 10, 80]])
    out = outputs.Flood_WaterLC_and_STRM_Cells_in_Flood_Map(flood, streams, land_cover, 80)
    assert out.tolist() == [[100, 100, 0], [0, 0, 100]]


def test_flood_bathy_mask_cells():
    flood = np.array([[0, 100, 0], [0, 0, 0]], np.uint8)
    mask = np.array([[1, 0, 0], [0, 0, 1]], np.uint8)
    assert outputs.Flood_Bathy_Mask_in_Flood_Map(flood, mask).tolist() == [[100, 100, 0], [0, 0, 100]]


# The grid of every raster _write writes: 30 m cells in UTM zone 15N
_GEOTRANSFORM = (500000.0, 30.0, 0.0, 4500000.0, 0.0, -30.0)
_SRS = osr.SpatialReference()
_SRS.ImportFromEPSG(32615)


def _write(path, array, data_type):
    ds = gdal.GetDriverByName("GTiff").Create(str(path), array.shape[1], array.shape[0], 1, data_type)
    ds.SetGeoTransform(_GEOTRANSFORM)
    ds.SetProjection(_SRS.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(array)
    ds = None


@pytest.mark.parametrize("flood_mask", [False, True])
def test_the_bathymetry_water_mask_is_flooded_with_flood_bathy_mask(tmp_path, flood_mask):
    # A valley whose stream runs along row 10, flooded 0.5 m deep; the water mask also has a pond up its side, 3 m
    # above the stream and joined to it, which the water doesn't reach
    rows, cols = np.mgrid[0:21, 0:30]
    dem = (100.0 + 0.01 * cols + 1.0 * np.abs(rows - 10)).astype(np.float32)
    streams = np.zeros(dem.shape, np.int32)
    streams[10, 2:28] = 7
    mask = (streams > 0).astype(np.uint8)
    mask[7:10, 14:17] = 1
    _write(tmp_path / "dem.tif", dem, gdal.GDT_Float32)
    _write(tmp_path / "streams.tif", streams, gdal.GDT_Int32)
    _write(tmp_path / "mask.tif", mask, gdal.GDT_Byte)
    Curve2Flood_MainFunction(args={
        "DEM_File": str(tmp_path / "dem.tif"), "Stream_File": str(tmp_path / "streams.tif"),
        "OutFLD": str(tmp_path / "flood.tif"), "Set_Depth": 0.5, "TopWidthPlausibleLimit": 150,
        "Make_Output_GPKG": False, "BathyWaterMask": str(tmp_path / "mask.tif"), "Flood_Bathy_Mask": flood_mask,
    }, quiet=True)
    flood = gdal.Open(str(tmp_path / "flood.tif")).ReadAsArray()
    assert (flood[10, 2:28] > 0).all()
    assert (flood[7, 14:17] > 0).all() == flood_mask


def _fldpln_depth(tmp_path, vdt_columns=None, library_metadata=None, with_dem=True, **inputs):
    """A stream along the middle row of a 3x5 grid, flowing east, 1.5 m deep (WSE 101.5 over a 100 m Elev, with no
    fill), mapped with the FLDPLN mapper and these extra inputs, from tables that all end in .pq; the depth raster.
    Its own cells flood 1.4 m deep, the cell above its first pixel 1.3 m, and the cell below its last, with a DTF of
    1.8 m, stays dry. ``vdt_columns`` are added to its VDT database, and ``library_metadata`` to its library.
    Without ``with_dem`` the run isn't given the DEM."""
    _write(tmp_path / "dem.tif", np.full((3, 5), 100.0, np.float32), gdal.GDT_Float32)
    flow_direction = np.zeros((3, 5), np.uint8)
    flow_direction[1, :4] = 2  # Whitebox east
    _write(tmp_path / "flowdir.tif", flow_direction, gdal.GDT_Byte)
    pd.DataFrame({"COMID": 1, "Row": 1, "Col": range(5), "Elev": 100.0, "QBaseflow": 0.0, "Slope": 0.01,
                  "q_1": 10.0, "q_2": 20.0, "t_1": 5.0, "t_2": 10.0, "wse_1": 101.0, "wse_2": 102.0,
                  "v_1": 1.0, "v_2": 2.0, **(vdt_columns or {})}).to_parquet(tmp_path / "vdt.pq")
    pd.DataFrame({"COMID": [1], "q": [15.0]}).to_parquet(tmp_path / "flows.pq")
    pd.DataFrame({"start_pixel": [5], "end_pixel": [9], "length": [5], "stream_id": [1]}).to_parquet(
        tmp_path / "stream_info.pq")
    fsp = np.array([5, 6, 7, 8, 9, 5, 9], np.int32)
    fldpln.library.save_library({"FSP": fsp, "FPP": np.array([5, 6, 7, 8, 9, 0, 14], np.int32),
                         "DTF": np.array([0.1] * 5 + [0.2, 1.8], np.float32), "segment": np.zeros(7, np.int32),
                         "fill": np.zeros(7, np.float32)}, tmp_path / "library.pq", metadata=library_metadata)

    _rerun_fldpln(tmp_path, with_dem, OutDEP=tmp_path / "depth.tif", **inputs)
    return gdal.Open(str(tmp_path / "depth.tif")).ReadAsArray()


def _rerun_fldpln(tmp_path, with_dem=True, **inputs):
    """Map ``_fldpln_depth``'s inputs, which it made, again, with these inputs."""
    Curve2Flood_MainFunction(args={
        "Print_VDT_Database": tmp_path / "vdt.pq",
        "COMID_Flow_File": tmp_path / "flows.pq", "mapper": "Curve2Flood-FLDPLNpy",
        "Flow_Direction_File": tmp_path / "flowdir.tif", "Stream_Info_File": tmp_path / "stream_info.pq",
        "FLDPLN_Library": tmp_path / "library.pq", "Make_Output_GPKG": False,
        **({"DEM_File": tmp_path / "dem.tif"} if with_dem else {}), **inputs,
    }, quiet=True)


def test_the_fldpln_mapper_floods_only_from_its_minimum_depth(tmp_path):
    """FLDPLN_Min_Flood_Depth reaches the mapper: at 1.35 m the cell 1.3 m deep stays dry, and the stream's own cells,
    1.4 m deep, still flood."""
    depth = _fldpln_depth(tmp_path, FLDPLN_Min_Flood_Depth=1.35)
    assert depth[1].tolist() == pytest.approx([1.4] * 5)
    assert (depth == -9999.0).sum() == 10


def test_the_fldpln_mapper_reads_pq_tables_as_parquet(tmp_path):
    """The library's build takes .pq for Parquet, and so does the mapper (see _fldpln_depth)."""
    depth = _fldpln_depth(tmp_path)
    assert depth[1].tolist() == pytest.approx([1.4] * 5)
    assert depth[0, 0] == pytest.approx(1.3)
    assert (depth == -9999.0).sum() == 9


def test_uniform_smoothing_averages_only_the_masked_cells():
    values = np.array([[1.0, 3.0, np.nan]], np.float32)
    mask = np.isfinite(values)
    outputs.uniform_smoothing(mask, values)
    assert values[0].tolist() == pytest.approx([5 / 3, 2.0, 3.0])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_uniform_smoothing_is_scipy_s_uniform_filters_to_the_last_bit(dtype):
    """Over several bands of rows, as it was written with uniform_filter, whose running sums drift."""
    rng = np.random.default_rng(1)
    values = (rng.random((150, 37)) * 1e3).astype(dtype)
    values[rng.random(values.shape) < 0.3] = np.nan
    mask = np.isfinite(values) & (values > 200)
    expected = values.copy()
    value_sum = uniform_filter(np.where(mask, expected, 0), size=3, mode="nearest") * 3
    count = uniform_filter(mask.astype(np.float32), size=3, mode="nearest") * 3
    np.divide(value_sum, count, out=expected, where=count > 0)
    outputs.uniform_smoothing(mask, values)
    assert np.array_equal(values, expected, equal_nan=True)


def test_fldpln_slopes_are_the_nearest_stream_cell_s_averaged_over_3x3():
    """A tie goes to the lower column, as in Flood_Flooded_Cells_in_Map."""
    slopes = np.array([[0.001, np.nan, np.nan, np.nan, 0.004, np.nan, np.nan]], np.float32)
    flood = np.array([[1, 1, 1, 1, 1, 0, 0]], np.uint8)

    spread = outputs.spread_stream_slopes(slopes, flood)

    # nearest: 0.001, 0.001, 0.001, 0.004, 0.004; then each the mean of its row's three (the ends repeated)
    assert spread[0, :5].tolist() == pytest.approx([0.001, 0.001, 0.002, 0.003, 0.004])
    assert np.isnan(spread[0, 6])  # no flooded cell beside it


# --------------------------------------------------------------------------
# Velocity over the floodplain
# --------------------------------------------------------------------------

def test_stream_cell_values_are_set_and_read_as_a_raster_s():
    """A raster's values at its stream cells: set and read there as the raster's would be, a cell set twice keeping
    its last value, the other cells left out."""
    raster = np.full((4, 5), -1.0, np.float32)
    rows, cols = np.array([0, 2, 3]), np.array([4, 1, 1])
    values = core.StreamCellValues(rows * 5 + cols, 5, -1.0)
    for target in (raster, values):
        target[np.array([2, 0, 1, 2]), np.array([1, 4, 1, 1])] = np.array([7.0, 8.0, 9.0, 10.0])
    assert values.values.tolist() == raster[rows, cols].tolist() == [8.0, 10.0, -1.0]
    assert values[2, 1] == raster[2, 1] and values[1, 1] == -1.0  # not a stream cell
    values[:] = np.nan
    assert np.isnan(values.raster(4)).all()


@pytest.mark.parametrize("arrow", [True, False])
def test_flood_polygons_cover_the_flooded_cells(monkeypatch, arrow):
    """The flooded cells' polygons, read at once through Arrow or a feature at a time: together they are those cells,
    each with its value, in the raster's coordinate system; a raster with nothing flooded gives none."""
    import shapely
    from osgeo import ogr
    if not arrow:
        monkeypatch.delattr(ogr.Layer, "__arrow_c_stream__", raising=False)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    rng = np.random.default_rng(8)
    flood = (rng.random((40, 50)) < 0.4).astype(np.uint8) * rng.integers(1, 3, (40, 50), dtype=np.uint8)
    transform = (-99.2, 9.259259259e-05, 0.0, 29.8, 0.0, -9.259259259e-05)
    polygons = outputs.Write_Output_Raster_As_GeoDataFrame(flood, 50, 40, transform, srs.ExportToWkt(), gdal.GDT_Byte)
    assert len(polygons) > 10 and polygons.crs.to_epsg() == 4326 and polygons["Value"].dtype == np.int64
    rows, cols = np.nonzero(flood)
    centres = shapely.points(transform[0] + (cols + 0.5) * transform[1], transform[3] + (rows + 0.5) * transform[5])
    tree = shapely.STRtree(polygons.geometry.values)
    cell, polygon = tree.query(centres, predicate="within")
    assert np.array_equal(cell, np.arange(len(rows)))  # every flooded cell in one polygon
    assert np.array_equal(polygons["Value"].to_numpy()[polygon], flood[rows, cols])
    assert shapely.area(polygons.geometry.values).sum() == pytest.approx(len(rows) * transform[1] ** 2, rel=1e-9)

    dry = outputs.Write_Output_Raster_As_GeoDataFrame(np.zeros_like(flood), 50, 40, transform, srs.ExportToWkt(),
                                                   gdal.GDT_Byte)
    assert len(dry) == 0 and dry.crs.to_epsg() == 4326


def test_gdal_threads_follow_gdal_num_threads(monkeypatch):
    """GDAL_NUM_THREADS, when set, and every CPU otherwise."""
    monkeypatch.delenv("GDAL_NUM_THREADS", raising=False)
    with gdal.config_option("GDAL_NUM_THREADS", None):
        assert _raster.write_threads() == "NUM_THREADS=ALL_CPUS"
    with gdal.config_option("GDAL_NUM_THREADS", "3"):
        assert _raster.write_threads() == "NUM_THREADS=3"


def test_a_vdt_database_has_valley_slopes_with_their_column(tmp_path):
    vdt = pd.DataFrame({"COMID": [1], "Row": [0], "Col": [0], "Slope": [0.001]})
    vdt.to_csv(tmp_path / "plain.csv", index=False)
    vdt.assign(Valley_Slope=0.0015).to_csv(tmp_path / "valley.csv", index=False)
    vdt.assign(Valley_Slope=0.0015).to_parquet(tmp_path / "valley.parquet")
    assert not core.has_valley_slopes(str(tmp_path / "plain.csv"))
    assert core.has_valley_slopes(str(tmp_path / "valley.csv"))
    assert core.has_valley_slopes(str(tmp_path / "valley.parquet"))
    assert not core.has_valley_slopes(None)


def test_the_channels_are_within_half_the_baseflow_top_width_of_a_stream_cell(tmp_path):
    """Reach 1's baseflow, 15 m³/s, first reaches the 20 m³/s step, 40 m wide at two of its cells and 400 m at the
    third, whose rating curve tops its banks: the reach's median, 40 m, puts the cells within 20 m of each of its
    cells in the channel. Reach 2's baseflow is above every flow, so its last top width counts, 8 m: under a 10 m
    cell, so only its stream cell."""
    pd.DataFrame({"COMID": [1, 1, 1, 2], "Row": [5, 5, 5, 0], "Col": [4, 5, 6, 12], "Elev": 100.0,
                  "QBaseflow": [15.0, 15.0, 15.0, 100.0], "Slope": 0.001, "q_1": 10.0, "q_2": 20.0, "q_3": 30.0,
                  "t_1": [20.0, 20.0, 20.0, 4.0], "t_2": [40.0, 400.0, 40.0, 6.0],
                  "t_3": [60.0, 600.0, 60.0, 8.0]}).to_parquet(tmp_path / "vdt.parquet")

    channel = core.channel_cells(str(tmp_path / "vdt.parquet"), 11, 14, 10.0, 10.0)

    rows, cols = np.nonzero(channel)
    near = {(5 + dr, c + dc) for c in (4, 5, 6) for dr in range(-2, 3) for dc in range(-2, 3) if dr * dr + dc * dc <= 4}
    assert {(int(r), int(c)) for r, c in zip(rows, cols)} == near | {(0, 12)}


def _velocity_inputs(tmp_path, n=0.05):
    """A land-cover raster of class 1 for a 21 x 20 grid, and Manning's n of ``n`` for it."""
    _write(tmp_path / "land.tif", np.ones((21, 20), np.uint8), gdal.GDT_Byte)
    (tmp_path / "mannings.txt").write_text(f"LC_ID\tDescription\tManning_n\n1\tGrass\t{n}\n", encoding="utf-8")
    return {"LU_Raster_SameRes": tmp_path / "land.tif", "LU_Manning_n": tmp_path / "mannings.txt"}


def _slope_from_velocity(tmp_path, n=0.05):
    """The slope that Manning's equation gives each flooded cell's velocity at its depth (NaN where dry)."""
    velocity = gdal.Open(str(tmp_path / "velocity.tif")).ReadAsArray().astype(np.float64)
    depth = gdal.Open(str(tmp_path / "depth.tif")).ReadAsArray().astype(np.float64)
    depth = np.where(depth > 0, depth, np.nan)
    return (velocity * n / depth ** (2 / 3)) ** 2


@pytest.mark.parametrize("mapper", ["Curve2Flood-Kernel Weighted", "Curve2Flood-FLDPLNpy"])
def test_the_rasters_written_a_strip_at_a_time_are_the_whole_s(tmp_path, monkeypatch, mapper):
    """Depth, WSE and velocity worked out and written two rows at a time, and the bathymetry burned so, are what one
    strip of every row gives."""
    def run(folder):
        folder.mkdir()
        if mapper == "Curve2Flood-FLDPLNpy":
            _write(folder / "land.tif", np.ones((3, 5), np.uint8), gdal.GDT_Byte)
            (folder / "mannings.txt").write_text("LC_ID\tDescription\tManning_n\n1\tGrass\t0.05\n", encoding="utf-8")
            _fldpln_depth(folder, OutWSE=folder / "wse.tif", OutVEL=folder / "velocity.tif",
                          LU_Raster_SameRes=folder / "land.tif", LU_Manning_n=folder / "mannings.txt")
            names = ["depth", "wse", "velocity"]
        else:
            rows, _ = np.mgrid[0:21, 0:20]
            _write(folder / "dem.tif", (100.0 + 0.5 * np.abs(rows - 10)).astype(np.float32), gdal.GDT_Float32)
            bathy = np.full((21, 20), -9999.0, np.float32)
            bathy[10, 3:17] = 98.0
            _write(folder / "bathy.tif", bathy, gdal.GDT_Float32)
            pd.DataFrame({"COMID": 1, "Row": 10, "Col": range(2, 18), "Elev": 100.0, "QBaseflow": 5.0, "Slope": 0.001,
                          "q_1": 10.0, "q_2": 20.0, "t_1": 70.0, "t_2": 200.0, "wse_1": 101.0, "wse_2": 102.0,
                          "v_1": 1.0, "v_2": 2.0}).to_parquet(folder / "vdt.parquet")
            pd.DataFrame({"COMID": [1], "q": [15.0]}).to_csv(folder / "flows.csv", index=False)
            Curve2Flood_MainFunction(args={
                "DEM_File": folder / "dem.tif", "Print_VDT_Database": folder / "vdt.parquet",
                "COMID_Flow_File": folder / "flows.csv", "OutDEP": folder / "depth.tif", "OutWSE": folder / "wse.tif",
                "OutVEL": folder / "velocity.tif", "OutFLD": folder / "flood.tif", "Make_Output_GPKG": False,
                "BATHY_Out_File": folder / "bathy.tif", "FSOutBATHY": folder / "burned.tif",
                **_velocity_inputs(folder)}, quiet=True)
            names = ["depth", "wse", "velocity", "flood", "burned"]
        return [gdal.Open(str(folder / f"{name}.tif")).ReadAsArray() for name in names]

    whole = run(tmp_path / "whole")
    monkeypatch.setattr(_raster, "_STRIP_ROWS", 2)
    for strips, expected in zip(run(tmp_path / "strips"), whole):
        assert np.array_equal(strips, expected, equal_nan=True)


@pytest.mark.parametrize("valley_slopes", [True, False])
def test_the_kernel_s_velocity_takes_the_valley_s_slope_outside_the_channel(tmp_path, valley_slopes):
    """A stream along row 10 of a valley whose sides rise 0.5 m a 30 m cell, its water 1.5 m deep, so rows 8 to 12
    flood: the baseflow top width, 70 m, makes rows 9 to 11 the channel, at the stream's slope (0.001), and rows 8
    and 12 take the valley's (0.004). Without the VDT's Valley_Slope, every cell takes the stream's."""
    rows, _ = np.mgrid[0:21, 0:20]
    _write(tmp_path / "dem.tif", (100.0 + 0.5 * np.abs(rows - 10)).astype(np.float32), gdal.GDT_Float32)
    vdt = pd.DataFrame({"COMID": 1, "Row": 10, "Col": range(2, 18), "Elev": 100.0, "QBaseflow": 5.0, "Slope": 0.001,
                        "q_1": 10.0, "q_2": 20.0, "t_1": 70.0, "t_2": 200.0, "wse_1": 101.0, "wse_2": 102.0,
                        "v_1": 1.0, "v_2": 2.0})
    if valley_slopes:
        vdt["Valley_Slope"] = 0.004
    vdt.to_parquet(tmp_path / "vdt.parquet")
    pd.DataFrame({"COMID": [1], "q": [15.0]}).to_csv(tmp_path / "flows.csv", index=False)

    Curve2Flood_MainFunction(args={
        "DEM_File": tmp_path / "dem.tif", "Print_VDT_Database": tmp_path / "vdt.parquet",
        "COMID_Flow_File": tmp_path / "flows.csv", "TW_MultFact": 1.0, "OutDEP": tmp_path / "depth.tif",
        "OutVEL": tmp_path / "velocity.tif", "Make_Output_GPKG": False, **_velocity_inputs(tmp_path)}, quiet=True)

    slope = _slope_from_velocity(tmp_path)
    middle = slice(4, 16)  # clear of the stream's ends
    assert np.isfinite(slope[[8, 12], middle]).all() and np.isnan(slope[[7, 13], middle]).all()
    np.testing.assert_allclose(slope[9:12, middle], 0.001, rtol=1e-3)
    np.testing.assert_allclose(slope[[8, 12], middle], 0.004 if valley_slopes else 0.001, rtol=1e-3)


@pytest.mark.parametrize("valley_slopes", [True, False])
def test_the_fldpln_velocity_takes_the_valley_s_slope_outside_the_channel(tmp_path, valley_slopes):
    """_fldpln_depth's stream, whose baseflow top width (5 m) is under a cell: its own cells are the channel, at the
    stream's slope (0.01), and the cell it floods above it takes the valley's (0.02)."""
    _write(tmp_path / "land.tif", np.ones((3, 5), np.uint8), gdal.GDT_Byte)
    (tmp_path / "mannings.txt").write_text("LC_ID\tDescription\tManning_n\n1\tGrass\t0.05\n", encoding="utf-8")
    _fldpln_depth(tmp_path, vdt_columns={"Valley_Slope": 0.02} if valley_slopes else {},
                  OutVEL=tmp_path / "velocity.tif", LU_Raster_SameRes=tmp_path / "land.tif",
                  LU_Manning_n=tmp_path / "mannings.txt")

    slope = _slope_from_velocity(tmp_path)
    np.testing.assert_allclose(slope[1], 0.01, rtol=1e-3)
    assert slope[0, 0] == pytest.approx(0.02 if valley_slopes else 0.01, rel=1e-3)


@pytest.mark.parametrize("use_banks, expected", [(False, [8, 10, 10, -120]), (True, [8, 10, 12, -120])])
def test_topobathy_is_arc_s_bathymetry_where_it_has_one_and_the_dem_elsewhere(use_banks, expected):
    E = np.array([10.0, 10.0, 10.0, 10.0], np.float32)
    bathy = np.array([8.0, np.nan, 12.0, -120.0], np.float32)  # a gap, a bed above the DEM, and one far below 0
    assert bathymetry.Create_Topobathy_Dataset(E, bathy, use_banks).tolist() == pytest.approx(expected)


def test_the_bathymetry_is_burned_in_as_arc_wrote_it_whatever_the_water_mask(tmp_path):
    """ARC's bed goes into the DEM wherever ARC carved, in the water mask or not, and the cells ARC left without
    bathymetry (no data, or NaN) keep the DEM, whatever the mask says. A run making only the bathymetry needs no
    flows."""
    dem = np.full((5, 6), 100.0, np.float32)
    bathy = np.full(dem.shape, -9999.0, np.float32)
    bathy[2, 1:5] = [97.0, 96.0, 96.5, 98.0]
    bathy[1, 2] = np.nan
    mask = np.zeros(dem.shape, np.uint8)
    mask[1:4, 2:4] = 1  # misses two of ARC's cells, and has cells ARC didn't carve
    _write(tmp_path / "dem.tif", dem, gdal.GDT_Float32)
    _write(tmp_path / "bathy.tif", bathy, gdal.GDT_Float32)
    ds = gdal.Open(str(tmp_path / "bathy.tif"), gdal.GA_Update)
    ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    ds = None
    _write(tmp_path / "mask.tif", mask, gdal.GDT_Byte)
    Curve2Flood_MainFunction(args={
        "DEM_File": str(tmp_path / "dem.tif"), "BATHY_Out_File": str(tmp_path / "bathy.tif"),
        "FSOutBATHY": str(tmp_path / "burned.tif"), "BathyWaterMask": str(tmp_path / "mask.tif"),
        "Bathy_Use_Banks": True,
    }, quiet=True)
    burned = gdal.Open(str(tmp_path / "burned.tif")).ReadAsArray()
    expected = dem.copy()
    expected[2, 1:5] = [97.0, 96.0, 96.5, 98.0]
    assert burned.tolist() == expected.tolist()


def test_the_water_surface_is_the_depth_over_the_dem():
    depth = np.array([[1.0, 0.0, np.nan], [2.0, 0.5, 1.0]], dtype=np.float32)
    dem = np.array([[10.0, 10.0, 10.0], [12.0, -9999.0, 11.0]], dtype=np.float32)
    wse = outputs.water_surface(depth, dem)
    np.testing.assert_allclose(wse, [[11.0, np.nan, np.nan], [14.0, np.nan, 12.0]])


def test_the_fldpln_water_surface_is_measured_from_below_raised_bathymetry():
    """With the FLDPLN mapper, the depth over a cell the bathymetry raised above the filled DEM is measured from the
    filled DEM, and so is its water surface; below the filled DEM (a pit or a burned channel) it is the DEM's, and so
    where the filled DEM has no value."""
    depth = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    dem = np.array([[103.0, 98.0, 100.0]], dtype=np.float32)       # raised 3 m, burned 2 m, not in the library
    below = np.array([[-3.0, 0.0, np.nan]], dtype=np.float32)      # min(fill, 0), as the mapper writes it
    np.testing.assert_allclose(outputs.water_surface(depth, dem, below), [[101.0, 99.0, 101.0]])


# --------------------------------------------------------------------------
# Flow events, stream cells and the grid
# --------------------------------------------------------------------------

def test_each_cell_floods_in_the_percentage_of_the_events_that_flood_it(tmp_path):
    """Three events: at 15 m3/s the cell above the stream floods, at 1.5 m3/s only the stream's own cells do (0.15 m
    of water over their DTF of 0.1 m, and short of that cell's 0.2 m), so it floods in a third of them, rounded
    down. The depth is the deepest of the events'."""
    _fldpln_depth(tmp_path)
    pd.DataFrame({"COMID": [1], "q_a": [15.0], "q_b": [1.5], "q_c": [1.5]}).to_parquet(tmp_path / "flows.pq")
    _rerun_fldpln(tmp_path, OutFLD=tmp_path / "flood.tif", OutDEP=tmp_path / "depth.tif")
    flood = gdal.Open(str(tmp_path / "flood.tif")).ReadAsArray()
    assert flood[1].tolist() == [100] * 5
    assert flood[0, 0] == 33 and flood.sum() == 5 * 100 + 33
    assert gdal.Open(str(tmp_path / "depth.tif")).ReadAsArray()[0, 0] == pytest.approx(1.3)


def test_a_flow_file_has_at_most_655_events_and_no_negative_flow(tmp_path):
    path = tmp_path / "flows.csv"
    pd.DataFrame(np.ones((2, 657)), columns=["COMID"] + [f"q{i}" for i in range(656)]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="656 flow events; a run maps at most 655"):
        core.check_flow_file(str(path))
    pd.DataFrame({"COMID": [1, 2], "q1": [5.0, 1.0], "q2": [3.0, -1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match=r"negative flows in event\(s\) q2"):
        core.check_flow_file(str(path))
    pd.DataFrame({"COMID": [1, 2], "q1": [5.0, 1.0], "q2": [3.0, np.nan]}).to_csv(path, index=False)
    assert core.check_flow_file(str(path)) == 2
    pd.DataFrame({"COMID": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="has no flow events"):
        core.check_flow_file(str(path))


def test_reaches_without_a_flow_are_named(tmp_path, caplog):
    path = tmp_path / "flows.csv"
    pd.DataFrame({"COMID": [1, 3], "q": [1.0, 2.0]}).to_csv(path, index=False)
    with caplog.at_level(logging.WARNING, logger=LOG.name):
        core._warn_reaches_without_flows(str(path), np.array([1, 2], np.int64))
    assert "1 of the 2 reaches with stream cells have no flow" in caplog.text and "(such as 2)" in caplog.text


def test_stream_cells_must_lie_on_the_dem(tmp_path):
    path = tmp_path / "vdt.csv"
    pd.DataFrame({"COMID": [7, 8], "Row": [1, 3], "Col": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="1 of the 2 cells .* lie outside the DEM's 3 by 2 cells, the first at row 3"):
        core.Set_Stream_Locations(3, 2, str(path))
    # Counted from 1, they lie on it
    assert core.Set_Stream_Locations(3, 2, str(path), one_based=True).tolist() == [[7, -9999], [-9999, -9999],
                                                                                   [-9999, 8]]


def test_stream_ids_that_do_not_fit_in_int32_are_kept_whole(tmp_path):
    path = tmp_path / "vdt.csv"
    big = 55000400012345  # an NHDPlus HR id
    pd.DataFrame({"COMID": [7, big], "Row": [0, 1], "Col": [0, 0]}).to_csv(path, index=False)
    streams = core.Set_Stream_Locations(2, 1, str(path))
    assert streams.dtype == np.int64 and streams[1, 0] == big
    pd.DataFrame({"COMID": [7, 2**31 - 1], "Row": [0, 1], "Col": [0, 0]}).to_csv(path, index=False)
    assert core.Set_Stream_Locations(2, 1, str(path)).dtype == np.int32


@pytest.mark.parametrize(("ids", "dtype"), [([1, 2**31 - 1], np.int32), ([-2**31, 5], np.int32), ([2**31], np.int64),
                                            ([], np.int32), (np.array([200], np.uint8), np.int32),
                                            (np.array([5.0, 2.0**40]), np.int64)])
def test_ids_are_int32_when_they_fit(ids, dtype):
    assert _ids.id_dtype(np.asarray(ids)) == dtype


def test_stream_cells_are_the_stream_raster_s_positive_cells(tmp_path):
    """The cells of Set_Stream_Locations' raster with an id above 0, without the raster, ascending: a cell listed
    twice keeps its last row's id, and the ids have the raster's type."""
    path = tmp_path / "vdt.csv"
    big = 55000400012345  # an NHDPlus HR id
    pd.DataFrame({"COMID": [7, 8, 0, -3, 9, big], "Row": [2, 0, 1, 1, 2, 0],
                  "Col": [1, 0, 1, 0, 1, 2]}).to_csv(path, index=False)
    raster = core.Set_Stream_Locations(3, 3, str(path)).ravel()
    cells, ids = core._stream_cells(3, 3, str(path))
    assert cells.tolist() == [0, 2, 7] and ids.tolist() == [8, big, 9]  # (2, 1) listed twice: 9 replaced 7
    assert np.array_equal(cells, np.flatnonzero(raster > 0)) and np.array_equal(ids, raster[raster > 0])
    assert ids.dtype == raster.dtype == np.int64


@pytest.mark.parametrize("num_flows", [1, 3, core.MAX_FLOW_EVENTS])
def test_percent_of_events_rounds_down(num_flows):
    rng = np.random.default_rng(1)
    counts = rng.integers(0, num_flows + 1, (5, 7)).astype(np.uint8 if num_flows == 1 else np.uint16)
    counts[0, :2] = (0, num_flows)
    expected = counts.astype(np.int64) * 100 // num_flows
    percent = core._percent_of_events(counts, num_flows)
    assert percent.dtype == np.uint8 and np.array_equal(percent, expected) and percent[0, 1] == 100


def test_flows_with_decimals_after_whole_numbers_are_floats(tmp_path):
    """A CSV flow file's columns are typed from all of their rows, as pandas typed them: a flow with decimals after
    hundreds of whole ones is read, not refused."""
    path = tmp_path / "flows.csv"
    path.write_text("COMID,q\n" + "".join(f"{i},5\n" for i in range(1, 201)) + "201,7.5\n")
    assert core.check_flow_file(str(path)) == 1
    flows = core.FindFlowRateForEachCOMID_Ensemble(str(path), 0)
    assert flows[1] == 5.0 and flows[201] == 7.5


def test_mapping_with_a_parquet_library_imports_no_pandas(tmp_path):
    """The driver imports neither pandas, SciPy's image tools nor GeoPandas, and reading a Parquet library imports
    no pandas: only the multi-point mapper, curve files, velocity and the GeoPackage need them, and pandas alone
    took a third of a second to import."""
    path = tmp_path / "library.parquet"
    from curve2flood.spreaders.fldpln.library import save_library
    save_library({"FSP": np.array([5, 5], np.int32), "FPP": np.array([5, 0], np.int32),
                  "DTF": np.array([0.0, 0.25], np.float32), "segment": np.array([0, 0], np.int32),
                  "fill": np.array([0.0, 0.5], np.float32)}, path)
    code = ("import sys\n"
            "import curve2flood.core\n"
            "from curve2flood.spreaders.fldpln import library_sources\n"
            f"sources = library_sources({str(path)!r}, 6)\n"
            "assert sources.fill_cells.tolist() == [0]\n"
            "print(sorted(m for m in ('pandas', 'scipy.ndimage', 'geopandas', 'shapely') if m in sys.modules))\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


def test_an_fldpln_flood_map_takes_its_grid_from_its_library(tmp_path):
    """Without a DEM the library's metadata places the flood map, and a library made for other ground is refused."""
    metadata = {"version": 1, "nrows": 3, "ncols": 5, "geotransform": list(_GEOTRANSFORM), "crs": _SRS.ExportToWkt(),
                "cell_size_m": [30.0, 30.0]}
    depth = _fldpln_depth(tmp_path, library_metadata=metadata, with_dem=False)
    assert depth[1].tolist() == pytest.approx([1.4] * 5)
    assert gdal.Open(str(tmp_path / "depth.tif")).GetGeoTransform() == _GEOTRANSFORM

    shifted = {**metadata, "geotransform": [500030.0, *_GEOTRANSFORM[1:]]}
    with pytest.raises(ValueError, match="cover different ground"):
        _fldpln_depth(tmp_path, library_metadata=shifted, with_dem=False)


def test_an_fldpln_library_without_metadata_takes_the_flow_directions_grid(tmp_path):
    depth = _fldpln_depth(tmp_path, with_dem=False)
    assert depth[1].tolist() == pytest.approx([1.4] * 5)
    assert gdal.Open(str(tmp_path / "depth.tif")).GetGeoTransform() == _GEOTRANSFORM


def test_land_cover_the_manning_table_does_not_list_takes_0_035(tmp_path, caplog):
    path = tmp_path / "manning.txt"
    path.write_text("LC_ID\tDescription\tManning_n\n10\tTrees\t0.12\n80\tWater\t0.03\n")
    n = outputs.DEFAULT_MANNING_N
    assert n == 0.035
    with caplog.at_level(logging.WARNING, logger=LOG.name):
        # 20 isn't in the table, 255 is the no-data of a byte raster
        lookup = _manning_n(path, np.array([[10, 20], [255, 80]], np.uint8))
    assert lookup.tolist() == [[np.float32(0.12), np.float32(n)], [np.float32(n), np.float32(0.03)]]
    assert "(20, 255)" in caplog.text
    assert _manning_n(path, np.array([[-9999, 80]], np.int32)).tolist() == [
        [np.float32(n), np.float32(0.03)]]
    assert _manning_n(path, np.array([[np.nan, 10.0]], np.float32)).tolist() == [
        [np.float32(n), np.float32(0.12)]]


def _valley(tmp_path, comid=1, flows=None, **vdt_columns):
    """A valley whose stream runs east along row 10 of a 21 x 20 grid of 30 m cells, cells 2 to 17, its sides
    rising 0.5 m a cell, and the inputs to map it: at 14 m3/s the VDT gives a water surface of 101.4 m (clear of
    the ground's contours) and a top width of 122 m, so rows 8 to 12 flood. ``flows`` are the flow file's, by
    reach."""
    rows, _ = np.mgrid[0:21, 0:20]
    _write(tmp_path / "dem.tif", (100.0 + 0.5 * np.abs(rows - 10)).astype(np.float32), gdal.GDT_Float32)
    vdt = pd.DataFrame({"COMID": comid, "Row": 10, "Col": range(2, 18), "Elev": 100.0, "QBaseflow": 5.0,
                        "Slope": 0.001, "XS_Angle": np.pi / 2, "q_1": 10.0, "q_2": 20.0, "t_1": 70.0, "t_2": 200.0,
                        "wse_1": 101.0, "wse_2": 102.0, "v_1": 1.0, "v_2": 2.0, **vdt_columns})
    vdt.to_parquet(tmp_path / "vdt.parquet")
    flows = flows or {comid: 14.0}
    pd.DataFrame({"COMID": list(flows), "q": list(flows.values())}).to_csv(tmp_path / "flows.csv", index=False)
    return {"DEM_File": tmp_path / "dem.tif", "Print_VDT_Database": tmp_path / "vdt.parquet",
            "COMID_Flow_File": tmp_path / "flows.csv", "Make_Output_GPKG": False, "TW_MultFact": 1.0}


def _read(path):
    return gdal.Open(str(path)).ReadAsArray()


@pytest.mark.parametrize("mapper", ["Curve2Flood-Kernel Weighted", "Curve2Flood-Multi-Point Interpolation"])
def test_a_run_keeps_reach_ids_that_do_not_fit_in_int32(tmp_path, mapper):
    inputs = _valley(tmp_path, comid=55000400012345)
    Curve2Flood_MainFunction(args={**inputs, "mapper": mapper, "OutFLD": tmp_path / "flood.tif"}, quiet=False)
    flood = _read(tmp_path / "flood.tif")
    assert (flood[8:13, 4:16] == 100).all() and not flood[:7].any() and not flood[14:].any()


def test_the_multi_point_mapper_floods_the_valley(tmp_path):
    """Its points carry the VDT's water surface across the corridor (one top width around them), and every cell
    below it there floods, as deep as the water is over the ground."""
    inputs = _valley(tmp_path)
    Curve2Flood_MainFunction(args={**inputs, "mapper": "Curve2Flood-Multi-Point Interpolation",
                                   "OutFLD": tmp_path / "flood.tif", "OutDEP": tmp_path / "depth.tif"}, quiet=True)
    flood, depth = _read(tmp_path / "flood.tif"), _read(tmp_path / "depth.tif")
    rows = np.arange(21)
    assert (flood[8:13, 2:18] == 100).all() and not flood[(rows < 8) | (rows > 12)].any()
    np.testing.assert_allclose(depth[8:13, 5], [0.4, 0.9, 1.4, 0.9, 0.4], atol=1e-4)


@pytest.mark.parametrize("curves", ["vdt", "curve file"])
def test_the_multi_point_mapper_ignores_tw_multfact(tmp_path, curves):
    """TW_MultFact is the kernel-weighted mapper's: the multi-point mapper sizes its corridor from the rating curves'
    own top widths, with its MPI factors. At 0.1 its corridor would reach no cell but the stream's own."""
    inputs = _valley(tmp_path)
    if curves == "curve file":
        _constant_curves(tmp_path / "curves.csv")
        inputs = {**{k: v for k, v in inputs.items() if k != "Print_VDT_Database"},
                  "Print_Curve_File": tmp_path / "curves.csv"}
    floods = []
    for factor in (0.1, 1.0, 6.0):
        Curve2Flood_MainFunction(args={**inputs, "TW_MultFact": factor, "mapper": "Curve2Flood-Multi-Point Interpolation",
                                       "OutFLD": tmp_path / f"flood_{factor}.tif"}, quiet=True)
        floods.append(_read(tmp_path / f"flood_{factor}.tif"))
    assert (floods[0][8:13, 4:16] == 100).all()
    assert all(np.array_equal(flood, floods[0]) for flood in floods[1:])


def _constant_curves(path):
    """``_valley``'s rating curves as a curve file of power laws, constant: 1.4 m deep and 122 m wide at any flow."""
    pd.DataFrame({"COMID": 1, "Row": 10, "Col": range(2, 18), "BaseElev": 100.0, "QMax": 30.0, "Slope": 0.001,
                  "XS_Angle": np.pi / 2, "depth_a": 1.4, "depth_b": 0.0, "tw_a": 122.0, "tw_b": 0.0, "vel_a": 1.0,
                  "vel_b": 0.0}).to_csv(path, index=False)


def test_the_multi_point_mapper_never_floods_no_data(tmp_path):
    inputs = _valley(tmp_path)
    dem = _read(tmp_path / "dem.tif")
    dem[9, 5] = -9999.0
    _write(tmp_path / "dem.tif", dem, gdal.GDT_Float32)
    Curve2Flood_MainFunction(args={**inputs, "mapper": "Curve2Flood-Multi-Point Interpolation",
                                   "OutFLD": tmp_path / "flood.tif", "OutDEP": tmp_path / "depth.tif"}, quiet=True)
    flood, depth = _read(tmp_path / "flood.tif"), _read(tmp_path / "depth.tif")
    assert flood[9, 5] == 0 and flood[9, 6] == 100
    assert depth.max() < 2.0


def test_the_multi_point_mapper_floods_the_stream_cells_of_reaches_without_flow(tmp_path):
    """As the other mappers do: a second reach up the valley side, with no flow, floods its stream cells alone."""
    inputs = _valley(tmp_path, flows={1: 14.0, 2: 0.0})
    vdt = pd.read_parquet(tmp_path / "vdt.parquet")
    pd.concat([vdt, vdt.assign(COMID=2, Row=3, Elev=103.5)], ignore_index=True).to_parquet(tmp_path / "vdt.parquet")
    Curve2Flood_MainFunction(args={**inputs, "mapper": "Curve2Flood-Multi-Point Interpolation",
                                   "OutFLD": tmp_path / "flood.tif"}, quiet=True)
    flood = _read(tmp_path / "flood.tif")
    assert (flood[3, 2:18] == 100).all() and not flood[[2, 4]].any()


def test_reaches_without_a_flow_above_0_have_no_points(tmp_path):
    _valley(tmp_path)
    scenario = multi_point.build_fhs_scenario_dataframe_from_vdt(str(tmp_path / "vdt.parquet"), {1: 0.0})
    assert scenario.empty
    scenario = multi_point.build_fhs_scenario_dataframe_from_vdt(str(tmp_path / "vdt.parquet"), {1: 15.0})
    assert scenario["TopWidth_m"].tolist() == pytest.approx([135.0] * 16)
    assert scenario["WaterSurfaceElev_m"].tolist() == pytest.approx([101.5] * 16)


@pytest.mark.parametrize("mapper", ["Curve2Flood-Kernel Weighted", "Curve2Flood-Multi-Point Interpolation"])
def test_a_curve_file_maps_as_the_vdt_does(tmp_path, mapper):
    """The curve file's power laws, constant here: 1.4 m deep and 122 m wide at any flow, over a 100 m bed."""
    inputs = _valley(tmp_path)
    _constant_curves(tmp_path / "curves.csv")
    del inputs["Print_VDT_Database"]
    Curve2Flood_MainFunction(args={**inputs, "Print_Curve_File": tmp_path / "curves.csv", "mapper": mapper,
                                   "OutFLD": tmp_path / "flood.tif"}, quiet=True)
    flood = _read(tmp_path / "flood.tif")
    assert (flood[8:13, 4:16] == 100).all() and not flood[:7].any() and not flood[14:].any()


@pytest.fixture
def log_level():
    """Put the logger's level back after a test that sets it."""
    levels = LOG.level, LOG.handlers[0].level
    yield
    LOG.setLevel(levels[0])
    LOG.handlers[0].setLevel(levels[1])


def test_the_command_line_maps_an_input_file(tmp_path, monkeypatch, log_level):
    inputs = _valley(tmp_path)
    config = tmp_path / "input.yaml"
    config.write_text("".join(f"{key}: {value}\n" for key, value in
                              {**inputs, "OutFLD": tmp_path / "flood.tif"}.items()), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["curve2flood", str(config), "-l", "error"])
    cli.main()
    assert (_read(tmp_path / "flood.tif")[8:13, 4:16] == 100).all()
    assert LOG.level == logging.ERROR
