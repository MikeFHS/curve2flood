"""Tests of Config: reading, checking and typing Curve2Flood's inputs."""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pytest

from curve2flood import Config
from curve2flood.config import FLDPLN, MULTI_POINT, is_parquet

BASE_INPUTS = {"DEM_File": "dem.tif"}
# A flood map, which needs flows and rating curves
FLOOD_MAP = {**BASE_INPUTS, "OutFLD": "flood.tif"}
CURVES = {"COMID_Flow_File": "q.csv", "Print_VDT_Database": "vdt.parquet"}


def _config(**inputs) -> Config:
    return Config.from_mapping({**FLOOD_MAP, **inputs})


def test_text_yaml_and_mapping_inputs_build_equal_configs(tmp_path: Path) -> None:
    """All three input styles parse to the same, correctly typed values."""
    text_file = tmp_path / "input.txt"
    text_file.write_text(
        "# a comment\n"
        "DEM_File\tinputs/my dem.tif\n"
        "COMID_Flow_File path/to/q.csv\n"
        "\n"
        "TopWidthPlausibleLimit\t200\n"
        "LAND_WaterValue\t11\n"
        "LocalFloodOption\tTrue\n"
        "Make_Output_GPKG\tfalse\n"
        "mapper\tCurve2Flood-Multi-Point Interpolation\n",
        encoding="utf-8",
    )
    yaml_file = tmp_path / "input.yaml"
    yaml_file.write_text(
        "DEM_File: inputs/my dem.tif\n"
        "COMID_Flow_File: path/to/q.csv\n"
        "TopWidthPlausibleLimit: 200\n"
        "LAND_WaterValue: 11\n"
        "LocalFloodOption: true\n"
        "Make_Output_GPKG: false\n"
        "mapper: Curve2Flood-Multi-Point Interpolation\n",
        encoding="utf-8",
    )
    mapping = {
        "dem_file": "inputs/my dem.tif",
        "comid_flow_file": "path/to/q.csv",
        "topwidthplausiblelimit": 200,
        "land_watervalue": 11,
        "localfloodoption": True,
        "make_output_gpkg": False,
        "mapper": MULTI_POINT,
    }

    config = Config.from_file(text_file)

    assert config == Config.from_file(yaml_file) == Config.from_mapping(mapping)
    assert config.dem_file == "inputs/my dem.tif"  # a value may contain spaces
    assert config.topwidthplausiblelimit == 200.0 and isinstance(config.topwidthplausiblelimit, float)
    assert config.land_watervalue == 11 and isinstance(config.land_watervalue, int)
    assert config.localfloodoption is True
    assert config.make_output_gpkg is False
    assert config.mapper == MULTI_POINT
    # Missing keys take their defaults
    assert config.outdep is None
    assert config.tw_multfact == 3.0
    assert config.set_depth == -1.1


def test_keys_are_case_insensitive() -> None:
    config = Config.from_mapping({"dem_file": "dem.tif", "Comid_Flow_File": "q.csv", "TW_MULTFACT": "2.5"})

    assert config.comid_flow_file == "q.csv"
    assert config.tw_multfact == 2.5


def test_the_stream_network_s_id_fields_are_arc_s_and_rejected() -> None:
    """No mapper reads the stream network's reach ID fields any more: they are ARC's alone."""
    with pytest.raises(ValueError, match="Unrecognized config key"):
        Config.from_mapping({**BASE_INPUTS, "reach_id": "COMID", "Downstream_Reach_ID": "NextDownID"})


def test_paths_may_be_path_objects() -> None:
    path = Path("inputs") / "dem.tif"

    assert Config.from_mapping({"DEM_File": path}).dem_file == str(path)


def test_a_key_set_to_nothing_takes_its_default(tmp_path: Path) -> None:
    text_file = tmp_path / "input.txt"
    text_file.write_text("DEM_File dem.tif\nTW_MultFact\nOutDEP\n", encoding="utf-8")
    yaml_file = tmp_path / "input.yaml"
    yaml_file.write_text("DEM_File: dem.tif\nTW_MultFact: null\nOutDEP: null\nMake_Output_GPKG:\n", encoding="utf-8")

    assert Config.from_file(text_file) == Config.from_file(yaml_file) == Config.from_mapping(BASE_INPUTS)
    assert Config.from_file(yaml_file).make_output_gpkg is True


@pytest.mark.parametrize("depths, expected", [
    ({}, -1.1),
    ({"Set_Depth": "0.3"}, 0.3),
    ({"FloodSpreader_SpecifyDepth": "0.2"}, 0.2),
    ({"Set_Depth": "0.3", "FloodSpreader_SpecifyDepth": "0.2"}, 0.3),
    ({"Set_Depth": "", "FloodSpreader_SpecifyDepth": "0.2"}, 0.2),
])
def test_set_depth_comes_from_either_key(depths: dict, expected: float) -> None:
    assert Config.from_mapping({**BASE_INPUTS, **depths}).set_depth == expected


@pytest.mark.parametrize("flags, expected", [
    ({}, False),
    ({"FloodLocalOnly": "True"}, True),
    ({"LocalFloodOption": "True", "FloodLocalOnly": "False"}, True),
    ({"LocalFloodOption": "False", "FloodLocalOnly": "True"}, True),
    ({"LocalFloodOption": "False", "FloodLocalOnly": "False"}, False),
])
def test_either_local_flood_key_turns_it_on(flags: dict, expected: bool) -> None:
    assert Config.from_mapping({**BASE_INPUTS, **flags}).localfloodoption is expected


def test_aroutbathy_is_bathy_out_file_and_wins() -> None:
    """ARC's name for the bathymetry it writes, which wins as it does in ARC."""
    assert Config.from_mapping({**BASE_INPUTS, "AROutBATHY": "arc.tif"}).bathy_out_file == "arc.tif"
    both = {**BASE_INPUTS, "AROutBATHY": "arc.tif", "BATHY_Out_File": "other.tif"}
    assert Config.from_mapping(both).bathy_out_file == "arc.tif"


@pytest.mark.parametrize("value, expected", [
    ("False", False), ("false", False), ("no", False), ("0", False), (False, False), (0, False),
    ("True", True), ("TRUE", True), ("yes", True), ("1", True), (True, True), (1, True),
])
def test_switches(value, expected: bool) -> None:
    assert Config.from_mapping({**BASE_INPUTS, "Bathy_Use_Banks": value}).bathy_use_banks is expected


def test_choices_match_whatever_their_case() -> None:
    config = Config.from_mapping({**BASE_INPUTS, "mapper": "curve2flood-fldplnpy", "MPI_HWM_Outlier_Method": "IQR",
                                  "MPI_Connectivity": "8"})

    assert config.mapper == FLDPLN
    assert config.mpi_hwm_outlier_method == "iqr"
    assert config.mpi_connectivity == 8


def test_floodspreader_s_unused_keys_are_accepted_and_left_out() -> None:
    """FloodSpreader's input files still run."""
    inputs = {**BASE_INPUTS, "OutSHP": "flood.shp", "FS_ADJUST_FLOW_BY_FRACTION": 1.0}

    assert Config.from_mapping(inputs) == Config.from_mapping(BASE_INPUTS)


@pytest.mark.parametrize("key", ["Flow_File", "Flow_File_BF", "X_Section_Dist", "Stream_Slope_Method", "Bathy_Trap_H",
                                 "Spatial_Units", "Str_Limit_Val", "Standard_Step", "Print_AP_Database"])
def test_arc_s_keys_are_rejected(key: str) -> None:
    """An input file is Curve2Flood's alone: ARC gets its own."""
    with pytest.raises(ValueError, match="Unrecognized config key"):
        Config.from_mapping({**BASE_INPUTS, key: 1})


def test_the_keys_both_models_read_are_curve2flood_s_too() -> None:
    """ARC and Curve2Flood share a few key names (each reads them for itself)."""
    config = Config.from_mapping({**BASE_INPUTS, "Bathy_Use_Banks": True, "BathyWaterMask": "mask.tif",
                                  "Print_VDT_Database": "vdt.parquet", "Compression": "ZSTD"})
    assert config.bathy_use_banks is True and config.compression == "ZSTD"


def test_check_keys_checks_names_and_types_without_a_run() -> None:
    # A run needs DEM_File, but the keys given are checked alone
    assert Config.check_keys({"TW_MultFact": "2", "FloodSpreader_SpecifyDepth": "0.3", "OutDEP": None}) == \
        {"tw_multfact": 2.0, "set_depth": 0.3}
    with pytest.raises(ValueError, match=r"'standard_step'"):
        Config.check_keys({"Standard_Step": True})
    with pytest.raises(ValueError, match="TW_MultFact|tw_multfact"):
        Config.check_keys({"TW_MultFact": "wide"})


def test_unrecognized_key_is_rejected_with_a_suggestion() -> None:
    with pytest.raises(ValueError, match=r"'tw_multfactor' \(did you mean 'tw_multfact'\?\)"):
        Config.from_mapping({**BASE_INPUTS, "TW_MultFactor": 1.5})


@pytest.mark.parametrize("key", ["max_wse_rise", "FLDPLN_Median_Filter_Size", "FLDPLN_Max_Drop_Below_Source",
                                 "Filled_DEM_File"])
def test_removed_keys_are_ignored_with_a_warning(key: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="curve2flood._log"):
        config = Config.from_mapping({**BASE_INPUTS, key: 1})

    assert config == Config.from_mapping(BASE_INPUTS)
    assert f"{key.lower()} is no longer used" in caplog.text


def test_the_multi_point_options_drop_their_prefix() -> None:
    options = Config.from_mapping({**BASE_INPUTS, "MPI_K": "20", "MPI_Fast_Mode": "True"}).multi_point_options

    assert len(options) == 26
    assert options["k"] == 20 and options["fast_mode"] is True
    assert options["hwm_outlier_method"] == "mad"
    assert "dem_file" not in options


def test_derived_options() -> None:
    config = Config.from_mapping(BASE_INPUTS)
    assert not config.uses_set_depth and not config.makes_bathymetry and not config.needs_flood_map

    assert Config.from_mapping({**BASE_INPUTS, "Set_Depth": 0.5, "Stream_File": "s.tif"}).uses_set_depth
    assert _config(**CURVES).needs_flood_map
    bathymetry = Config.from_mapping({**BASE_INPUTS, **CURVES, "BATHY_Out_File": "arc.tif", "FSOutBATHY": "fs.tif"})
    assert bathymetry.makes_bathymetry
    # The bathymetry is ARC's raster burned in as it is, so it needs no flood map, with a water mask or without
    assert not bathymetry.needs_flood_map


def test_the_config_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        Config.from_mapping(BASE_INPUTS).tw_multfact = 2.0


def test_without_outputs_only_the_dem_is_needed() -> None:
    Config.from_mapping(BASE_INPUTS)
    with pytest.raises(ValueError, match="Missing required parameters: DEM_File$"):
        Config.from_mapping({})


def test_flood_bathy_mask_needs_the_mask(tmp_path: Path) -> None:
    assert Config.from_mapping(BASE_INPUTS).flood_bathy_mask is False
    settings = dict(**CURVES, Flood_Bathy_Mask="True")
    with pytest.raises(ValueError, match="Missing required parameters: BathyWaterMask$"):
        _config(**settings)
    with pytest.raises(ValueError, match="doesn't exist"):
        _config(BathyWaterMask=str(tmp_path / "missing.tif"), **settings)
    mask = tmp_path / "mask.tif"
    mask.write_bytes(b"")
    _config(BathyWaterMask=str(mask), **settings)


def test_fldpln_floods_from_1_cm_unless_told_otherwise() -> None:
    assert Config.from_mapping(BASE_INPUTS).fldpln_min_flood_depth == 0.01
    assert Config.from_mapping({**BASE_INPUTS, "FLDPLN_Min_Flood_Depth": "0"}).fldpln_min_flood_depth == 0.0
    assert Config.from_mapping({**BASE_INPUTS, "fldpln_min_flood_depth": 0.05}).fldpln_min_flood_depth == 0.05


def test_fldpln_inputs_are_named() -> None:
    with pytest.raises(ValueError) as error:
        _config(mapper=FLDPLN, COMID_Flow_File="q.csv", Print_Curve_File="c.csv")
    for key in ("Print_VDT_Database", "Flow_Direction_File", "Stream_Info_File", "FLDPLN_Library"):
        assert key in str(error.value)
    # The FLDPLN mapper reads neither the stream network nor the filled DEM
    assert "StrmShp_File" not in str(error.value) and "Filled_DEM_File" not in str(error.value)


@pytest.mark.parametrize(
    ("path", "expected"),
    [("vdt.parquet", True), ("library.pq", True), ("LIBRARY.PQ", True), (Path("dir.csv") / "flows.Parquet", True),
     ("flows.csv", False), ("library.parquet.csv", False), ("stream_info", False)],
)
def test_tables_ending_in_parquet_or_pq_are_parquet(path, expected: bool) -> None:
    assert is_parquet(path) is expected


def test_set_depth_needs_the_stream_raster_not_flows() -> None:
    with pytest.raises(ValueError, match="Missing required parameters: Stream_File$"):
        _config(FloodSpreader_SpecifyDepth="0.1")
    _config(FloodSpreader_SpecifyDepth="0.1", Stream_File="streams.tif")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({}, "COMID_Flow_File, Print_VDT_Database or Print_Curve_File", id="flood-map-needs-flows-and-curves"),
        pytest.param({**CURVES, "OutVEL": "v.tif"}, "LU_Manning_n, LU_Raster_SameRes", id="velocity-needs-land-cover"),
        pytest.param({**CURVES, "Flood_WaterLC_and_STRM_Cells": True}, "LU_Raster_SameRes$", id="water-cells-need-land-cover"),
        pytest.param({"Set_Depth": "0.1", "Stream_File": "s.tif", "mapper": FLDPLN}, "not supported by the FLDPLN mapper", id="fldpln-set-depth"),
        pytest.param({"Set_Depth": "0.1", "Stream_File": "s.tif", "OutVEL": "v.tif"}, "OutVEL", id="set-depth-velocity"),
        pytest.param({"mapper": "Curve2Flood-Mult-Point"}, r"mapper must be one of .* \(did you mean 'Curve2Flood-Multi-Point Interpolation'\?\)", id="unknown-mapper"),
        pytest.param({"MPI_HWM_Outlier_Method": "zscore"}, "mpi_hwm_outlier_method must be one of mad, iqr", id="unknown-outlier-method"),
        pytest.param({"MPI_Connectivity": 6}, "mpi_connectivity must be one of 4, 8", id="connectivity"),
        pytest.param({"TW_MultFact": "wide"}, "tw_multfact must be a number", id="non-numeric-float"),
        pytest.param({"LAND_WaterValue": "80.5"}, "land_watervalue must be an integer", id="non-integer-int"),
        pytest.param({"Bathy_Use_Banks": "maybe"}, "bathy_use_banks must be True or False", id="non-boolean"),
        pytest.param({"TW_MultFact": 0}, "TW_MultFact must be finite and positive", id="tw-multfact"),
        pytest.param({"TopWidthPlausibleLimit": "nan"}, "TopWidthPlausibleLimit must be finite and positive", id="plausible-limit"),
        pytest.param({"FLDPLN_Min_Flood_Depth": -0.01}, "FLDPLN_Min_Flood_Depth must be finite and 0 or more", id="min-flood-depth"),
    ],
)
def test_invalid_options_are_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"MPI_XS_Point_Spacing_m": 0}, "MPI_XS_Point_Spacing_m must be more than 0.0"),
        ({"MPI_K": 0}, "MPI_K must be 1 or more"),
        ({"MPI_Smooth_Sigma_Pixels": -1}, "MPI_Smooth_Sigma_Pixels must be 0.0 or more"),
        ({"MPI_Max_Distance_m": "inf"}, "MPI_Max_Distance_m must be more than 0.0"),
        ({"MPI_Min_TopWidth_m": 50, "MPI_Max_TopWidth_m": 20}, "MPI_Max_TopWidth_m .* is less than MPI_Min_TopWidth_m"),
    ],
)
def test_multi_point_settings_out_of_range_are_rejected(overrides: dict, message: str) -> None:
    """A spacing of 0 marched across a cross section forever."""
    with pytest.raises(ValueError, match=message):
        _config(**CURVES, mapper=MULTI_POINT, **overrides)


FLDPLN_INPUTS = {"mapper": FLDPLN, **CURVES, "Flow_Direction_File": "fdr.tif", "Stream_Info_File": "info.pq",
                 "FLDPLN_Library": "library.pq"}


def test_an_fldpln_flood_map_needs_no_dem_but_its_water_surface_does() -> None:
    config = Config.from_mapping({**FLDPLN_INPUTS, "OutFLD": "flood.tif", "OutDEP": "depth.tif"})
    assert not config.needs_dem and "DEM_File" not in config.input_paths()
    with pytest.raises(ValueError, match="Missing required parameters: DEM_File"):
        Config.from_mapping({**FLDPLN_INPUTS, "OutWSE": "wse.tif"})
    with pytest.raises(ValueError, match="Missing required parameters: DEM_File"):
        Config.from_mapping({**CURVES, "OutFLD": "flood.tif"})  # the other mappers always need it


@pytest.mark.parametrize(("flood", "gpkg"), [("out/flood.tif", "out/flood.gpkg"), ("flood.tiff", "flood.gpkg"),
                                             ("flood", "flood.gpkg"), ("my.flood.tif", "my.flood.gpkg")])
def test_the_flood_polygons_go_beside_the_flood_raster(flood: str, gpkg: str) -> None:
    assert Path(_config(**CURVES, OutFLD=flood).gpkg_file) == Path(gpkg)
    assert _config(**CURVES, OutFLD=flood, Make_Output_GPKG=False).gpkg_file is None


def test_the_paths_are_checked_before_a_run(tmp_path: Path) -> None:
    """Every missing input and every output that can't be written is named, all at once."""
    (tmp_path / "dem.tif").write_bytes(b"")
    (tmp_path / "q.csv").write_bytes(b"")
    (tmp_path / "taken").mkdir()
    config = Config.from_mapping({
        "DEM_File": tmp_path / "dem.tif", "COMID_Flow_File": tmp_path / "q.csv",
        "Print_VDT_Database": tmp_path / "vdt.parquet", "OutFLD": tmp_path / "missing" / "flood.tif",
        "OutDEP": tmp_path / "taken", "OutWSE": tmp_path / "wse.tif",
    })
    with pytest.raises(FileNotFoundError) as error:
        config.check_paths()
    message = str(error.value)
    assert "Print_VDT_Database" in message and "OutFLD's folder" in message and "OutDEP" in message
    assert "DEM_File" not in message and "COMID_Flow_File" not in message and "OutWSE" not in message
    assert "Make_Output_GPKG's folder" in message  # beside OutFLD
