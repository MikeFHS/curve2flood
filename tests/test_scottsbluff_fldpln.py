from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from curve2flood import Curve2Flood_MainFunction


DATA_DIR = Path(__file__).parent / "data" / "scottsbluff_fldpln"


def read_raster(path: Path) -> np.ndarray:
    dataset = gdal.Open(str(path))
    if dataset is None:
        raise FileNotFoundError(path)
    return dataset.GetRasterBand(1).ReadAsArray()


def csi_metrics(prediction_file: Path) -> dict[str, float | int]:
    prediction = read_raster(prediction_file) > 0
    reference = read_raster(DATA_DIR / "reference" / "scbfne_16.tif") > 0
    boundary = read_raster(DATA_DIR / "reference" / "fabdem_boundary.tif") > 0

    prediction &= boundary
    reference &= boundary

    true_positive = int((prediction & reference).sum())
    false_positive = int((prediction & ~reference).sum())
    false_negative = int((~prediction & reference).sum())
    true_negative = int((~prediction & ~reference & boundary).sum())
    csi = true_positive / (true_positive + false_positive + false_negative)

    return {
        "csi": csi,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "predicted_wet": int(prediction.sum()),
        "reference_wet": int(reference.sum()),
        "boundary_cells": int(boundary.sum()),
    }


def scottsbluff_params(output_file: Path) -> dict[str, str | int | bool]:
    inputs = DATA_DIR / "inputs"
    return {
        "DEM_File": str(inputs / "fabdem_fldpln_bathymetry.tif"),
        "Stream_File": str(inputs / "fabdem_matched.tif"),
        "LU_Manning_n": str(inputs / "AR_Manning_n_MED.txt"),
        "Print_VDT_Database": str(inputs / "GEOGLOWS_fabdem_VDT_Database_Bathy.parquet"),
        "StrmShp_File": str(inputs / "fabdem_matched.parquet"),
        "Comid_Flow_File": str(inputs / "stage=16.csv"),
        "mapper": "Curve2Flood-FLDPLNpy",
        "Flow_Direction_File": str(inputs / "fabdem_flowdir.tif"),
        "Filled_DEM_File": str(inputs / "fabdem_filled.tif"),
        "Stream_Info_File": str(inputs / "fabdem_stream_info.parquet"),
        "FLDPLN_Library": str(inputs / "fldpln_library.parquet"),
        "TW_MultFact": "1.5",
        "TopWidthPlausibleLimit": 6000,
        "percentile": "0.5",
        "Flood_WaterLC_and_STRM_Cells": False,
        "Make_Output_GPKG": False,
        "LU_Raster_SameRes": str(inputs / "fabdem_LAND_Raster.tif"),
        "LAND_WaterValue": 80,
        "OutFLD": str(output_file),
    }


@pytest.mark.integration
def test_scottsbluff_fldpln_csi():
    if not (DATA_DIR / "inputs" / "fldpln_library.parquet").exists():
        pytest.skip("Scottsbluff FLDPLN fixture is not available.")

    output_dir = DATA_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "scottsbluff_fldpln_test.tif"
    Curve2Flood_MainFunction(args=scottsbluff_params(output_file), quiet=True)

    metrics = csi_metrics(output_file)

    assert metrics["csi"] >= 0.493
    assert metrics["true_positive"] == 11530
    assert metrics["false_positive"] == 8374
    assert metrics["false_negative"] == 3483
