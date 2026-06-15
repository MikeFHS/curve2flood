from pathlib import Path

import numpy as np
import pandas as pd
from affine import Affine
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from shapely.geometry import Point
from rasterio.features import rasterize

from curve2flood._log import LOG

ID_SYNONYMS = ["COMID", "RIVID", "river_id", "LINKNO"]

def find_id_column(columns, context: str) -> str:
    """Find a stream identifier column using the FHS synonym set."""
    col_map = {str(c).lower(): c for c in columns}
    for candidate in ID_SYNONYMS:
        key = candidate.lower()
        if key in col_map:
            return col_map[key]
    raise ValueError(
        f"Could not find an ID column in {context}. Expected one of: {ID_SYNONYMS}. "
        f"Available columns: {list(columns)}"
    )


def _coerce_fhs_metadata_columns(
    table_df: pd.DataFrame,
    table_filename: str,
    one_based_rc: bool = False,
) -> pd.DataFrame:
    """
    Normalize COMID/row/col/elevation/slope/angle metadata into the shape used
    by the FHS point-generation helpers.
    """
    if table_df.empty:
        return pd.DataFrame(columns=["COMID", "Row", "Col", "Elev", "Slope", "XS_Angle"])

    id_col = find_id_column(table_df.columns, table_filename)
    required = {id_col, "Row", "Col"}
    missing_required = required - set(table_df.columns)
    if missing_required:
        raise ValueError(f"{Path(table_filename).name} is missing required columns: {sorted(missing_required)}")

    # Normalize the source table into the compact metadata layout used by the
    # MPI scenario builders and metadata backfill step.
    work_df = table_df.copy()
    work_df["COMID"] = pd.to_numeric(work_df[id_col], errors="coerce")
    work_df["Row"] = pd.to_numeric(work_df["Row"], errors="coerce")
    work_df["Col"] = pd.to_numeric(work_df["Col"], errors="coerce")

    if "Elev" in work_df.columns:
        work_df["Elev"] = pd.to_numeric(work_df["Elev"], errors="coerce")
    elif "DEM_Elev" in work_df.columns:
        work_df["Elev"] = pd.to_numeric(work_df["DEM_Elev"], errors="coerce")
    elif "BaseElev" in work_df.columns:
        work_df["Elev"] = pd.to_numeric(work_df["BaseElev"], errors="coerce")
    else:
        work_df["Elev"] = np.nan

    if "Slope" in work_df.columns:
        work_df["Slope"] = pd.to_numeric(work_df["Slope"], errors="coerce")
    else:
        work_df["Slope"] = np.nan

    if "XS_Angle" in work_df.columns:
        work_df["XS_Angle"] = pd.to_numeric(work_df["XS_Angle"], errors="coerce")
    else:
        work_df["XS_Angle"] = np.nan

    work_df = work_df.dropna(subset=["COMID", "Row", "Col"]).copy()
    if work_df.empty:
        return pd.DataFrame(columns=["COMID", "Row", "Col", "Elev", "Slope", "XS_Angle"])

    work_df["COMID"] = work_df["COMID"].astype(np.int64)
    work_df["Row"] = work_df["Row"].astype(np.int64)
    work_df["Col"] = work_df["Col"].astype(np.int64)
    if one_based_rc:
        work_df["Row"] -= 1
        work_df["Col"] -= 1

    return work_df[["COMID", "Row", "Col", "Elev", "Slope", "XS_Angle"]]


def merge_missing_fhs_metadata(
    scenario_df: pd.DataFrame,
    metadata_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Fill missing `Elev`, `Slope`, and `XS_Angle` values in a scenario table from
    a secondary metadata table keyed by COMID/Row/Col.
    """
    if scenario_df.empty or metadata_df is None or metadata_df.empty:
        return scenario_df

    # Join on the stable spatial key so missing elevation, slope, and cross-
    # section angle values can be borrowed from an alternate table.
    key_cols = ["COMID", "Row", "Col"]
    rhs = metadata_df.drop_duplicates(subset=key_cols)[key_cols + ["Elev", "Slope", "XS_Angle"]].rename(
        columns={
            "Elev": "Elev_metadata",
            "Slope": "Slope_metadata",
            "XS_Angle": "XS_Angle_metadata",
        }
    )
    merged = scenario_df.merge(rhs, on=key_cols, how="left")
    for col in ("Elev", "Slope", "XS_Angle"):
        fallback_col = f"{col}_metadata"
        if col not in merged.columns:
            merged[col] = merged[fallback_col]
        else:
            merged[col] = merged[col].where(pd.notna(merged[col]), merged[fallback_col])
        merged = merged.drop(columns=[fallback_col])
    return merged



def _compute_outlier_mask(values: np.ndarray, method: str = "mad", threshold: float = 3.5) -> np.ndarray:
    """
    FHS outlier-mask helper for depth-based HWM filtering.
    """
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 4:
        return np.zeros(values.shape, dtype=bool)

    valid_values = values[finite]
    method = str(method).lower()
    if method == "mad":
        center = np.median(valid_values)
        mad = np.median(np.abs(valid_values - center))
        if not np.isfinite(mad) or mad <= 1e-12:
            return np.zeros(values.shape, dtype=bool)
        robust_z = 0.6745 * (values - center) / mad
        outliers = np.zeros(values.shape, dtype=bool)
        outliers[finite] = np.abs(robust_z[finite]) > float(threshold)
        return outliers
    if method == "iqr":
        q1, q3 = np.percentile(valid_values, [25.0, 75.0])
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 1e-12:
            return np.zeros(values.shape, dtype=bool)
        lower = q1 - float(threshold) * iqr
        upper = q3 + float(threshold) * iqr
        outliers = np.zeros(values.shape, dtype=bool)
        outliers[finite] = (values[finite] < lower) | (values[finite] > upper)
        return outliers
    raise ValueError(f"Unsupported HWM outlier method: {method}")

def filter_hwm_outliers_single_scenario(
    results_df: pd.DataFrame,
    method: str = "mad",
    threshold: float = 3.5,
    group_by: str = "comid",
    min_samples: int = 5,
) -> pd.DataFrame:
    """
    Apply the FHS HWM outlier filter to one interpolated scenario table.

    The filter operates on flood depth (`WaterSurfaceElev_m - Elev`) and can
    run globally or per COMID. Rows that lose their scenario values are
    dropped from the returned DataFrame.
    """
    filtered_df = results_df.copy()
    group_by = str(group_by).lower()
    if group_by not in {"global", "comid"}:
        raise ValueError("group_by must be either 'global' or 'comid'.")

    if "Elev" not in filtered_df.columns or "WaterSurfaceElev_m" not in filtered_df.columns:
        return filtered_df

    elev = filtered_df["Elev"].to_numpy(dtype=np.float64, copy=False)
    wse = filtered_df["WaterSurfaceElev_m"].to_numpy(dtype=np.float64, copy=False)
    valid_depth = np.isfinite(wse) & np.isfinite(elev)
    depth = wse - elev
    scenario_mask = np.zeros(len(filtered_df), dtype=bool)

    if group_by == "global":
        candidate_mask = valid_depth
        if np.count_nonzero(candidate_mask) >= int(min_samples):
            local_mask = _compute_outlier_mask(depth[candidate_mask], method=method, threshold=threshold)
            idx = np.where(candidate_mask)[0]
            scenario_mask[idx[local_mask]] = True
    else:
        comids = filtered_df["COMID"].to_numpy()
        for comid in pd.unique(comids):
            group_mask = (comids == comid) & valid_depth
            if np.count_nonzero(group_mask) < int(min_samples):
                continue
            local_mask = _compute_outlier_mask(depth[group_mask], method=method, threshold=threshold)
            idx = np.where(group_mask)[0]
            scenario_mask[idx[local_mask]] = True

    if np.any(scenario_mask):
        cols_to_nan = ["WaterSurfaceElev_m"]
        for candidate in ("TopWidth_m", "Velocity_mps", "Flow"):
            if candidate in filtered_df.columns:
                cols_to_nan.append(candidate)
        filtered_df.loc[scenario_mask, cols_to_nan] = np.nan

    keep_rows = filtered_df["WaterSurfaceElev_m"].notna()
    return filtered_df.loc[keep_rows].copy()

def read_curve2flood_table(table_filename: str) -> pd.DataFrame:
    """
    Read a CSV/TSV/parquet table used by Curve2Flood and FHS helper paths.
    """
    if not table_filename:
        raise ValueError("A table filename is required.")
    lower_name = table_filename.lower()
    if lower_name.endswith(".parquet"):
        return pd.read_parquet(table_filename, engine="fastparquet")
    if lower_name.endswith(".tsv"):
        return pd.read_csv(table_filename, sep="\t")
    return pd.read_csv(table_filename)

def get_curve_columns(columns) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Return sorted q_, v_, t_, and wse_ column groups from an FHS/VDT table.
    """
    q_cols = sorted([c for c in columns if str(c).startswith("q_")], key=lambda x: int(str(x).split("_")[1]))
    v_cols = sorted([c for c in columns if str(c).startswith("v_")], key=lambda x: int(str(x).split("_")[1]))
    t_cols = sorted([c for c in columns if str(c).startswith("t_")], key=lambda x: int(str(x).split("_")[1]))
    wse_cols = sorted([c for c in columns if str(c).startswith("wse_")], key=lambda x: int(str(x).split("_")[1]))
    if not len(q_cols) == len(v_cols) == len(t_cols) == len(wse_cols):
        raise ValueError("Mismatch in number of q_, v_, t_, and wse_ columns.")
    if len(q_cols) == 0:
        raise ValueError("No q_, v_, t_, wse_ columns found in the VDT file.")
    return q_cols, v_cols, t_cols, wse_cols


def interp_row_clamped(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    """
    FHS-style 1D interpolation with endpoint clamping.
    """
    n = xp.size
    if n == 0:
        return np.nan
    if n == 1:
        return float(fp[0])
    if x <= xp[0]:
        return float(fp[0])
    if x >= xp[-1]:
        return float(fp[-1])
    idx = np.searchsorted(xp, x)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]
    if x1 == x0:
        return float(y0)
    w = (x - x0) / (x1 - x0)
    return float(y0 + w * (y1 - y0))

def clean_and_interp_one_row(target_q: float, q_row: np.ndarray, v_row: np.ndarray, t_row: np.ndarray, wse_row: np.ndarray) -> tuple[float, float, float]:
    """
    Clean one VDT row and interpolate velocity, top width, and WSE at `target_q`.
    This is a direct port of the FHS row-interpolation logic.
    """
    valid = (
        np.isfinite(q_row) &
        np.isfinite(v_row) &
        np.isfinite(t_row) &
        np.isfinite(wse_row) &
        (q_row > 0.0) &
        (v_row >= 0.0) &
        (t_row > 0.0) &
        (wse_row > -90.0)
    )
    if not np.any(valid):
        return np.nan, np.nan, np.nan

    q = q_row[valid]
    v = v_row[valid]
    t = t_row[valid]
    wse = wse_row[valid]

    order = np.argsort(q)
    q = q[order]
    v = v[order]
    t = t[order]
    wse = wse[order]

    q_unique, unique_idx = np.unique(q, return_index=True)
    v = v[unique_idx]
    t = t[unique_idx]
    wse = wse[unique_idx]

    out_v = interp_row_clamped(target_q, q_unique, v)
    out_t = interp_row_clamped(target_q, q_unique, t)
    out_wse = interp_row_clamped(target_q, q_unique, wse)
    return out_v, out_t, out_wse

def build_fhs_scenario_dataframe_from_vdt(
    vdt_database_filename: str,
    comid_unique_flow: dict,
    one_based_rc: bool = False,
) -> pd.DataFrame:
    """
    Build a single-scenario HWM-style table directly from a VDT database and a
    COMID->flow mapping, following the interpolation behavior of
    `FHS_FloodMapper_AllInOne.py`.

    Returns a DataFrame with one row per interpolated centerline point and the
    columns needed by the FHS point-expansion and flood-mapping steps.
    """
    if not vdt_database_filename:
        raise ValueError("A VDT database is required for FHS VDT point generation.")

    vdt_df = read_curve2flood_table(vdt_database_filename)
    if vdt_df.empty:
        raise ValueError("The VDT database is empty.")

    q_cols, v_cols, t_cols, wse_cols = get_curve_columns(vdt_df.columns)
    work_df = _coerce_fhs_metadata_columns(vdt_df, vdt_database_filename, one_based_rc=one_based_rc)
    flow_series = work_df["COMID"].map(comid_unique_flow)
    work_df["Flow"] = pd.to_numeric(flow_series, errors="coerce")
    work_df = work_df.dropna(subset=["Flow"]).copy()
    if work_df.empty:
        return pd.DataFrame(columns=[
            "COMID", "Row", "Col", "Elev", "Slope", "XS_Angle",
            "Flow", "Velocity_mps", "TopWidth_m", "WaterSurfaceElev_m",
        ])

    # Interpolate the VDT vectors at the scenario discharge for each stream
    # cell that participates in the current event.
    q_arr = vdt_df.loc[work_df.index, q_cols].to_numpy(dtype=np.float64, copy=False)
    v_arr = vdt_df.loc[work_df.index, v_cols].to_numpy(dtype=np.float64, copy=False)
    t_arr = vdt_df.loc[work_df.index, t_cols].to_numpy(dtype=np.float64, copy=False)
    wse_arr = vdt_df.loc[work_df.index, wse_cols].to_numpy(dtype=np.float64, copy=False)
    target_q_arr = work_df["Flow"].to_numpy(dtype=np.float64, copy=False)

    out_v = np.full(len(work_df), np.nan, dtype=np.float64)
    out_t = np.full(len(work_df), np.nan, dtype=np.float64)
    out_wse = np.full(len(work_df), np.nan, dtype=np.float64)

    for i in range(len(work_df)):
        if not np.isfinite(target_q_arr[i]):
            continue
        vi, ti, wi = clean_and_interp_one_row(target_q_arr[i], q_arr[i, :], v_arr[i, :], t_arr[i, :], wse_arr[i, :])
        out_v[i] = vi
        out_t[i] = ti
        out_wse[i] = wi

    work_df["Velocity_mps"] = out_v
    work_df["TopWidth_m"] = out_t
    work_df["WaterSurfaceElev_m"] = out_wse
    work_df = work_df.dropna(subset=["TopWidth_m", "WaterSurfaceElev_m"]).copy()
    return work_df[[
        "COMID", "Row", "Col", "Elev", "Slope", "XS_Angle",
        "Flow", "Velocity_mps", "TopWidth_m", "WaterSurfaceElev_m",
    ]]

def filter_outliers(group):
    """
    A function to filter outliers based on mean and standard deviation for each COMID group
    """
    # Calculate mean and standard deviation for TopWidth
    topwidth_mean = group['TopWidth'].mean()
    topwidth_std = group['TopWidth'].std()
    lower_bound_tw = topwidth_mean - 2 * topwidth_std
    upper_bound_tw = topwidth_mean + 2 * topwidth_std

    # Filter TopWidth outliers
    group = group[(group['TopWidth'] >= lower_bound_tw) & (group['TopWidth'] <= upper_bound_tw)]

    # Calculate mean and standard deviation for WSE
    wse_mean = group['WSE'].mean()
    wse_std = group['WSE'].std()
    lower_bound_wse = wse_mean - 2 * wse_std
    upper_bound_wse = wse_mean + 2 * wse_std

    # Filter WSE outliers
    group = group[(group['WSE'] >= lower_bound_wse) & (group['WSE'] <= upper_bound_wse)]

    # Calculate mean and standard deviation for Velocity
    wse_mean = group['Velocity'].mean()
    wse_std = group['Velocity'].std()
    lower_bound_wse = wse_mean - 2 * wse_std
    upper_bound_wse = wse_mean + 2 * wse_std

    # Filter WSE outliers
    group = group[(group['Velocity'] >= lower_bound_wse) & (group['Velocity'] <= upper_bound_wse)]

    return group

def compute_tw_multfact_scale(flow: np.ndarray,
                              qbase: np.ndarray,
                              k: float = 1,
                              min_s: float = 0.1,
                              max_s: float = 3) -> np.ndarray:
    flow = flow.astype(np.float32, copy=False)
    qbase = qbase.astype(np.float32, copy=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(qbase > 0, flow / qbase, 1.0)
        scale = np.power(ratio, k)
    scale = np.clip(scale, min_s, max_s)
    scale = np.where(np.isfinite(scale), scale, 1.0)
    return scale.astype(np.float32)

def build_fhs_scenario_dataframe_from_curve_file(
    curve_param_filename: str,
    comid_unique_flow: dict,
    tw_multfact: float,
    one_based_rc: bool = False,
) -> pd.DataFrame:
    """
    Build a single-scenario FHS point table from a Curve2Flood curve-parameter
    file.

    Supported curve-file layouts:
    - legacy coefficient tables with `depth_*`, `tw_*`, and `vel_*` columns
    - ARC reach-average outputs that also carry `q_*`, `v_*`, `t_*`, and
      `wse_*` columns

    When both VDT and curve files exist, `multi_point_interpolation(...)`
    continues to prefer the VDT interpolation path because that is the original
    FHS workflow. This builder is used when the curve file is the only source of
    usable scenario points or when it carries the only available `XS_Angle`
    values.
    """
    if not curve_param_filename:
        raise ValueError("A curve-parameter file is required for curve-driven FHS point generation.")

    curve_df = read_curve2flood_table(curve_param_filename)
    if curve_df.empty:
        raise ValueError("The curve-parameter file is empty.")

    work_df = _coerce_fhs_metadata_columns(curve_df, curve_param_filename, one_based_rc=one_based_rc)
    flow_series = work_df["COMID"].map(comid_unique_flow)
    work_df["Flow"] = pd.to_numeric(flow_series, errors="coerce")
    work_df = work_df.dropna(subset=["Flow"]).copy()
    if work_df.empty:
        return pd.DataFrame(columns=[
            "COMID", "Row", "Col", "Elev", "Slope", "XS_Angle",
            "Flow", "Velocity_mps", "TopWidth_m", "WaterSurfaceElev_m",
        ])

    curve_df = curve_df.loc[work_df.index].copy()

    q_cols = [c for c in curve_df.columns if str(c).startswith("q_")]
    v_cols = [c for c in curve_df.columns if str(c).startswith("v_")]
    t_cols = [c for c in curve_df.columns if str(c).startswith("t_")]
    wse_cols = [c for c in curve_df.columns if str(c).startswith("wse_")]
    has_vdt_style_curve = bool(q_cols and v_cols and t_cols and wse_cols)

    if has_vdt_style_curve:
        # Some curve files already contain VDT-style hydraulic vectors, so
        # route them through the same interpolation logic as a true VDT table.
        q_cols, v_cols, t_cols, wse_cols = get_curve_columns(curve_df.columns)
        q_arr = curve_df[q_cols].to_numpy(dtype=np.float64, copy=False)
        v_arr = curve_df[v_cols].to_numpy(dtype=np.float64, copy=False)
        t_arr = curve_df[t_cols].to_numpy(dtype=np.float64, copy=False)
        wse_arr = curve_df[wse_cols].to_numpy(dtype=np.float64, copy=False)
        target_q_arr = work_df["Flow"].to_numpy(dtype=np.float64, copy=False)

        out_v = np.full(len(work_df), np.nan, dtype=np.float64)
        out_t = np.full(len(work_df), np.nan, dtype=np.float64)
        out_wse = np.full(len(work_df), np.nan, dtype=np.float64)

        for i in range(len(work_df)):
            if not np.isfinite(target_q_arr[i]):
                continue
            vi, ti, wi = clean_and_interp_one_row(target_q_arr[i], q_arr[i, :], v_arr[i, :], t_arr[i, :], wse_arr[i, :])
            out_v[i] = vi
            out_t[i] = ti
            out_wse[i] = wi

        work_df["Velocity_mps"] = out_v
        work_df["TopWidth_m"] = out_t
        work_df["WaterSurfaceElev_m"] = out_wse
        work_df = work_df.dropna(subset=["TopWidth_m", "WaterSurfaceElev_m"]).copy()
        return work_df[[
            "COMID", "Row", "Col", "Elev", "Slope", "XS_Angle",
            "Flow", "Velocity_mps", "TopWidth_m", "WaterSurfaceElev_m",
        ]]

    required_curve_cols = {"depth_a", "depth_b", "tw_a", "tw_b", "vel_a", "vel_b"}
    missing_curve_cols = required_curve_cols - set(curve_df.columns)
    if missing_curve_cols:
        raise ValueError(
            f"{Path(curve_param_filename).name} is missing curve columns {sorted(missing_curve_cols)} "
            "and does not expose VDT-style q_/v_/t_/wse_ fields."
        )

    for col in required_curve_cols:
        curve_df[col] = pd.to_numeric(curve_df[col], errors="coerce")
    if "BaseElev" in curve_df.columns:
        curve_df["BaseElev"] = pd.to_numeric(curve_df["BaseElev"], errors="coerce")
    elif "Elev" in curve_df.columns:
        curve_df["BaseElev"] = pd.to_numeric(curve_df["Elev"], errors="coerce")
    elif "DEM_Elev" in curve_df.columns:
        curve_df["BaseElev"] = pd.to_numeric(curve_df["DEM_Elev"], errors="coerce")
    else:
        curve_df["BaseElev"] = np.nan

    # Otherwise rebuild one hydraulic state per row from the power-law curve
    # coefficients at the current event flow.
    curve_df["Flow"] = work_df["Flow"]
    curve_df["Depth"] = curve_df["depth_a"] * curve_df["Flow"] ** curve_df["depth_b"]
    curve_df["TopWidth"] = curve_df["tw_a"] * curve_df["Flow"] ** curve_df["tw_b"]
    curve_df["Velocity"] = curve_df["vel_a"] * curve_df["Flow"] ** curve_df["vel_b"]
    curve_df = curve_df[curve_df["Depth"] > 0].copy()
    curve_df = curve_df[curve_df["TopWidth"] > 0].copy()
    curve_df["WSE"] = curve_df["Depth"] + curve_df["BaseElev"]

    if curve_df.empty:
        return pd.DataFrame(columns=[
            "COMID", "Row", "Col", "Elev", "Slope", "XS_Angle",
            "Flow", "Velocity_mps", "TopWidth_m", "WaterSurfaceElev_m",
        ])

    curve_df = curve_df.groupby("COMID", group_keys=False)[curve_df.columns].apply(filter_outliers)
    if curve_df.empty:
        return pd.DataFrame(columns=[
            "COMID", "Row", "Col", "Elev", "Slope", "XS_Angle",
            "Flow", "Velocity_mps", "TopWidth_m", "WaterSurfaceElev_m",
        ])

    if "QBaseflow" in curve_df.columns:
        qbase = pd.to_numeric(curve_df["QBaseflow"], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        tw_scale = compute_tw_multfact_scale(
            curve_df["Flow"].to_numpy(dtype=np.float32, copy=False),
            qbase,
        )
    else:
        tw_scale = np.ones(len(curve_df), dtype=np.float32)

    curve_df["TopWidth"] = curve_df["TopWidth"].to_numpy(dtype=np.float32, copy=False) * (float(tw_multfact) * tw_scale)
    work_df = work_df.loc[curve_df.index].copy()
    work_df["Velocity_mps"] = curve_df["Velocity"].to_numpy(dtype=np.float64, copy=False)
    work_df["TopWidth_m"] = curve_df["TopWidth"].to_numpy(dtype=np.float64, copy=False)
    work_df["WaterSurfaceElev_m"] = curve_df["WSE"].to_numpy(dtype=np.float64, copy=False)
    return work_df[[
        "COMID", "Row", "Col", "Elev", "Slope", "XS_Angle",
        "Flow", "Velocity_mps", "TopWidth_m", "WaterSurfaceElev_m",
    ]]

def sample_dem_elevation_at_local_xy(dem_array: np.ndarray, x: float, y: float, dx: float, dy: float) -> float:
    """
    Sample the DEM using local metric coordinates generated by `_grid_xy_from_rc`.
    """
    col = int(np.floor(float(x) / float(dx)))
    row = int(np.floor(-float(y) / float(dy)))
    if row < 0 or col < 0 or row >= dem_array.shape[0] or col >= dem_array.shape[1]:
        return np.nan
    elev = float(dem_array[row, col])
    return elev if np.isfinite(elev) else np.nan

def _normalize_xs_angle(xs_angle: float) -> float:
    xs_angle = float(xs_angle) % np.pi
    if xs_angle < 0:
        xs_angle += np.pi
    return xs_angle

def _get_xs_unit_vector_in_local_grid(xs_angle: float, dx: float, dy: float) -> tuple[float, float]:
    """
    Convert an FHS/VDT `XS_Angle` to a unit vector on Curve2Flood's local
    metric grid.
    """
    xs_angle = _normalize_xs_angle(xs_angle)
    dc = np.cos(xs_angle)
    dr = np.sin(xs_angle)
    vx = float(dx) * float(dc)
    vy = -float(dy) * float(dr)
    norm = float(np.hypot(vx, vy))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("Could not compute a valid cross-section direction vector.")
    return vx / norm, vy / norm

def create_fhs_points_from_scenario_dataframe(
    results_df: pd.DataFrame,
    dem_array: np.ndarray,
    dx: float,
    dy: float,
    topwidth_threshold_m: float | None = None,
    xs_point_spacing_m: float = 100.0,
) -> dict[str, np.ndarray]:
    """
    Convert one FHS-style scenario DataFrame into point arrays for
    `create_fhs_flood_map_from_points`.

    This ports the point-generation behavior of `create_geojson_features(...)`
    into an array-native form suited to Curve2Flood. Centerline points are
    always included. If `topwidth_threshold_m` is provided and `XS_Angle`
    values are available, cross-section extension points are added on both
    sides until the water surface is blocked by terrain or the half-topwidth is
    exhausted.
    """
    if results_df.empty:
        empty_float = np.empty(0, dtype=np.float32)
        empty_int = np.empty(0, dtype=np.int32)
        empty_obj = np.empty(0, dtype=object)
        return {
            "rows": empty_int,
            "cols": empty_int,
            "x": empty_float,
            "y": empty_float,
            "wse": empty_float,
            "topwidth": empty_float,
            "dem": empty_float,
            "ids": empty_int,
            "types": empty_obj,
            "slope": empty_float,
        }

    rows = results_df["Row"].to_numpy(dtype=np.int32, copy=False)
    cols = results_df["Col"].to_numpy(dtype=np.int32, copy=False)
    x, y = _grid_xy_from_rc(rows, cols, dx, dy)
    center_dem = results_df["Elev"].to_numpy(dtype=np.float32, copy=False)
    if np.all(~np.isfinite(center_dem)):
        center_dem = np.full(len(results_df), np.nan, dtype=np.float32)
        inside = (rows >= 0) & (rows < dem_array.shape[0]) & (cols >= 0) & (cols < dem_array.shape[1])
        center_dem[inside] = dem_array[rows[inside], cols[inside]].astype(np.float32)

    # Start with one centerline point per row in the scenario table. Optional
    # cross-section extension points are appended below.
    point_data = {
        "rows": rows.astype(np.int32),
        "cols": cols.astype(np.int32),
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "wse": results_df["WaterSurfaceElev_m"].to_numpy(dtype=np.float32, copy=False),
        "topwidth": results_df["TopWidth_m"].to_numpy(dtype=np.float32, copy=False),
        "dem": center_dem.astype(np.float32),
        "ids": results_df["COMID"].to_numpy(dtype=np.int32, copy=False),
        "types": np.full(len(results_df), "Centerline", dtype=object),
        "slope": results_df["Slope"].to_numpy(dtype=np.float32, copy=False) if "Slope" in results_df.columns else np.full(len(results_df), np.nan, dtype=np.float32),
    }

    if topwidth_threshold_m is None or "XS_Angle" not in results_df.columns:
        return point_data

    extra_rows = []
    extra_cols = []
    extra_x = []
    extra_y = []
    extra_wse = []
    extra_topwidth = []
    extra_dem = []
    extra_ids = []
    extra_types = []
    extra_slope = []

    for _, row in results_df.iterrows():
        # March outward from the centerline along the cross section until the
        # DEM blocks the water surface or the half-top-width is exhausted.
        xs_angle = row["XS_Angle"]
        topwidth_value = row["TopWidth_m"]
        scenario_wse = row["WaterSurfaceElev_m"]
        if not np.isfinite(xs_angle) or not np.isfinite(topwidth_value) or not np.isfinite(scenario_wse):
            continue
        if float(topwidth_value) <= float(topwidth_threshold_m):
            continue

        center_x, center_y = _grid_xy_from_rc(
            np.asarray([int(row["Row"])], dtype=np.int32),
            np.asarray([int(row["Col"])], dtype=np.int32),
            dx,
            dy,
        )
        ux, uy = _get_xs_unit_vector_in_local_grid(float(xs_angle), dx, dy)
        half_width = 0.5 * float(topwidth_value)
        offset_m = float(xs_point_spacing_m)
        side_blocked = {"positive": False, "negative": False}

        while offset_m <= half_width + 1e-9:
            for side_sign, side_label in ((1.0, "positive"), (-1.0, "negative")):
                if side_blocked[side_label]:
                    continue

                px = float(center_x[0] + side_sign * offset_m * ux)
                py = float(center_y[0] + side_sign * offset_m * uy)
                point_elev = sample_dem_elevation_at_local_xy(dem_array, px, py, dx, dy)
                if not np.isfinite(point_elev):
                    side_blocked[side_label] = True
                    continue
                if float(scenario_wse) <= float(point_elev):
                    side_blocked[side_label] = True
                    continue
                if float(scenario_wse) - float(point_elev) > 20.0:
                    LOG.warning(
                        f"Large depth ({float(scenario_wse) - float(point_elev):.2f} m) at COMID {int(row['COMID'])}, "
                        f"{side_label} side, offset {offset_m:.2f} m."
                    )

                extra_rows.append(-1)
                extra_cols.append(-1)
                extra_x.append(px)
                extra_y.append(py)
                extra_wse.append(float(scenario_wse))
                extra_topwidth.append(float(topwidth_value))
                extra_dem.append(float(point_elev))
                extra_ids.append(int(row["COMID"]))
                extra_types.append("XS_Extension")
                extra_slope.append(float(row["Slope"]) if "Slope" in row and np.isfinite(row["Slope"]) else np.nan)

            if all(side_blocked.values()):
                break
            offset_m += float(xs_point_spacing_m)

    if not extra_x:
        return point_data

    for key, arr in {
        "rows": np.asarray(extra_rows, dtype=np.int32),
        "cols": np.asarray(extra_cols, dtype=np.int32),
        "x": np.asarray(extra_x, dtype=np.float32),
        "y": np.asarray(extra_y, dtype=np.float32),
        "wse": np.asarray(extra_wse, dtype=np.float32),
        "topwidth": np.asarray(extra_topwidth, dtype=np.float32),
        "dem": np.asarray(extra_dem, dtype=np.float32),
        "ids": np.asarray(extra_ids, dtype=np.int32),
        "types": np.asarray(extra_types, dtype=object),
        "slope": np.asarray(extra_slope, dtype=np.float32),
    }.items():
        point_data[key] = np.concatenate((point_data[key], arr))

    return point_data


def _grid_xy_from_rc(rows: np.ndarray, cols: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert zero-based raster row/column indices to cell-center coordinates on a
    local projected grid measured in meters.

    The x axis increases to the right. The y axis is negative downward so the
    generated Affine transform matches raster row indexing.
    """
    x = (cols.astype(np.float32) + np.float32(0.5)) * np.float32(dx)
    y = -((rows.astype(np.float32) + np.float32(0.5)) * np.float32(dy))
    return x.astype(np.float32), y.astype(np.float32)

def build_fhs_points_from_curve2flood_inputs(
    RR: np.ndarray,
    CC: np.ndarray,
    T_Rast: np.ndarray,
    W_Rast: np.ndarray,
    S_Rast: np.ndarray | None,
    E: np.ndarray,
    B: np.ndarray,
    COMID_Unique_TW: dict[int, float],
    COMID_Unique_Depth: dict[int, float],
    TopWidthPlausibleLimit: float,
    Set_Depth: float,
    dx: float,
    dy: float,
    COMID_Averaging_Method: int = 0,
    extra_points: dict | None = None,
) -> dict[str, np.ndarray]:
    """
    Build an HWM-style point set from the stream cells already available inside
    Curve2Flood.

    This is the array-based bridge between Curve2Flood's centerline/VDT inputs
    and the FHS flood-mapping engine. By default it emits one point per stream
    cell. Optional `extra_points` can be supplied to append any off-center
    points, such as cross-section extension points generated elsewhere.

    Parameters
    ----------
    RR, CC : ndarray
        Padded stream-cell row/column indices, as used throughout Curve2Flood.
    T_Rast, W_Rast, S_Rast : ndarray
        AutoRoute/ARC top-width, WSE, and slope rasters on the unpadded DEM
        grid. `S_Rast` may be None.
    E, B : ndarray
        Padded DEM and stream-ID rasters.
    COMID_Unique_TW, COMID_Unique_Depth : dict
        Per-COMID fallback top width and depth values.
    TopWidthPlausibleLimit : float
        Maximum plausible top width in meters, used when Set_Depth is active.
    Set_Depth : float
        Fixed depth override. When positive, WSE is computed as E + Set_Depth.
    dx, dy : float
        Cell spacing in meters.
    COMID_Averaging_Method : int, default 0
        Non-zero forces use of COMID-mean depth/top-width rather than cellwise
        ARC rasters.
    extra_points : dict, optional
        Optional additional HWM-style points. Supported keys are:
        `rows`, `cols`, `x`, `y`, `wse`, `topwidth`, `dem`, `ids`, `types`,
        and `slope`.

    Returns
    -------
    dict
        Dictionary containing point arrays: `rows`, `cols`, `x`, `y`, `wse`,
        `topwidth`, `dem`, `ids`, `types`, and `slope`.

    Notes
    -----
    This adapter does not generate the cross-section expansion points from the
    original FHS script on its own because Curve2Flood does not currently carry
    XS angle metadata through this code path. To reproduce the exact FHS point
    cloud, pass those extra points through `extra_points`.
    """
    rows = []
    cols = []
    wse_vals = []
    topwidth_vals = []
    dem_vals = []
    id_vals = []
    slope_vals = []

    for i in range(len(RR)):
        r = int(RR[i])
        c = int(CC[i])
        comid_value = int(B[r, c])
        point_dem = float(E[r, c])

        # Match Curve2Flood's existing precedence: fixed-depth override first,
        # then COMID fallback hydraulics, then cellwise raster values.
        if Set_Depth > 0.0:
            point_wse = point_dem + float(Set_Depth)
            point_topwidth = float(TopWidthPlausibleLimit)
            point_slope = float(S_Rast[r - 1, c - 1]) if S_Rast is not None else np.nan
        elif COMID_Averaging_Method != 0 or W_Rast[r - 1, c - 1] < 0.001 or T_Rast[r - 1, c - 1] < 0.00001:
            point_topwidth = float(COMID_Unique_TW.get(comid_value, 0.0))
            point_depth = float(COMID_Unique_Depth.get(comid_value, 0.0))
            point_wse = point_dem + point_depth
            point_slope = float(S_Rast[r - 1, c - 1]) if S_Rast is not None else np.nan
        else:
            point_wse = float(np.round(W_Rast[r - 1, c - 1], 2))
            point_topwidth = float(T_Rast[r - 1, c - 1])
            point_slope = float(S_Rast[r - 1, c - 1]) if S_Rast is not None else np.nan

        if (not np.isfinite(point_wse)) or point_wse < 0.001:
            continue
        if (not np.isfinite(point_topwidth)) or point_topwidth < 0.00001:
            continue
        if (not np.isfinite(point_dem)) or point_dem <= -9998.0:
            continue
        if point_wse - point_dem < 0.001:
            continue

        rows.append(r - 1)
        cols.append(c - 1)
        wse_vals.append(point_wse)
        topwidth_vals.append(point_topwidth)
        dem_vals.append(point_dem)
        id_vals.append(comid_value)
        slope_vals.append(point_slope)

    point_rows = np.asarray(rows, dtype=np.int32)
    point_cols = np.asarray(cols, dtype=np.int32)
    point_x, point_y = _grid_xy_from_rc(point_rows, point_cols, dx, dy)

    point_data = {
        "rows": point_rows,
        "cols": point_cols,
        "x": point_x,
        "y": point_y,
        "wse": np.asarray(wse_vals, dtype=np.float32),
        "topwidth": np.asarray(topwidth_vals, dtype=np.float32),
        "dem": np.asarray(dem_vals, dtype=np.float32),
        "ids": np.asarray(id_vals, dtype=np.int32),
        "types": np.full(len(point_rows), "Centerline", dtype=object),
        "slope": np.asarray(slope_vals, dtype=np.float32),
    }

    if extra_points is None:
        return point_data

    if "wse" not in extra_points:
        raise ValueError("extra_points must include a 'wse' array.")

    extra_wse = np.asarray(extra_points["wse"], dtype=np.float32)
    extra_count = len(extra_wse)
    if extra_count == 0:
        return point_data

    if "x" in extra_points and "y" in extra_points:
        extra_x = np.asarray(extra_points["x"], dtype=np.float32)
        extra_y = np.asarray(extra_points["y"], dtype=np.float32)
        if len(extra_x) != extra_count or len(extra_y) != extra_count:
            raise ValueError("extra_points x/y arrays must match the length of extra_points['wse'].")
        if "rows" in extra_points and "cols" in extra_points:
            extra_rows = np.asarray(extra_points["rows"], dtype=np.int32)
            extra_cols = np.asarray(extra_points["cols"], dtype=np.int32)
        else:
            extra_rows = np.full(extra_count, -1, dtype=np.int32)
            extra_cols = np.full(extra_count, -1, dtype=np.int32)
    elif "rows" in extra_points and "cols" in extra_points:
        extra_rows = np.asarray(extra_points["rows"], dtype=np.int32)
        extra_cols = np.asarray(extra_points["cols"], dtype=np.int32)
        if len(extra_rows) != extra_count or len(extra_cols) != extra_count:
            raise ValueError("extra_points rows/cols arrays must match the length of extra_points['wse'].")
        extra_x, extra_y = _grid_xy_from_rc(extra_rows, extra_cols, dx, dy)
    else:
        raise ValueError("extra_points must provide either x/y arrays or rows/cols arrays.")

    extra_topwidth = np.asarray(
        extra_points.get("topwidth", np.full(extra_count, TopWidthPlausibleLimit, dtype=np.float32)),
        dtype=np.float32,
    )
    extra_dem = np.asarray(extra_points.get("dem", np.full(extra_count, np.nan, dtype=np.float32)), dtype=np.float32)
    extra_ids = np.asarray(extra_points.get("ids", np.full(extra_count, -1, dtype=np.int32)), dtype=np.int32)
    extra_types = np.asarray(extra_points.get("types", np.full(extra_count, "XS_Extension", dtype=object)), dtype=object)
    extra_slope = np.asarray(extra_points.get("slope", np.full(extra_count, np.nan, dtype=np.float32)), dtype=np.float32)

    for key, arr in {
        "topwidth": extra_topwidth,
        "dem": extra_dem,
        "ids": extra_ids,
        "types": extra_types,
        "slope": extra_slope,
    }.items():
        if len(arr) != extra_count:
            raise ValueError(f"extra_points['{key}'] must match the length of extra_points['wse'].")

    for key, arr in {
        "rows": extra_rows,
        "cols": extra_cols,
        "x": extra_x,
        "y": extra_y,
        "wse": extra_wse,
        "topwidth": extra_topwidth,
        "dem": extra_dem,
        "ids": extra_ids,
        "types": extra_types,
        "slope": extra_slope,
    }.items():
        point_data[key] = np.concatenate((point_data[key], arr))

    return point_data

def build_variable_buffer_masks_from_points(
    x: np.ndarray,
    y: np.ndarray,
    topwidth: np.ndarray,
    dem_shape: tuple[int, int],
    transform: Affine,
    fixed_corridor_buffer_m: float,
    fixed_anchor_buffer_m: float,
    use_topwidth_buffers: bool = True,
    corridor_topwidth_factor: float = 1.5,
    anchor_topwidth_factor: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Array-based version of the FHS point buffering step.

    Each point produces two raster masks:
    - `corridor`: the area eligible for IDW interpolation
    - `anchor`: the seed region used for hydraulic connectivity propagation
    """
    if len(x) == 0:
        empty = np.zeros(dem_shape, dtype=bool)
        return empty, empty

    # Rasterize per-point buffers so interpolation is constrained to the stream
    # corridor and connectivity starts from an anchor region near the points.
    points = [Point(float(px), float(py)) for px, py in zip(x, y)]
    if use_topwidth_buffers:
        corridor_shapes = (
            (geom.buffer(float(max(width, 0.0) * corridor_topwidth_factor)), 1)
            for geom, width in zip(points, topwidth)
        )
        anchor_shapes = (
            (geom.buffer(float(max(width, 0.0) * anchor_topwidth_factor)), 1)
            for geom, width in zip(points, topwidth)
        )
    else:
        corridor_shapes = ((geom.buffer(float(fixed_corridor_buffer_m)), 1) for geom in points)
        anchor_shapes = ((geom.buffer(float(fixed_anchor_buffer_m)), 1) for geom in points)

    corridor = rasterize(corridor_shapes, out_shape=dem_shape, transform=transform, all_touched=False).astype(bool)
    anchor = rasterize(anchor_shapes, out_shape=dem_shape, transform=transform, all_touched=False).astype(bool)
    return corridor, anchor

def build_target_coordinates(xs: np.ndarray, ys: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.where(mask)
    return xs[cols].astype(np.float32), ys[rows].astype(np.float32), rows.astype(np.int32), cols.astype(np.int32)

def precompute_idw_neighbors(
    x: np.ndarray,
    y: np.ndarray,
    point_max_distance: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    out_shape: tuple[int, int],
    target_rows: np.ndarray,
    target_cols: np.ndarray,
    k: int,
) -> dict[str, np.ndarray | tuple[int, int]]:
    """Precompute KD-tree neighbors for repeated IDW evaluations."""
    # Query the nearest points once, then record which of those neighbors are
    # actually usable after applying each point's interpolation radius.
    pts = np.column_stack((x, y))
    tree = cKDTree(pts)
    xy = np.column_stack((target_x, target_y))
    global_max_dist = float(np.nanmax(point_max_distance))
    dist, idx = tree.query(xy, k=k, distance_upper_bound=global_max_dist)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    valid_idx = idx < len(x)
    valid_dist = np.isfinite(dist) & valid_idx
    neighbor_radii = np.full(dist.shape, np.nan, dtype=np.float32)
    neighbor_radii[valid_idx] = point_max_distance[idx[valid_idx]]
    usable = valid_dist & (dist <= neighbor_radii)
    return {
        "dist": dist.astype(np.float32),
        "idx": idx.astype(np.int32),
        "usable": usable,
        "rows": target_rows.astype(np.int32),
        "cols": target_cols.astype(np.int32),
        "shape": out_shape,
    }

def idw_from_precomputed(z: np.ndarray, valid_points: np.ndarray, pre: dict, power: float) -> np.ndarray:
    """
    Evaluate IDW interpolation from a precomputed neighbor table.

    This is a direct port of the FHS interpolation routine, including the
    `len(z)` sentinel handling used by `cKDTree.query(..., distance_upper_bound=...)`.
    """
    dist = pre["dist"]
    idx = pre["idx"]
    usable = pre["usable"].copy()
    rows = pre["rows"]
    cols = pre["cols"]
    out_shape = pre["shape"]

    mask_idx = idx < len(z)
    valid_neighbor = np.zeros(idx.shape, dtype=bool)
    valid_neighbor[mask_idx] = valid_points[idx[mask_idx]]
    usable &= valid_neighbor

    # Use standard inverse-distance weights except when a target lands exactly
    # on a point, in which case that point gets all the weight.
    weights = np.zeros(dist.shape, dtype=np.float32)
    positive = usable & (dist > 0)
    weights[positive] = 1.0 / np.power(dist[positive], power)

    exact = usable & (dist == 0)
    exact_rows = np.where(np.any(exact, axis=1))[0]
    for r in exact_rows:
        weights[r, :] = 0.0
        first_exact = np.where(exact[r])[0][0]
        weights[r, first_exact] = 1.0

    z_safe = np.zeros(len(z) + 1, dtype=np.float32)
    z_safe[:len(z)] = z

    weight_sum = weights.sum(axis=1)
    good = weight_sum > 0
    vals = np.full(len(weight_sum), np.nan, dtype=np.float32)
    if np.any(good):
        vals[good] = (
            np.sum(weights[good] * z_safe[idx[good]], axis=1) / weight_sum[good]
        ).astype(np.float32)

    out = np.full(out_shape, np.nan, dtype=np.float32)
    out[rows, cols] = vals
    return out



def summarize_sanity_filter(prefix: str, valid_before: np.ndarray, valid_after: np.ndarray, point_types: np.ndarray | None = None) -> str:
    """Build the same human-readable WSE sanity-filter summary used by FHS."""
    dropped = valid_before & ~valid_after
    total_before = int(np.count_nonzero(valid_before))
    total_after = int(np.count_nonzero(valid_after))
    total_dropped = int(np.count_nonzero(dropped))
    msg = f"{prefix}: kept {total_after} of {total_before} scenario points after WSE sanity filter"
    if total_dropped > 0 and point_types is not None:
        unique, counts = np.unique(point_types[dropped].astype(str), return_counts=True)
        parts = [f"{u}={int(c)}" for u, c in zip(unique, counts)]
        msg += f"; dropped {total_dropped} ({', '.join(parts)})"
    elif total_dropped > 0:
        msg += f"; dropped {total_dropped}"
    return msg


def smooth_nan(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smooth while preserving NaN masks."""
    # Smooth the values and the valid-data mask separately so NaN gaps do not
    # behave like zero-valued water surface cells.
    valid = np.isfinite(arr)
    arr_filled = np.where(valid, arr, 0.0).astype(np.float32)
    num = ndi.gaussian_filter(arr_filled, sigma)
    den = ndi.gaussian_filter(valid.astype(np.float32), sigma)
    out = np.full_like(arr, np.nan, dtype=np.float32)
    mask = den > 1.0e-6
    out[mask] = num[mask] / den[mask]
    return out

def compute_fhs_flood_from_wse(
    wse: np.ndarray,
    dem: np.ndarray,
    corridor: np.ndarray,
    anchor: np.ndarray,
    connectivity: int,
    smooth_sigma_pixels: float,
    permanent_water_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert an interpolated WSE surface to a flood mask using the original FHS
    corridor/anchor/connectivity rules.
    """
    if smooth_sigma_pixels > 0:
        wse = smooth_nan(wse, smooth_sigma_pixels)

    # A cell is initially eligible when the interpolated WSE overtops the DEM
    # inside the corridor. Anchor cells then enforce connected flood regions.
    valid_wse = np.isfinite(wse)
    flood = valid_wse & corridor & (dem < wse)
    flood |= anchor

    structure = ndi.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    flood = ndi.binary_propagation(anchor, mask=flood, structure=structure)

    dry_cells = dem >= wse
    wse = wse.copy()
    wse[dry_cells] = np.nan
    flood[dry_cells] = False

    if permanent_water_mask is not None:
        flood |= permanent_water_mask

    return flood.astype(np.uint8), wse.astype(np.float32)

def create_fhs_flood_map_from_points(
    dem: np.ndarray,
    point_wse: np.ndarray,
    point_rows: np.ndarray | None = None,
    point_cols: np.ndarray | None = None,
    point_x: np.ndarray | None = None,
    point_y: np.ndarray | None = None,
    point_topwidth: np.ndarray | None = None,
    point_ids: np.ndarray | None = None,
    point_dem: np.ndarray | None = None,
    point_types: np.ndarray | None = None,
    point_slope: np.ndarray | None = None,
    dx: float = 1.0,
    dy: float | None = None,
    prefix: str = "default",
    corridor_buffer_m: float = 500.0,
    anchor_buffer_m: float = 5.0,
    use_topwidth_buffers: bool = True,
    corridor_topwidth_factor: float = 1.5,
    anchor_topwidth_factor: float = 0.5,
    connectivity: int = 4,
    k: int = 20,
    power: float = 2.0,
    max_distance_m: float = 1000.0,
    use_topwidth_max_distance: bool = True,
    maxdist_topwidth_factor: float = 1.5,
    min_topwidth_m: float = 5.0,
    max_topwidth_m: float = 500.0,
    fallback_topwidth_m: float = 20.0,
    smooth_sigma_pixels: float = 0.5,
    apply_wse_sanity_filter: bool = True,
    wse_sanity_tolerance_m: float = 0.0,
    permanent_water_mask: np.ndarray | None = None,
    fast_mode: bool = False,
    precomputed_neighbors: dict | None = None,
    precomputed_corridor_mask: np.ndarray | None = None,
    precomputed_anchor_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray | str | bool]:
    """
    Port of the FHS `create_flood_map_from_hwms` single-scenario flood engine.

    Parameters
    ----------
    dem : ndarray
        Unpadded DEM array for the working grid.
    point_wse : ndarray
        Water-surface elevations for one scenario.
    point_rows, point_cols : ndarray, optional
        Zero-based raster indices for the points. Required when `point_x` and
        `point_y` are not provided and also used to sample DEM elevations when
        `point_dem` is omitted.
    point_x, point_y : ndarray, optional
        Point coordinates on a local projected grid measured in meters.
    point_topwidth : ndarray, optional
        Per-point top width values in meters. If omitted, a constant fallback is
        used.
    point_ids : ndarray, optional
        Point IDs used for COMID-wise median interpolation distances.
    point_dem : ndarray, optional
        DEM elevation at the point locations. If omitted and rows/cols are
        available, values are sampled from `dem`.
    point_types : ndarray, optional
        Optional labels used in the sanity-filter summary.
    point_slope : ndarray, optional
        Optional slope values interpolated with the same IDW neighbors so the
        caller can still build velocity products.
    dx, dy : float
        Cell spacing in meters.
    prefix : str, default "default"
        Scenario label used in the returned summary string.
    corridor_buffer_m, anchor_buffer_m : float
        Fixed buffer distances used when top-width buffers are disabled.
    use_topwidth_buffers : bool
        If True, corridor/anchor radii are derived from point top widths.
    corridor_topwidth_factor, anchor_topwidth_factor : float
        Multipliers converting top width to corridor/anchor radii.
    connectivity : int
        Either 4 or 8. Controls raster connectivity during flood propagation.
    k, power : int, float
        IDW nearest-neighbor count and inverse-distance power.
    max_distance_m : float
        Fixed interpolation radius in meters.
    use_topwidth_max_distance : bool
        If True and `point_ids` are available, per-point max distance is the
        COMID-wise median top width times `maxdist_topwidth_factor`.
    min_topwidth_m, max_topwidth_m, fallback_topwidth_m : float
        Top-width clipping and fallback settings.
    smooth_sigma_pixels : float
        Gaussian smoothing applied to the interpolated WSE grid.
    apply_wse_sanity_filter : bool
        If True, drop points whose WSE falls below the DEM at the point.
    wse_sanity_tolerance_m : float
        Non-negative tolerance added before the WSE-vs-DEM sanity comparison.
    permanent_water_mask : ndarray, optional
        Boolean mask of always-wet cells.
    fast_mode : bool
        Mirrors the FHS fast-mode overrides by disabling top-width-based buffers
        and top-width-based max interpolation distance.
    precomputed_neighbors, precomputed_corridor_mask, precomputed_anchor_mask : optional
        Optional reusable state for repeated scenario evaluation. If supplied,
        they are used exactly as in the original FHS fast worker.

    Returns
    -------
    dict
        Keys: `flood`, `wse`, `depth`, `slope`, `stats_message`,
        `valid_point_mask`, and `has_valid_points`.

    Notes
    -----
    This function ports the flood-raster generation part of
    `FHS_FloodMapper_AllInOne.py`. It intentionally does not reimplement the
    original CLI, raster reprojection, or VDT-to-HWM point generation wrapper.
    Those are better handled outside this low-level helper.
    """
    dem = np.asarray(dem, dtype=np.float32)
    if dem.ndim != 2:
        raise ValueError("dem must be a 2D array.")

    # Normalize the caller inputs into one consistent in-memory point
    # representation before any buffering or interpolation begins.
    dy = dx if dy is None else dy
    point_wse = np.asarray(point_wse, dtype=np.float32)
    n_points = len(point_wse)

    if point_x is None or point_y is None:
        if point_rows is None or point_cols is None:
            raise ValueError("Provide either point_x/point_y or point_rows/point_cols.")
        point_rows = np.asarray(point_rows, dtype=np.int32)
        point_cols = np.asarray(point_cols, dtype=np.int32)
        if len(point_rows) != n_points or len(point_cols) != n_points:
            raise ValueError("point_rows and point_cols must match point_wse length.")
        point_x, point_y = _grid_xy_from_rc(point_rows, point_cols, dx, dy)
    else:
        point_x = np.asarray(point_x, dtype=np.float32)
        point_y = np.asarray(point_y, dtype=np.float32)
        if len(point_x) != n_points or len(point_y) != n_points:
            raise ValueError("point_x and point_y must match point_wse length.")
        if point_rows is None or point_cols is None:
            point_rows = np.full(n_points, -1, dtype=np.int32)
            point_cols = np.full(n_points, -1, dtype=np.int32)
        else:
            point_rows = np.asarray(point_rows, dtype=np.int32)
            point_cols = np.asarray(point_cols, dtype=np.int32)

    if point_topwidth is None:
        point_topwidth = np.full(n_points, fallback_topwidth_m, dtype=np.float32)
    else:
        point_topwidth = np.asarray(point_topwidth, dtype=np.float32)
        if len(point_topwidth) != n_points:
            raise ValueError("point_topwidth must match point_wse length.")
        point_topwidth = np.where(np.isfinite(point_topwidth), point_topwidth, fallback_topwidth_m)
    point_topwidth = np.clip(point_topwidth, min_topwidth_m, max_topwidth_m).astype(np.float32)

    if point_ids is not None:
        point_ids = np.asarray(point_ids)
        if len(point_ids) != n_points:
            raise ValueError("point_ids must match point_wse length.")

    if point_types is None:
        point_types = np.full(n_points, "Unknown", dtype=object)
    else:
        point_types = np.asarray(point_types, dtype=object)
        if len(point_types) != n_points:
            raise ValueError("point_types must match point_wse length.")

    if point_slope is not None:
        point_slope = np.asarray(point_slope, dtype=np.float32)
        if len(point_slope) != n_points:
            raise ValueError("point_slope must match point_wse length.")

    if point_dem is None:
        point_dem = np.full(n_points, np.nan, dtype=np.float32)
        inside = (
            (point_rows >= 0) & (point_rows < dem.shape[0]) &
            (point_cols >= 0) & (point_cols < dem.shape[1])
        )
        point_dem[inside] = dem[point_rows[inside], point_cols[inside]]
    else:
        point_dem = np.asarray(point_dem, dtype=np.float32)
        if len(point_dem) != n_points:
            raise ValueError("point_dem must match point_wse length.")

    valid = np.isfinite(point_wse)
    valid_before = valid.copy()
    if apply_wse_sanity_filter:
        valid &= np.isfinite(point_dem)
        valid &= point_wse + wse_sanity_tolerance_m >= point_dem

    stats_message = summarize_sanity_filter(prefix=prefix, valid_before=valid_before, valid_after=valid, point_types=point_types)

    if fast_mode:
        use_topwidth_buffers = False
        use_topwidth_max_distance = False

    nrows, ncols = dem.shape
    transform = Affine(float(dx), 0.0, 0.0, 0.0, -float(dy), 0.0)
    xs = (np.arange(ncols, dtype=np.float32) + np.float32(0.5)) * np.float32(dx)
    ys = -((np.arange(nrows, dtype=np.float32) + np.float32(0.5)) * np.float32(dy))

    # Build or reuse the masks that define where IDW can populate WSE values
    # and where hydraulic connectivity is allowed to start.
    if precomputed_corridor_mask is not None and precomputed_anchor_mask is not None:
        corridor = np.asarray(precomputed_corridor_mask, dtype=bool)
        anchor = np.asarray(precomputed_anchor_mask, dtype=bool)
    else:
        x_valid = point_x[valid]
        y_valid = point_y[valid]
        tw_valid = point_topwidth[valid]
        corridor, anchor = build_variable_buffer_masks_from_points(
            x=x_valid,
            y=y_valid,
            topwidth=tw_valid,
            dem_shape=dem.shape,
            transform=transform,
            fixed_corridor_buffer_m=corridor_buffer_m,
            fixed_anchor_buffer_m=anchor_buffer_m,
            use_topwidth_buffers=use_topwidth_buffers,
            corridor_topwidth_factor=corridor_topwidth_factor,
            anchor_topwidth_factor=anchor_topwidth_factor,
        )

    if not np.any(valid):
        flood = np.zeros(dem.shape, dtype=np.uint8)
        wse = np.full(dem.shape, np.nan, dtype=np.float32)
        depth = np.full(dem.shape, np.nan, dtype=np.float32)
        slope = np.full(dem.shape, np.nan, dtype=np.float32) if point_slope is not None else None
        if permanent_water_mask is not None:
            flood = np.where(permanent_water_mask, 1, flood).astype(np.uint8)
        return {
            "flood": flood,
            "wse": wse,
            "depth": depth,
            "slope": slope,
            "stats_message": stats_message,
            "valid_point_mask": valid,
            "has_valid_points": False,
        }

    if precomputed_neighbors is not None:
        z_all = np.where(valid, point_wse.astype(np.float32), 0.0).astype(np.float32)
        wse = idw_from_precomputed(z_all, valid, precomputed_neighbors, power)
        if point_slope is not None:
            slope_seed = np.where(valid, point_slope.astype(np.float32), 0.0).astype(np.float32)
            slope_interp = idw_from_precomputed(slope_seed, valid, precomputed_neighbors, power)
        else:
            slope_interp = None
    else:
        x_valid = point_x[valid]
        y_valid = point_y[valid]
        z_valid = point_wse[valid].astype(np.float32)
        tw_valid = point_topwidth[valid].astype(np.float32)

        # FHS can scale the interpolation reach by COMID using the median top
        # width, allowing larger rivers to influence a wider neighborhood.
        if point_ids is not None and use_topwidth_max_distance:
            tmp_df = pd.DataFrame({"id_col": point_ids[valid], "tw_based_dist": maxdist_topwidth_factor * tw_valid})
            grouped = tmp_df.groupby("id_col")["tw_based_dist"].median()
            point_max_distance = tmp_df["id_col"].map(grouped).to_numpy(dtype=np.float32)
            bad = ~np.isfinite(point_max_distance)
            point_max_distance[bad] = max_distance_m
        else:
            point_max_distance = np.full(len(z_valid), max_distance_m, dtype=np.float32)

        # Interpolate only within the buffered corridor; cells outside it stay
        # NaN unless they are later forced wet by another rule.
        interp_mask = corridor & np.isfinite(dem)
        target_x, target_y, target_rows, target_cols = build_target_coordinates(xs, ys, interp_mask)
        if len(target_x) == 0:
            wse = np.full(dem.shape, np.nan, dtype=np.float32)
            slope_interp = np.full(dem.shape, np.nan, dtype=np.float32) if point_slope is not None else None
        else:
            pre = precompute_idw_neighbors(
                x=x_valid,
                y=y_valid,
                point_max_distance=point_max_distance,
                target_x=target_x,
                target_y=target_y,
                out_shape=dem.shape,
                target_rows=target_rows,
                target_cols=target_cols,
                k=k,
            )
            valid_points = np.ones(len(z_valid), dtype=bool)
            wse = idw_from_precomputed(z_valid, valid_points, pre, power)
            if point_slope is not None:
                slope_interp = idw_from_precomputed(point_slope[valid].astype(np.float32), valid_points, pre, power)
            else:
                slope_interp = None

    flood, wse = compute_fhs_flood_from_wse(
        wse=wse,
        dem=dem,
        corridor=corridor,
        anchor=anchor,
        connectivity=connectivity,
        smooth_sigma_pixels=smooth_sigma_pixels,
        permanent_water_mask=permanent_water_mask,
    )
    depth = np.where(np.isfinite(wse), np.maximum(wse - dem, 0.0), np.nan).astype(np.float32)

    if slope_interp is not None:
        slope = np.where(flood > 0, slope_interp, np.nan).astype(np.float32)
        slope = np.where(slope <= 0.0, 0.0002, slope).astype(np.float32)
    else:
        slope = None

    return {
        "flood": flood,
        "wse": wse,
        "depth": depth,
        "slope": slope,
        "stats_message": stats_message,
        "valid_point_mask": valid,
        "has_valid_points": True,
    }


def multi_point_interpolation(
    E: np.ndarray,
    B: np.ndarray,
    RR: np.ndarray,
    CC: np.ndarray,
    T_Rast: np.ndarray,
    W_Rast: np.ndarray,
    S_Rast: np.ndarray | None,
    COMID_Unique_TW,
    COMID_Unique_Depth,
    COMID_Unique_Flow: dict,
    CurveParamFileName: str,
    VDTDatabaseFileName: str,
    TW_MultFact: float,
    dx: float,
    dy: float,
    TopWidthPlausibleLimit: float,
    Set_Depth: float,
    flood_vdt_cells: bool,
    mapper_options: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str]:
    """
    Curve2Flood-side wrapper for the FHS FloodMapper workflow.

    This function ports the FHS end-to-end logic that is relevant inside
    Curve2Flood's per-flow event loop:

    1. Optionally rebuild scenario HWM-style centerline points directly from the
       VDT database and the current COMID->flow mapping.
    2. Optionally filter HWM outliers using the FHS MAD/IQR depth filter.
    3. Optionally expand points across the cross section using `XS_Angle`,
       top width, and DEM blocking.
    4. Generate a flood raster via FHS inverse-distance interpolation,
       corridor/anchor buffering, smoothing, and connectivity enforcement.

    Parameters
    ----------
    E, B, RR, CC, T_Rast, W_Rast, S_Rast : ndarray
        The current Curve2Flood padded DEM/stream arrays and per-cell hydraulic
        rasters for the active flow event.
    COMID_Unique_TW, COMID_Unique_Depth : mapping
        Per-COMID fallback top width and depth used when a VDT-driven point set
        cannot be built.
    COMID_Unique_Flow : dict
        Current event COMID->flow mapping.
    CurveParamFileName, VDTDatabaseFileName : str
        Optional curve and VDT source tables. This function prefers the original
        FHS VDT->HWM interpolation path when the VDT database is available.
        Missing `XS_Angle`, `Slope`, or `Elev` values can be backfilled from the
        other table when possible.
    TW_MultFact : float
        Curve2Flood top-width multiplier. This is required so curve-file driven
        scenario generation produces the same scaled top widths as the rest of
        the run.
    dx, dy : float
        Cell spacing in meters on the local working grid.
    TopWidthPlausibleLimit, Set_Depth : float
        Curve2Flood fixed-depth controls used by the fallback centerline-point
        builder when VDT-driven point generation is unavailable.
    flood_vdt_cells : bool
        If True, centerline cells are forced into the returned flood mask.
    mapper_options : dict, optional
        Optional FHS tuning parameters. Supported keys:
        `topwidth_threshold_m`, `xs_point_spacing_m`,
        `remove_hwm_outliers`, `hwm_outlier_method`,
        `hwm_outlier_threshold`, `hwm_outlier_group_by`,
        `hwm_outlier_min_samples`, `corridor_buffer_m`,
        `anchor_buffer_m`, `use_topwidth_buffers`,
        `corridor_topwidth_factor`, `anchor_topwidth_factor`,
        `connectivity`, `k`, `power`, `max_distance_m`,
        `use_topwidth_max_distance`, `maxdist_topwidth_factor`,
        `min_topwidth_m`, `max_topwidth_m`, `fallback_topwidth_m`,
        `smooth_sigma_pixels`, `apply_wse_sanity_filter`,
        `wse_sanity_tolerance_m`, `fast_mode`, and `permanent_water_mask`.

    Returns
    -------
    tuple
        `(flood, depth, slope, stats_message)` for the current flow event.

    Notes
    -----
    The wrapper prefers rebuilding points from the VDT because that most
    closely matches `FHS_FloodMapper_AllInOne.py`. If VDT metadata is missing,
    it is backfilled from the curve file when possible. If a usable VDT table
    is not available, the function builds the scenario table from the curve
    file directly before falling back to the currently-available Curve2Flood
    centerline rasters.
    """
    options = {
        "topwidth_threshold_m": 200.0,
        "xs_point_spacing_m": 100.0,
        "remove_hwm_outliers": True,
        "hwm_outlier_method": "mad",
        "hwm_outlier_threshold": 3.5,
        "hwm_outlier_group_by": "comid",
        "hwm_outlier_min_samples": 5,
        "corridor_buffer_m": 500.0,
        "anchor_buffer_m": 5.0,
        "use_topwidth_buffers": True,
        "corridor_topwidth_factor": 1.0,
        "anchor_topwidth_factor": 0.1,
        "connectivity": 4,
        "k": 12,
        "power": 2.0,
        "max_distance_m": 1000.0,
        "use_topwidth_max_distance": True,
        "maxdist_topwidth_factor": 1.0,
        "min_topwidth_m": 5.0,
        "max_topwidth_m": 500.0,
        "fallback_topwidth_m": 20.0,
        "smooth_sigma_pixels": 0.25,
        "apply_wse_sanity_filter": True,
        "wse_sanity_tolerance_m": 0.0,
        "fast_mode": False,
        "permanent_water_mask": None,
        "one_based_vdt_rc": False,
    }
    if mapper_options:
        options.update({k: v for k, v in mapper_options.items() if v is not None})

    point_data = None
    stats_message = "curve2flood: no scenario points were generated"
    scenario_builder_errors = []

    if Set_Depth <= 0.0:
        # Collect metadata tables up front so any missing Elev, Slope, or
        # XS_Angle fields can be backfilled after a scenario is built.
        metadata_candidates = []
        if VDTDatabaseFileName:
            try:
                metadata_candidates.append(
                    _coerce_fhs_metadata_columns(
                        read_curve2flood_table(VDTDatabaseFileName),
                        VDTDatabaseFileName,
                        one_based_rc=bool(options["one_based_vdt_rc"]),
                    )
                )
            except Exception as ex:
                scenario_builder_errors.append(f"VDT metadata unavailable: {ex}")
        if CurveParamFileName:
            try:
                metadata_candidates.append(
                    _coerce_fhs_metadata_columns(
                        read_curve2flood_table(CurveParamFileName),
                        CurveParamFileName,
                        one_based_rc=bool(options["one_based_vdt_rc"]),
                    )
                )
            except Exception as ex:
                scenario_builder_errors.append(f"Curve metadata unavailable: {ex}")

        scenario_builders = []
        if VDTDatabaseFileName:
            scenario_builders.append((
                "VDT database",
                lambda: build_fhs_scenario_dataframe_from_vdt(
                    vdt_database_filename=VDTDatabaseFileName,
                    comid_unique_flow=COMID_Unique_Flow,
                    one_based_rc=bool(options["one_based_vdt_rc"]),
                ),
            ))
        if CurveParamFileName:
            scenario_builders.append((
                "curve parameter file",
                lambda: build_fhs_scenario_dataframe_from_curve_file(
                    curve_param_filename=CurveParamFileName,
                    comid_unique_flow=COMID_Unique_Flow,
                    tw_multfact=TW_MultFact,
                    one_based_rc=bool(options["one_based_vdt_rc"]),
                ),
            ))

        for source_name, scenario_builder in scenario_builders:
            try:
                # Prefer the VDT-based hydraulic reconstruction, then fall back
                # to the curve file if it is the only viable source.
                scenario_df = scenario_builder()
                for metadata_df in metadata_candidates:
                    scenario_df = merge_missing_fhs_metadata(scenario_df, metadata_df)
                if options["remove_hwm_outliers"]:
                    scenario_df = filter_hwm_outliers_single_scenario(
                        scenario_df,
                        method=str(options["hwm_outlier_method"]),
                        threshold=float(options["hwm_outlier_threshold"]),
                        group_by=str(options["hwm_outlier_group_by"]),
                        min_samples=int(options["hwm_outlier_min_samples"]),
                    )
                if scenario_df.empty:
                    continue
                point_data = create_fhs_points_from_scenario_dataframe(
                    results_df=scenario_df,
                    dem_array=E[1:-1, 1:-1],
                    dx=dx,
                    dy=dy,
                    topwidth_threshold_m=options["topwidth_threshold_m"],
                    xs_point_spacing_m=float(options["xs_point_spacing_m"]),
                )
                break
            except Exception as ex:
                scenario_builder_errors.append(f"{source_name} scenario generation failed: {ex}")

        if point_data is None and scenario_builder_errors:
            LOG.warning(
                "FHS scenario point generation failed; falling back to centerline-only points. Reasons: "
                + " | ".join(scenario_builder_errors)
            )

    if point_data is None:
        # Last resort: use the centerline hydraulics already present inside
        # Curve2Flood instead of an externally rebuilt FHS scenario table.
        point_data = build_fhs_points_from_curve2flood_inputs(
            RR=RR,
            CC=CC,
            T_Rast=T_Rast,
            W_Rast=W_Rast,
            S_Rast=S_Rast,
            E=E,
            B=B,
            COMID_Unique_TW=dict(COMID_Unique_TW),
            COMID_Unique_Depth=dict(COMID_Unique_Depth),
            TopWidthPlausibleLimit=TopWidthPlausibleLimit,
            Set_Depth=Set_Depth,
            dx=dx,
            dy=dy,
        )

    # Once the point cloud is ready, the remaining work is the lower-level FHS
    # raster engine: buffer, interpolate WSE, then derive flood/depth/slope.
    result = create_fhs_flood_map_from_points(
        dem=E[1:-1, 1:-1],
        point_rows=point_data["rows"],
        point_cols=point_data["cols"],
        point_x=point_data["x"],
        point_y=point_data["y"],
        point_wse=point_data["wse"],
        point_topwidth=point_data["topwidth"],
        point_ids=point_data["ids"],
        point_dem=point_data["dem"],
        point_types=point_data["types"],
        point_slope=point_data["slope"] if S_Rast is not None else None,
        dx=dx,
        dy=dy,
        prefix="curve2flood",
        corridor_buffer_m=float(options["corridor_buffer_m"]),
        anchor_buffer_m=float(options["anchor_buffer_m"]),
        use_topwidth_buffers=bool(options["use_topwidth_buffers"]),
        corridor_topwidth_factor=float(options["corridor_topwidth_factor"]),
        anchor_topwidth_factor=float(options["anchor_topwidth_factor"]),
        connectivity=int(options["connectivity"]),
        k=int(options["k"]),
        power=float(options["power"]),
        max_distance_m=float(options["max_distance_m"]),
        use_topwidth_max_distance=bool(options["use_topwidth_max_distance"]),
        maxdist_topwidth_factor=float(options["maxdist_topwidth_factor"]),
        min_topwidth_m=float(options["min_topwidth_m"]),
        max_topwidth_m=float(options["max_topwidth_m"]),
        fallback_topwidth_m=float(options["fallback_topwidth_m"]),
        smooth_sigma_pixels=float(options["smooth_sigma_pixels"]),
        apply_wse_sanity_filter=bool(options["apply_wse_sanity_filter"]),
        wse_sanity_tolerance_m=float(options["wse_sanity_tolerance_m"]),
        permanent_water_mask=options["permanent_water_mask"],
        fast_mode=bool(options["fast_mode"]),
    )
    stats_message = result["stats_message"]

    flood = result["flood"]
    depth = result["depth"]
    slope = result["slope"]

    if flood_vdt_cells:
        # Keep the original centerline cells wet even if interpolation or
        # buffering would otherwise omit some of them.
        valid_rows = point_data["rows"] >= 0
        valid_cols = point_data["cols"] >= 0
        valid_centerline = valid_rows & valid_cols
        flood[point_data["rows"][valid_centerline], point_data["cols"][valid_centerline]] = 1

    return flood, depth, slope, stats_message