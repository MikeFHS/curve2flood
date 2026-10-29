"""Multi-point interpolation flood mapper, ported from FHS_FloodMapper_AllInOne.

Water-surface points are placed on the stream cells (and, for wide rivers,
across their cross sections), interpolated by inverse-distance weighting
inside a buffered corridor around the points, and the cells below that surface
that connect to the points are flooded.  Point coordinates are metres on a
local grid whose origin is the DEM's top-left corner.  Ground at or below
``NO_GROUND`` (the DEM's -9999, or the padding around it) is no-data: it never
floods and blocks the points across a cross section.
"""
from pathlib import Path

import numba
import numpy as np
import pandas as pd
from numba import njit, prange
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from curve2flood._log import LOG
from curve2flood._patches import patches_holding
from curve2flood.config import is_parquet

# Cells interpolated at a time: each keeps its k nearest points meanwhile.  The KD-tree is queried with as many
# threads as numba has (NUMBA_NUM_THREADS)
_IDW_TARGETS = 1 << 20
GeoTransform = tuple[float, float, float, float, float, float]

# Ground at or below this is no-data, as the other mappers take it
NO_GROUND = -9998.0
# A water surface this far above a cross-section point's ground is worth a warning
_DEEP_POINT_M = 20.0
# A point's buffer is a regular polygon of this many sides with a vertex at angle 0, as shapely's Point.buffer
# makes it with 16 segments a quarter circle
_BUFFER_SIDES = 64

_METADATA_COLUMNS = ["COMID", "Row", "Col", "Elev", "Slope", "XS_Angle"]
_SCENARIO_COLUMNS = _METADATA_COLUMNS + ["Flow", "TopWidth_m", "WaterSurfaceElev_m"]


def _empty_scenario() -> pd.DataFrame:
    return pd.DataFrame(columns=_SCENARIO_COLUMNS)


def _coerce_fhs_metadata_columns(
    table_df: pd.DataFrame,
    table_filename: str,
    one_based_rc: bool = False,
) -> pd.DataFrame:
    """Reduce a VDT or curve table to ``COMID, Row, Col, Elev, Slope, XS_Angle``, keeping its index.

    ``Elev`` is the table's ``Elev``, ``DEM_Elev`` or ``BaseElev``; missing
    optional columns are NaN, and rows without a COMID, row or column are
    dropped.  ``one_based_rc`` converts 1-based rows and columns to 0-based.
    """
    missing_required = {"COMID", "Row", "Col"} - set(table_df.columns)
    if missing_required:
        raise ValueError(f"{Path(table_filename).name} is missing required columns: {sorted(missing_required)}")

    def numeric(column: str) -> pd.Series:
        if column in table_df.columns:
            return pd.to_numeric(table_df[column], errors="coerce")
        return pd.Series(np.nan, index=table_df.index)

    elev = next((c for c in ("Elev", "DEM_Elev", "BaseElev") if c in table_df.columns), "Elev")
    work_df = pd.DataFrame({"COMID": numeric("COMID"), "Row": numeric("Row"), "Col": numeric("Col"),
                            "Elev": numeric(elev), "Slope": numeric("Slope"), "XS_Angle": numeric("XS_Angle")})
    work_df = work_df.dropna(subset=["COMID", "Row", "Col"])
    work_df = work_df.astype({"COMID": np.int64, "Row": np.int64, "Col": np.int64})
    if one_based_rc:
        work_df["Row"] -= 1
        work_df["Col"] -= 1

    return work_df


def merge_missing_fhs_metadata(
    scenario_df: pd.DataFrame,
    metadata_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Fill missing ``Elev``, ``Slope`` and ``XS_Angle`` values from another table, matched on COMID, Row and Col."""
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


def filter_hwm_outliers_single_scenario(
    results_df: pd.DataFrame,
    method: str = "mad",
    threshold: float = 3.5,
    group_by: str = "comid",
    min_samples: int = 5,
) -> pd.DataFrame:
    """Drop the points whose depth (``WaterSurfaceElev_m - Elev``) is an outlier, and those without a WSE.

    Outliers are found over all points (``group_by="global"``) or within each
    reach (``"comid"``), among the points with a depth; a group with fewer
    than ``min_samples`` of them, or fewer than four, is kept as it is.  The
    tests are a robust z-score, ``0.6745 (depth - median) / MAD`` beyond
    ``threshold`` (``"mad"``), or beyond ``threshold`` interquartile ranges
    outside the quartiles (``"iqr"``); a group whose MAD or IQR is about 0
    has no outliers.
    """
    group_by = str(group_by).lower()
    if group_by not in {"global", "comid"}:
        raise ValueError("group_by must be either 'global' or 'comid'.")
    method = str(method).lower()
    if method not in {"mad", "iqr"}:
        raise ValueError(f"Unsupported HWM outlier method: {method}")

    if "Elev" not in results_df.columns or "WaterSurfaceElev_m" not in results_df.columns:
        return results_df.copy()

    elev = results_df["Elev"].to_numpy(dtype=np.float64)
    wse = results_df["WaterSurfaceElev_m"].to_numpy(dtype=np.float64)
    valid = np.isfinite(wse) & np.isfinite(elev)
    outlier = np.zeros(len(results_df), dtype=bool)
    if valid.any():
        depth = pd.Series(wse[valid] - elev[valid])
        groups = (np.zeros(depth.size, dtype=np.int64) if group_by == "global"
                  else results_df["COMID"].to_numpy()[valid])
        grouped = depth.groupby(groups)
        tested = grouped.transform("size").to_numpy() >= max(int(min_samples), 4)
        if method == "mad":
            center = grouped.transform("median")
            mad = (depth - center).abs().groupby(groups).transform("median")
            spread = mad.to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                found = np.abs(0.6745 * (depth - center).to_numpy() / spread) > float(threshold)
        else:
            q1 = grouped.transform("quantile", q=0.25).to_numpy()
            q3 = grouped.transform("quantile", q=0.75).to_numpy()
            spread = q3 - q1
            values = depth.to_numpy()
            found = (values < q1 - float(threshold) * spread) | (values > q3 + float(threshold) * spread)
        outlier[np.flatnonzero(valid)] = tested & (spread > 1e-12) & found

    return results_df.loc[np.isfinite(wse) & ~outlier].copy()


def read_curve2flood_table(table_filename: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a CSV, TSV or Parquet table, or only its ``columns`` that it has, numbering its rows from 0."""
    if not table_filename:
        raise ValueError("A table filename is required.")
    if is_parquet(table_filename):
        if columns is not None:
            import pyarrow.parquet as pq

            names = set(pq.read_schema(table_filename).names)
            columns = [c for c in columns if c in names]
        return pd.read_parquet(table_filename, columns=columns).reset_index(drop=True)
    sep = "\t" if str(table_filename).lower().endswith(".tsv") else ","
    usecols = None if columns is None else (lambda c: c in columns)
    return pd.read_csv(table_filename, sep=sep, usecols=usecols)


def get_curve_columns(columns) -> tuple[list[str], list[str], list[str], list[str]]:
    """The VDT's ``q_*``, ``v_*``, ``t_*`` and ``wse_*`` columns, each sorted by number."""
    q_cols = sorted([c for c in columns if str(c).startswith("q_")], key=lambda x: int(str(x).split("_")[1]))
    v_cols = sorted([c for c in columns if str(c).startswith("v_")], key=lambda x: int(str(x).split("_")[1]))
    t_cols = sorted([c for c in columns if str(c).startswith("t_")], key=lambda x: int(str(x).split("_")[1]))
    wse_cols = sorted([c for c in columns if str(c).startswith("wse_")], key=lambda x: int(str(x).split("_")[1]))
    if not len(q_cols) == len(v_cols) == len(t_cols) == len(wse_cols):
        raise ValueError("Mismatch in number of q_, v_, t_, and wse_ columns.")
    if len(q_cols) == 0:
        raise ValueError("No q_, v_, t_, wse_ columns found in the VDT file.")
    return q_cols, v_cols, t_cols, wse_cols


@njit(cache=True, parallel=True)
def _clean_and_interp_rows(target_q, q, v, t, wse, nodata):
    """Each VDT row's velocity, top width and WSE at its flow ``target_q``: ``(velocity, top_width, wse)`` arrays.

    A row's invalid entries (no positive flow or top width, a negative
    velocity, a no-data WSE at or below ``nodata``) are dropped first; the
    rest are interpolated linearly by flow and held at their end values
    beyond them.  A row with nothing left, or no flow, is NaN.
    """
    n, m = q.shape
    out_v = np.full(n, np.nan)
    out_t = np.full(n, np.nan)
    out_wse = np.full(n, np.nan)
    for i in prange(n):
        x = target_q[i]
        if not np.isfinite(x):
            continue
        # The valid entries, sorted by flow (stably), the first of equal flows kept
        xs = np.empty(m)
        vs = np.empty(m)
        ts = np.empty(m)
        ws = np.empty(m)
        k = 0
        for j in range(m):
            qj, vj, tj, wj = q[i, j], v[i, j], t[i, j], wse[i, j]
            if not (np.isfinite(qj) and np.isfinite(vj) and np.isfinite(tj) and np.isfinite(wj)):
                continue
            if not (qj > 0.0 and vj >= 0.0 and tj > 0.0 and wj > nodata):
                continue
            at = k
            while at > 0 and xs[at - 1] > qj:
                xs[at], vs[at], ts[at], ws[at] = xs[at - 1], vs[at - 1], ts[at - 1], ws[at - 1]
                at -= 1
            xs[at], vs[at], ts[at], ws[at] = qj, vj, tj, wj
            k += 1
        if k == 0:
            continue
        unique = 1
        for j in range(1, k):
            if xs[j] != xs[unique - 1]:
                xs[unique], vs[unique], ts[unique], ws[unique] = xs[j], vs[j], ts[j], ws[j]
                unique += 1
        out_v[i] = _clamped(x, xs, vs, unique)
        out_t[i] = _clamped(x, xs, ts, unique)
        out_wse[i] = _clamped(x, xs, ws, unique)

    return out_v, out_t, out_wse


@njit(cache=True, inline="always")
def _clamped(x, xp, fp, n):
    """Linear interpolation of ``fp(xp)`` (their first ``n``, ``xp`` ascending) at ``x``, holding the end values
    outside ``xp``."""
    if n == 1 or x <= xp[0]:
        return fp[0]
    if x >= xp[n - 1]:
        return fp[n - 1]
    idx = np.searchsorted(xp[:n], x)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]
    if x1 == x0:
        return y0
    w = (x - x0) / (x1 - x0)
    return y0 + w * (y1 - y0)


def _interpolate_curves(table: pd.DataFrame, work_df: pd.DataFrame) -> pd.DataFrame:
    """``work_df``'s rows (with a ``Flow``) with the top width and WSE that ``table``'s VDT-style curves give at their
    flows; rows without either are dropped."""
    q_cols, v_cols, t_cols, wse_cols = get_curve_columns(table.columns)
    rows = table.loc[work_df.index]
    curves = [rows[cols].to_numpy(dtype=np.float64) for cols in (q_cols, v_cols, t_cols, wse_cols)]
    _, top_width, wse = _clean_and_interp_rows(work_df["Flow"].to_numpy(dtype=np.float64), *curves, NO_GROUND)
    work_df = work_df.assign(TopWidth_m=top_width, WaterSurfaceElev_m=wse)
    return work_df.dropna(subset=["TopWidth_m", "WaterSurfaceElev_m"])[_SCENARIO_COLUMNS]


def _with_flows(work_df: pd.DataFrame, comid_unique_flow: dict) -> pd.DataFrame:
    """``work_df`` with each row's reach's flow, without the rows whose reach has none or none above 0."""
    flow = pd.to_numeric(work_df["COMID"].map(comid_unique_flow), errors="coerce")
    work_df = work_df.assign(Flow=flow)
    return work_df[work_df["Flow"] > 0]


def build_fhs_scenario_dataframe_from_vdt(
    vdt_database_filename: str,
    comid_unique_flow: dict,
    one_based_rc: bool = False,
) -> pd.DataFrame:
    """One water-surface point per VDT stream cell at its reach's flow.

    Returns ``COMID, Row, Col, Elev, Slope, XS_Angle, Flow, TopWidth_m,
    WaterSurfaceElev_m``; cells without a flow above 0 or a valid curve are
    dropped.  The top widths are the VDT's: the mapper scales them with its
    own settings (``MPI_*_TopWidth_Factor``), not ``TW_MultFact``.
    """
    if not vdt_database_filename:
        raise ValueError("A VDT database is required for FHS VDT point generation.")

    vdt_df = read_curve2flood_table(vdt_database_filename)
    if vdt_df.empty:
        raise ValueError("The VDT database is empty.")

    work_df = _with_flows(_coerce_fhs_metadata_columns(vdt_df, vdt_database_filename, one_based_rc),
                          comid_unique_flow)
    if work_df.empty:
        return _empty_scenario()

    return _interpolate_curves(vdt_df, work_df)


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the rows more than two standard deviations from their reach's mean, for TopWidth, then WSE, then
    Velocity.

    The reaches are the ``COMID`` column's, or the whole table without one.
    A reach with a single row has no spread and keeps it.  Rows whose value
    is NaN are dropped.
    """
    for col in ('TopWidth', 'WSE', 'Velocity'):
        values = df[col]
        if 'COMID' in df.columns:
            grouped = values.groupby(df['COMID'])
            mean, std = grouped.transform('mean'), grouped.transform('std').fillna(0.0)
        else:
            mean, std = values.mean(), values.std()
            std = 0.0 if np.isnan(std) else std
        df = df[(values >= mean - 2 * std) & (values <= mean + 2 * std)]

    return df


def build_fhs_scenario_dataframe_from_curve_file(
    curve_param_filename: str,
    comid_unique_flow: dict,
    one_based_rc: bool = False,
) -> pd.DataFrame:
    """One water-surface point per curve-file stream cell at its reach's flow.

    Curve files with VDT-style ``q_*``/``v_*``/``t_*``/``wse_*`` columns are
    interpolated like a VDT; otherwise the ``depth_*``, ``tw_*`` and ``vel_*``
    power laws are evaluated and outliers dropped (``filter_outliers``).
    Returns the same columns as ``build_fhs_scenario_dataframe_from_vdt``,
    whose top widths, likewise, aren't scaled.
    """
    if not curve_param_filename:
        raise ValueError("A curve-parameter file is required for curve-driven FHS point generation.")

    curve_df = read_curve2flood_table(curve_param_filename)
    if curve_df.empty:
        raise ValueError("The curve-parameter file is empty.")

    work_df = _with_flows(_coerce_fhs_metadata_columns(curve_df, curve_param_filename, one_based_rc),
                          comid_unique_flow)
    if work_df.empty:
        return _empty_scenario()

    has_vdt_style_curve = all(any(str(c).startswith(prefix) for c in curve_df.columns)
                              for prefix in ("q_", "v_", "t_", "wse_"))
    if has_vdt_style_curve:
        # Some curve files already contain VDT-style hydraulic vectors, so
        # route them through the same interpolation logic as a true VDT table.
        return _interpolate_curves(curve_df, work_df)

    required_curve_cols = {"depth_a", "depth_b", "tw_a", "tw_b", "vel_a", "vel_b"}
    missing_curve_cols = required_curve_cols - set(curve_df.columns)
    if missing_curve_cols:
        raise ValueError(
            f"{Path(curve_param_filename).name} is missing curve columns {sorted(missing_curve_cols)} "
            "and does not expose VDT-style q_/v_/t_/wse_ fields."
        )

    # One hydraulic state per row from the power-law curve coefficients at the event's flow
    coefficients = curve_df.loc[work_df.index, sorted(required_curve_cols)].apply(pd.to_numeric, errors="coerce")
    base = next((c for c in ("BaseElev", "Elev", "DEM_Elev") if c in curve_df.columns), None)
    flow = work_df["Flow"]
    states = pd.DataFrame({
        "COMID": work_df["COMID"],
        "Depth": coefficients["depth_a"] * flow ** coefficients["depth_b"],
        "TopWidth": coefficients["tw_a"] * flow ** coefficients["tw_b"],
        "Velocity": coefficients["vel_a"] * flow ** coefficients["vel_b"],
        "BaseElev": (pd.to_numeric(curve_df.loc[work_df.index, base], errors="coerce") if base
                     else pd.Series(np.nan, index=work_df.index)),
    })
    states = states[(states["Depth"] > 0) & (states["TopWidth"] > 0)]
    states = filter_outliers(states.assign(WSE=states["Depth"] + states["BaseElev"]))
    if states.empty:
        return _empty_scenario()

    work_df = work_df.loc[states.index]
    work_df["TopWidth_m"] = states["TopWidth"].to_numpy(dtype=np.float32)
    work_df["WaterSurfaceElev_m"] = states["WSE"].to_numpy(dtype=np.float64)
    return work_df[_SCENARIO_COLUMNS]


def _ground_at_cells(dem: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """The DEM's ground at cells ``(rows, cols)``, float32: NaN outside the DEM and where it has no ground."""
    ground = np.full(rows.size, np.nan, dtype=np.float32)
    inside = (rows >= 0) & (rows < dem.shape[0]) & (cols >= 0) & (cols < dem.shape[1])
    values = dem[rows[inside], cols[inside]].astype(np.float32)
    ground[inside] = np.where(values > NO_GROUND, values, np.float32(np.nan))
    return ground


def _fill_missing_ground(scenario_df: pd.DataFrame, dem: np.ndarray) -> pd.DataFrame:
    """``scenario_df`` with each point's missing ``Elev`` taken from the DEM at its cell."""
    missing = scenario_df["Elev"].isna().to_numpy()
    if not missing.any():
        return scenario_df
    rows = scenario_df["Row"].to_numpy(np.int64)[missing]
    cols = scenario_df["Col"].to_numpy(np.int64)[missing]
    elev = scenario_df["Elev"].to_numpy(dtype=np.float64, copy=True)
    elev[missing] = _ground_at_cells(dem, rows, cols)
    return scenario_df.assign(Elev=elev)


def _xs_unit_vectors(xs_angle: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors along the cross sections, in the local grid, from the VDT's ``XS_Angle`` (radians)."""
    angle = np.mod(np.asarray(xs_angle, dtype=np.float64), np.pi)
    vx = float(dx) * np.cos(angle)
    vy = -float(dy) * np.sin(angle)
    norm = np.hypot(vx, vy)
    return vx / norm, vy / norm


def create_fhs_points_from_scenario_dataframe(
    results_df: pd.DataFrame,
    dem_array: np.ndarray,
    dx: float,
    dy: float,
    topwidth_threshold_m: float | None = None,
    xs_point_spacing_m: float = 100.0,
) -> dict[str, np.ndarray]:
    """Turn a scenario table into the point arrays ``create_fhs_flood_map_from_points`` takes.

    Every row is a centreline point, whose ground is its ``Elev``.  Where the
    top width is above ``topwidth_threshold_m`` and ``XS_Angle`` is known,
    points are also added across the cross section every
    ``xs_point_spacing_m``, on both sides, until the ground rises to the water
    (or has no data) or half the top width is reached.
    """
    if xs_point_spacing_m <= 0:
        raise ValueError(f"The cross-section point spacing must be positive, not {xs_point_spacing_m}.")
    rows = results_df["Row"].to_numpy(dtype=np.int32)
    cols = results_df["Col"].to_numpy(dtype=np.int32)
    x, y = _grid_xy_from_rc(rows, cols, dx, dy)
    wse = results_df["WaterSurfaceElev_m"].to_numpy(dtype=np.float64)
    topwidth = results_df["TopWidth_m"].to_numpy(dtype=np.float64)
    ids = results_df["COMID"].to_numpy(dtype=np.int64)
    slope = (results_df["Slope"].to_numpy(dtype=np.float32) if "Slope" in results_df.columns
             else np.full(len(results_df), np.nan, dtype=np.float32))
    point_data = {
        "rows": rows,
        "cols": cols,
        "x": x,
        "y": y,
        "wse": wse.astype(np.float32),
        "topwidth": topwidth.astype(np.float32),
        "dem": results_df["Elev"].to_numpy(dtype=np.float32),
        "ids": ids,
        "types": np.full(len(results_df), "Centerline", dtype=object),
        "slope": slope,
    }

    if topwidth_threshold_m is None or "XS_Angle" not in results_df.columns or results_df.empty:
        return point_data

    # March outward from the centerline along each wide river's cross section
    xs_angle = results_df["XS_Angle"].to_numpy(dtype=np.float64)
    wide = np.isfinite(xs_angle) & np.isfinite(topwidth) & np.isfinite(wse) & (topwidth > float(topwidth_threshold_m))
    if not wide.any():
        return point_data
    ux, uy = _xs_unit_vectors(xs_angle[wide], dx, dy)
    args = (x[wide], y[wide], ux, uy, 0.5 * topwidth[wide], wse[wide], float(xs_point_spacing_m),
            np.asarray(dem_array), float(dx), float(dy), NO_GROUND)
    n = _cross_section_points(*args, np.empty(0, np.int64), np.empty(0, np.float32), np.empty(0, np.float32),
                              np.empty(0, np.float32))
    if n == 0:
        return point_data
    owner = np.empty(n, np.int64)
    extra_x = np.empty(n, np.float32)
    extra_y = np.empty(n, np.float32)
    extra_dem = np.empty(n, np.float32)
    _cross_section_points(*args, owner, extra_x, extra_y, extra_dem)
    owner = np.flatnonzero(wide)[owner]

    extra_wse = wse[owner]
    deep = int(np.count_nonzero(extra_wse - extra_dem > _DEEP_POINT_M))
    if deep:
        LOG.warning(f"{deep} of the {n} cross-section points are more than {_DEEP_POINT_M:g} m below the water.")
    for key, arr in {
        "rows": np.full(n, -1, dtype=np.int32),
        "cols": np.full(n, -1, dtype=np.int32),
        "x": extra_x,
        "y": extra_y,
        "wse": extra_wse.astype(np.float32),
        "topwidth": topwidth[owner].astype(np.float32),
        "dem": extra_dem,
        "ids": ids[owner],
        "types": np.full(n, "XS_Extension", dtype=object),
        "slope": slope[owner],
    }.items():
        point_data[key] = np.concatenate((point_data[key], arr))

    return point_data


@njit(cache=True)
def _cross_section_points(x, y, ux, uy, half_width, wse, spacing, dem, dx, dy, nodata, owner, px_out, py_out,
                          ground_out):
    """The points across each cross section ``i``, from ``(x[i], y[i])`` along ``(ux[i], uy[i])``: every
    ``spacing`` metres out to ``half_width[i]``, on the positive side and then the negative, until that side's
    ground is at or above ``wse[i]`` or has no data.  Returns how many there are, and writes them (whose, where,
    and their ground) into the arrays when they are that long, in their order.  The coordinates are float32, as
    each offset added to its float32 centre."""
    write = owner.size > 0
    nrows, ncols = dem.shape
    n = 0
    for i in range(x.size):
        offset = spacing
        blocked = np.zeros(2, dtype=np.bool_)
        while offset <= half_width[i] + 1e-9:
            for side in range(2):
                if blocked[side]:
                    continue
                sign = 1.0 if side == 0 else -1.0
                px = np.float32(x[i] + np.float32(sign * offset * ux[i]))
                py = np.float32(y[i] + np.float32(sign * offset * uy[i]))
                col = int(np.floor(np.float64(px) / dx))
                row = int(np.floor(-np.float64(py) / dy))
                ground = np.nan
                if 0 <= row < nrows and 0 <= col < ncols:
                    value = np.float64(dem[row, col])
                    if np.isfinite(value) and value > nodata:
                        ground = value
                if not ground == ground or wse[i] <= ground:     # noqa: PLR0124, SIM201
                    blocked[side] = True
                    continue
                if write:
                    owner[n] = i
                    px_out[n] = px
                    py_out[n] = py
                    ground_out[n] = ground
                n += 1
            if blocked[0] and blocked[1]:
                break
            offset += spacing

    return n


def _grid_xy_from_rc(rows: np.ndarray, cols: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre coordinates in metres from the DEM's top-left corner; y is negative downward."""
    x = (cols.astype(np.float32) + np.float32(0.5)) * np.float32(dx)
    y = -((rows.astype(np.float32) + np.float32(0.5)) * np.float32(dy))
    return x.astype(np.float32), y.astype(np.float32)


def build_fhs_points_from_stream_cells(
    E: np.ndarray,
    stream_ids: np.ndarray,
    RR: np.ndarray,
    CC: np.ndarray,
    Set_Depth: float,
    TopWidthPlausibleLimit: float,
    dx: float,
    dy: float,
) -> dict[str, np.ndarray]:
    """``Set_Depth``'s points: one per stream cell with ground, its water ``Set_Depth`` above the DEM, its top
    width ``TopWidthPlausibleLimit``.

    ``RR``/``CC`` index the stream cells of the padded DEM ``E``, and
    ``stream_ids`` are their reach ids, in the same order.  Returns the point
    arrays ``create_fhs_flood_map_from_points`` takes.
    """
    ground = E[RR, CC].astype(np.float32)
    wse = (ground.astype(np.float64) + float(Set_Depth)).astype(np.float32)
    keep = np.isfinite(ground) & (ground > NO_GROUND) & (wse.astype(np.float64) - ground >= 0.001)
    rows = (RR[keep] - 1).astype(np.int32)
    cols = (CC[keep] - 1).astype(np.int32)
    x, y = _grid_xy_from_rc(rows, cols, dx, dy)
    n = rows.size
    return {
        "rows": rows,
        "cols": cols,
        "x": x,
        "y": y,
        "wse": wse[keep],
        "topwidth": np.full(n, TopWidthPlausibleLimit, dtype=np.float32),
        "dem": ground[keep],
        "ids": np.asarray(stream_ids)[keep].astype(np.int64),
        "types": np.full(n, "Centerline", dtype=object),
        "slope": np.full(n, np.nan, dtype=np.float32),
    }


def build_variable_buffer_masks_from_points(
    x: np.ndarray,
    y: np.ndarray,
    topwidth: np.ndarray,
    dem_shape: tuple[int, int],
    transform: GeoTransform,
    fixed_corridor_buffer_m: float,
    fixed_anchor_buffer_m: float,
    use_topwidth_buffers: bool = True,
    corridor_topwidth_factor: float = 1.5,
    anchor_topwidth_factor: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterised buffers around the points: ``(corridor, anchor)`` boolean masks.

    The corridor is where WSE is interpolated, and the anchor is where the
    connected flood starts.  Buffer radii are the top width times a factor, or
    fixed distances when ``use_topwidth_buffers`` is False.  Each buffer is
    shapely's ``Point.buffer`` (a 64-sided polygon), and a cell is inside when
    its centre is, as GDAL rasterizes a polygon (see ``_paint_buffers``).
    """
    corridor = np.zeros(dem_shape, dtype=bool)
    anchor = np.zeros(dem_shape, dtype=bool)
    if len(x) == 0:
        return corridor, anchor

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if use_topwidth_buffers:
        width = np.maximum(np.asarray(topwidth), np.float32(0.0))
        corridor_radius = np.asarray(width * corridor_topwidth_factor, dtype=np.float64)
        anchor_radius = np.asarray(width * anchor_topwidth_factor, dtype=np.float64)
    else:
        corridor_radius = np.full(x.size, float(fixed_corridor_buffer_m))
        anchor_radius = np.full(x.size, float(fixed_anchor_buffer_m))

    cell = (float(transform[1]), float(transform[5]))
    _paint_buffers(corridor, x, y, corridor_radius, *cell)
    _paint_buffers(anchor, x, y, anchor_radius, *cell)
    return corridor, anchor


# Rows of cells whose buffers' spans _paint_buffers gathers at a time, in a thread of their own
_PAINT_BAND = 64


@njit(cache=True, parallel=True)
def _paint_buffers(mask, x, y, radius, pixel_width, pixel_height):
    """Set ``mask`` True at the cells whose centres are inside any point's buffer, as GDAL's ``RasterizeLayer``
    burns shapely's buffers on the grid ``(0, pixel_width, 0, 0, 0, pixel_height)``.

    A buffer is the polygon of ``_BUFFER_SIDES`` vertices ``(x + r cos a,
    y + r sin a)`` at ``a = -k 2 pi / _BUFFER_SIDES`` (GEOS's circle).  Each
    row of cells is a scan line through their centres: it crosses an edge
    whose lower end it is at or above and whose upper end it is below, and
    a cell is burned where its centre lies between the two crossings, as
    GDAL's filled-polygon scan rounds them.  The buffers of points along a
    river overlap hundreds of times over, so each band of ``_PAINT_BAND``
    rows keeps, for every column a span starts at, the farthest it reaches,
    and writes the union of the spans once.
    """
    nrows, ncols = mask.shape
    sides = _BUFFER_SIDES
    increment = (2.0 * np.pi) / sides
    cos_a = np.empty(sides)
    sin_a = np.empty(sides)
    for k in range(sides):
        angle = -k * increment
        cos_a[k] = np.cos(angle)
        sin_a[k] = np.sin(angle)
    inverse_x = 1.0 / pixel_width
    inverse_y = 1.0 / pixel_height
    # The polygon's two sides, from its top vertex (a = -3 pi / 2) to its bottom one (a = -pi / 2), each a chain
    # of vertices whose v only grows (or only falls) along it, so that a scan line crosses one edge of each
    half = sides // 2
    right = np.empty(half + 1, dtype=np.int64)
    left = np.empty(half + 1, dtype=np.int64)
    for t in range(half + 1):
        right[t] = (3 * sides // 4 + t) % sides
        left[t] = 3 * sides // 4 - t
    if inverse_y > 0.0:
        right, left = right[::-1].copy(), left[::-1].copy()

    # The rows each buffer may cover, its points sorted by the first
    first = np.full(x.size, nrows, dtype=np.int64)
    last = np.full(x.size, -1, dtype=np.int64)
    for i in range(x.size):
        r = radius[i]
        if r > 0.0:
            # The polygon's vertices reach y + r at a = -3 pi / 2 and y - r at a = -pi / 2, the extremes of v
            top, bottom = (y[i] + r * sin_a[3 * sides // 4]) * inverse_y, (y[i] + r * sin_a[sides // 4]) * inverse_y
            first[i] = int(np.floor(min(top, bottom)))
            last[i] = int(np.floor(max(top, bottom)))
    order = np.argsort(first, kind="mergesort")
    sorted_first = first[order]
    longest = 0
    for i in range(x.size):
        longest = max(longest, last[i] - first[i])

    for band in prange((nrows + _PAINT_BAND - 1) // _PAINT_BAND):
        r0 = band * _PAINT_BAND
        r1 = min(r0 + _PAINT_BAND, nrows)
        # reach[row, c]: the end (exclusive) of the farthest span starting at column c
        reach = np.zeros((r1 - r0, ncols + 1), dtype=np.int64)
        us = np.empty(sides)
        vs = np.empty(sides)
        lo = np.searchsorted(sorted_first, r0 - longest)
        hi = np.searchsorted(sorted_first, r1)
        for at in range(lo, hi):
            i = order[at]
            if last[i] < r0:
                continue
            r = radius[i]
            for k in range(sides):
                us[k] = (x[i] + r * cos_a[k]) * inverse_x
                vs[k] = (y[i] + r * sin_a[k]) * inverse_y
            for row in range(max(first[i], r0), min(last[i], r1 - 1) + 1):
                line = row + 0.5
                a = _chain_crossing(us, vs, right, line)
                b = _chain_crossing(us, vs, left, line)
                if a != a or b != b:    # noqa: PLR0124
                    continue
                start = max(int(np.floor(min(a, b) + 0.5)), 0)
                end = min(int(np.floor(max(a, b) + 0.5)), ncols)
                if start < end and end > reach[row - r0, start]:
                    reach[row - r0, start] = end
        for row in range(r0, r1):
            end = 0
            for c in range(ncols):
                end = max(end, reach[row - r0, c])
                if c < end:
                    mask[row, c] = True


@njit(cache=True, inline="always")
def _chain_crossing(us, vs, chain, line):
    """Where the scan line ``line`` crosses a chain of the polygon's vertices (their indices, ``vs`` ascending along
    it), or NaN: on its edge whose lower end the line is at or below and whose upper end it is above, worked out as
    GDAL works it out."""
    n = chain.size
    if not vs[chain[0]] <= line < vs[chain[n - 1]]:
        return np.nan
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if vs[chain[mid]] <= line:
            lo = mid
        else:
            hi = mid
    v1, v2 = vs[chain[lo]], vs[chain[hi]]
    u1, u2 = us[chain[lo]], us[chain[hi]]
    return (line - v1) * (u2 - u1) / (v2 - v1) + u1


def build_target_coordinates(xs: np.ndarray, ys: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Coordinates, rows and columns of the cells where ``mask`` is True."""
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
    tree: cKDTree | None = None,
    workers: int = 1,
) -> dict[str, np.ndarray | tuple[int, int]]:
    """The ``k`` nearest points of each target cell, for ``idw_values``.

    A point is usable for a cell only within its own ``point_max_distance``.
    ``tree``, a KD-tree of the points, may be made once for many calls, and
    ``workers`` threads query it: each cell's points are its own.
    """
    if tree is None:
        tree = cKDTree(np.column_stack((x, y)))
    xy = np.column_stack((target_x, target_y))
    global_max_dist = float(np.nanmax(point_max_distance))
    dist, idx = tree.query(xy, k=k, distance_upper_bound=global_max_dist, workers=workers)
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


def idw_values(z: np.ndarray, pre: dict, power: float) -> np.ndarray:
    """Inverse-distance-weighted values of ``z`` (weights ``1 / distance**power``) at ``pre``'s target cells, in
    their order (see ``precompute_idw_neighbors``).

    A cell exactly on a point takes that point's value; cells with no usable
    point are NaN.
    """
    dist = pre["dist"]
    idx = pre["idx"]
    usable = pre["usable"]

    # Use standard inverse-distance weights except when a target lands exactly
    # on a point, in which case that point gets all the weight.
    weights = np.zeros(dist.shape, dtype=np.float32)
    positive = usable & (dist > 0)
    weights[positive] = 1.0 / np.power(dist[positive], power)

    exact = usable & (dist == 0)
    exact_rows = np.flatnonzero(exact.any(axis=1))
    if exact_rows.size:
        weights[exact_rows] = 0.0
        weights[exact_rows, exact[exact_rows].argmax(axis=1)] = 1.0

    z_safe = np.zeros(len(z) + 1, dtype=np.float32)
    z_safe[:len(z)] = z

    weight_sum = weights.sum(axis=1)
    good = weight_sum > 0
    vals = np.full(len(weight_sum), np.nan, dtype=np.float32)
    if np.any(good):
        vals[good] = (
            np.sum(weights[good] * z_safe[idx[good]], axis=1) / weight_sum[good]
        ).astype(np.float32)
    return vals

def _target_bands(mask: np.ndarray, targets: int):
    """Bands of rows ``(r0, r1)`` of ``mask`` with about ``targets`` of its cells each."""
    per_row = np.count_nonzero(mask, axis=1)
    cumulative = np.cumsum(per_row)
    r0 = 0
    while r0 < mask.shape[0]:
        before = cumulative[r0 - 1] if r0 else 0
        r1 = max(int(np.searchsorted(cumulative, before + targets, side='right')), r0 + 1)
        yield r0, min(r1, mask.shape[0])
        r0 = r1


def summarize_sanity_filter(prefix: str, valid_before: np.ndarray, valid_after: np.ndarray, point_types: np.ndarray | None = None) -> str:
    """One-line summary of how many points the WSE sanity filter kept."""
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
    """Gaussian smooth while preserving NaN masks.

    SciPy's ``gaussian_filter`` filters down the columns and then along the
    rows (``correlate1d``, reflecting at the edges), in float64, keeping the
    first pass in float32; ``_smooth_nan_rows`` does the same arithmetic a
    row at a time, in parallel, with no raster but the result.
    """
    sd = float(sigma)
    radius = int(4.0 * sd + 0.5)  # gaussian_filter's truncate of 4
    # gaussian_filter1d's weights, reversed for correlate1d (scipy.ndimage._filters._gaussian_kernel1d)
    x = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 / (sd * sd) * x ** 2)
    weights = (weights / weights.sum())[::-1]
    if (arr.ndim == 2 and sd > 1e-15 and radius < min(arr.shape)
            and np.all(np.abs(weights[radius + 1:] - weights[:radius][::-1]) <= np.finfo(np.float64).eps)):
        return _smooth_nan_rows(arr, np.ascontiguousarray(weights[radius::-1]))
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

@njit(cache=True, inline="always")
def _reflect(i: int, n: int) -> int:
    """Index ``i`` of a line of ``n``, reflected at its ends as SciPy's ``mode='reflect'`` extends it (for ``i``
    less than ``n`` beyond them)."""
    if i < 0:
        return -i - 1
    if i >= n:
        return 2 * n - i - 1
    return i

@njit(cache=True, parallel=True)
def _smooth_nan_rows(arr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``smooth_nan`` with a symmetric kernel: ``weights[0]`` its centre's weight and ``weights[j]`` that of the
    cells ``j`` away.  Each row's first pass (down the columns) is worked out as the row is reached."""
    nrows, ncols = arr.shape
    radius = weights.size - 1
    out = np.empty((nrows, ncols), dtype=np.float32)
    for r in prange(nrows):
        down_num = np.empty(ncols, dtype=np.float32)
        down_den = np.empty(ncols, dtype=np.float32)
        for c in range(ncols):
            num = _finite_or_zero(arr[r, c]) * weights[0]
            den = (1.0 if np.isfinite(arr[r, c]) else 0.0) * weights[0]
            for j in range(radius, 0, -1):
                above, below = arr[_reflect(r - j, nrows), c], arr[_reflect(r + j, nrows), c]
                num += (_finite_or_zero(above) + _finite_or_zero(below)) * weights[j]
                den += ((1.0 if np.isfinite(above) else 0.0) + (1.0 if np.isfinite(below) else 0.0)) * weights[j]
            down_num[c] = num
            down_den[c] = den

        for c in range(ncols):
            num = np.float64(down_num[c]) * weights[0]
            den = np.float64(down_den[c]) * weights[0]
            for j in range(radius, 0, -1):
                left, right = _reflect(c - j, ncols), _reflect(c + j, ncols)
                num += (np.float64(down_num[left]) + np.float64(down_num[right])) * weights[j]
                den += (np.float64(down_den[left]) + np.float64(down_den[right])) * weights[j]
            num32, den32 = np.float32(num), np.float32(den)
            out[r, c] = num32 / den32 if den32 > np.float32(1.0e-6) else np.float32(np.nan)

    return out

@njit(cache=True, inline="always")
def _finite_or_zero(value) -> float:
    """``value`` as float32 and then float64, or 0 where it isn't finite, as ``smooth_nan`` fills its gaps."""
    return np.float64(np.float32(value)) if np.isfinite(value) else 0.0

@njit(cache=True, parallel=True)
def _flood_candidates(wse, dem, corridor, anchor, nodata):
    """The cells that may flood, and the anchors among them: an anchor on ground, or a corridor cell whose ground
    is below the water."""
    candidates = np.empty(dem.shape, dtype=np.bool_)
    seeds = np.empty(dem.shape, dtype=np.bool_)
    for r in prange(dem.shape[0]):
        for c in range(dem.shape[1]):
            ground = dem[r, c] > nodata
            water = wse[r, c]
            seeds[r, c] = anchor[r, c] and ground
            candidates[r, c] = seeds[r, c] or (ground and corridor[r, c] and water == water and dem[r, c] < water)     # noqa: PLR0124

    return candidates, seeds

@njit(cache=True, parallel=True)
def _dry_out(flood, wse, dem, nodata):
    """Neither water nor flood, in place, where the ground is at or above the water or has no data."""
    for r in prange(dem.shape[0]):
        for c in range(dem.shape[1]):
            if not dem[r, c] > nodata or dem[r, c] >= wse[r, c]:
                wse[r, c] = np.nan
                flood[r, c] = False

def compute_fhs_flood_from_wse(
    wse: np.ndarray,
    dem: np.ndarray,
    corridor: np.ndarray,
    anchor: np.ndarray,
    connectivity: int,
    smooth_sigma_pixels: float,
    in_place: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Flood mask and WSE from an interpolated water surface: ``(flood, wse)``.

    After optional Gaussian smoothing, a corridor cell floods where the WSE is
    above the DEM and it connects (4- or 8-connected) to the anchor cells.
    WSE is NaN wherever the ground is at or above it, or has no data (at or
    below ``NO_GROUND``), and no such cell floods.  Unsmoothed, the WSE is
    ``wse`` itself with ``in_place`` (float32), and otherwise a copy.
    """
    smoothed = smooth_sigma_pixels > 0
    if smoothed:
        wse = smooth_nan(wse, smooth_sigma_pixels)
    elif not in_place:
        wse = wse.astype(np.float32, copy=True)  # the caller's; smooth_nan's is new

    # A cell is initially eligible when the interpolated WSE overtops the DEM inside the corridor, and the anchors
    # are, then the flood's patches that hold an anchor are kept (binary_propagation of the anchors)
    candidates, seeds = _flood_candidates(wse, dem, corridor, anchor, NO_GROUND)
    flood = patches_holding(candidates, seeds, connectivity == 8)
    del candidates, seeds
    _dry_out(flood, wse, dem, NO_GROUND)

    return flood.view(np.uint8), wse

@njit(cache=True, parallel=True)
def _depth_in_place(wse, dem):
    """The depth, ``wse`` less the ground and at least 0, in ``wse`` itself; NaN stays NaN."""
    for r in prange(wse.shape[0]):
        for c in range(wse.shape[1]):
            water = wse[r, c]
            if water == water:     # noqa: PLR0124
                wse[r, c] = max(water - dem[r, c], np.float32(0.0))

@njit(cache=True, parallel=True)
def _slope_in_flood(slope, flood):
    """``slope`` NaN off the flood and at least 0.0002 on it where it has a value, in place."""
    for r in prange(slope.shape[0]):
        for c in range(slope.shape[1]):
            if flood[r, c] == 0:
                slope[r, c] = np.nan
            elif slope[r, c] <= 0.0:
                slope[r, c] = np.float32(0.0002)

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
    fast_mode: bool = False,
    with_wse: bool = True,
    with_depth: bool = True,
) -> dict[str, np.ndarray | str | bool]:
    """Flood map for one event from water-surface points.

    The points' WSEs are interpolated by inverse-distance weighting (``k``
    nearest points, weight ``1 / distance**power``) inside a corridor buffered
    around them, then turned into a flood mask by ``compute_fhs_flood_from_wse``.

    Parameters
    ----------
    dem : ndarray
        The DEM, not padded; ground at or below ``NO_GROUND`` is no-data.
    point_wse : ndarray
        Each point's water-surface elevation.
    point_rows, point_cols, point_x, point_y : ndarray, optional
        Where the points are: x/y in metres on the local grid (see
        ``_grid_xy_from_rc``), or rows and columns.
    point_topwidth, point_ids, point_dem, point_types, point_slope : ndarray, optional
        Each point's top width, reach id, ground elevation (sampled from
        ``dem`` if omitted), label for the summary message, and slope (also
        interpolated, for velocity).
    dx, dy : float
        Cell size in metres.
    prefix : str
        Label for the summary message.
    corridor_buffer_m, anchor_buffer_m, use_topwidth_buffers, corridor_topwidth_factor, anchor_topwidth_factor
        Buffer sizes; see ``build_variable_buffer_masks_from_points``.
    connectivity : int
        4 or 8, the neighbours through which the flood connects.
    k, power : int, float
        Number of nearest points and the inverse-distance power.
    max_distance_m, use_topwidth_max_distance, maxdist_topwidth_factor
        How far a point reaches: ``max_distance_m``, or its reach's median top
        width times ``maxdist_topwidth_factor``.
    min_topwidth_m, max_topwidth_m, fallback_topwidth_m : float
        Top widths are clipped to this range, and missing ones replaced.
    smooth_sigma_pixels : float
        Gaussian smoothing of the interpolated WSE, in cells.
    apply_wse_sanity_filter, wse_sanity_tolerance_m
        Drop points whose WSE (plus the tolerance) is below their ground.
    fast_mode : bool
        Use the fixed buffers and distance instead of top-width-based ones.
    with_wse, with_depth : bool
        Return the WSE and the depth; without the WSE, the depth is worked
        out in its raster, and without either neither is kept.

    Returns
    -------
    dict
        ``flood``, ``wse``, ``depth``, ``slope`` (None unless asked for),
        ``stats_message``, ``valid_point_mask`` and ``has_valid_points``.
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
        point_dem = _ground_at_cells(dem, point_rows, point_cols)
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
    transform = (0.0, float(dx), 0.0, 0.0, 0.0, -float(dy))
    xs = (np.arange(ncols, dtype=np.float32) + np.float32(0.5)) * np.float32(dx)
    ys = -((np.arange(nrows, dtype=np.float32) + np.float32(0.5)) * np.float32(dy))

    if not np.any(valid):
        return {
            "flood": np.zeros(dem.shape, dtype=np.uint8),
            "wse": np.full(dem.shape, np.nan, dtype=np.float32) if with_wse else None,
            "depth": np.full(dem.shape, np.nan, dtype=np.float32) if with_depth else None,
            "slope": np.full(dem.shape, np.nan, dtype=np.float32) if point_slope is not None else None,
            "stats_message": stats_message,
            "valid_point_mask": valid,
            "has_valid_points": False,
        }

    # Where IDW can populate WSE values, and where the connected flood starts
    corridor, anchor = build_variable_buffer_masks_from_points(
        x=point_x[valid],
        y=point_y[valid],
        topwidth=point_topwidth[valid],
        dem_shape=dem.shape,
        transform=transform,
        fixed_corridor_buffer_m=corridor_buffer_m,
        fixed_anchor_buffer_m=anchor_buffer_m,
        use_topwidth_buffers=use_topwidth_buffers,
        corridor_topwidth_factor=corridor_topwidth_factor,
        anchor_topwidth_factor=anchor_topwidth_factor,
    )
    # Only cells with ground are interpolated, or flood
    corridor &= dem > NO_GROUND

    x_valid = point_x[valid]
    y_valid = point_y[valid]
    z_valid = point_wse[valid].astype(np.float32)
    tw_valid = point_topwidth[valid].astype(np.float32)

    # Each reach's points reach as far as its median top width, so larger
    # rivers influence a wider neighbourhood
    if point_ids is not None and use_topwidth_max_distance:
        tmp_df = pd.DataFrame({"id_col": point_ids[valid], "tw_based_dist": maxdist_topwidth_factor * tw_valid})
        grouped = tmp_df.groupby("id_col")["tw_based_dist"].median()
        point_max_distance = tmp_df["id_col"].map(grouped).to_numpy(dtype=np.float32, copy=True)
        bad = ~np.isfinite(point_max_distance)
        point_max_distance[bad] = max_distance_m
    else:
        point_max_distance = np.full(len(z_valid), max_distance_m, dtype=np.float32)

    # Interpolate only within the buffered corridor; cells outside it stay NaN.  A band of rows at a time: a cell's
    # k nearest points take 8 k bytes, too many to keep for a whole corridor at once.
    wse = np.full(dem.shape, np.nan, dtype=np.float32)
    slope = np.full(dem.shape, np.nan, dtype=np.float32) if point_slope is not None else None
    tree = cKDTree(np.column_stack((x_valid, y_valid))) if corridor.any() else None
    for r0, r1 in (_target_bands(corridor, _IDW_TARGETS) if tree is not None else ()):
        target_x, target_y, target_rows, target_cols = build_target_coordinates(xs, ys[r0:r1], corridor[r0:r1])
        target_rows += r0
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
            tree=tree,
            workers=numba.get_num_threads(),
        )
        wse[target_rows, target_cols] = idw_values(z_valid, pre, power)
        if slope is not None:
            slope[target_rows, target_cols] = idw_values(point_slope[valid].astype(np.float32), pre, power)
        del pre

    flood, wse = compute_fhs_flood_from_wse(
        wse=wse,
        dem=dem,
        corridor=corridor,
        anchor=anchor,
        connectivity=connectivity,
        smooth_sigma_pixels=smooth_sigma_pixels,
        in_place=True,
    )
    del corridor, anchor
    depth = None
    if with_depth:
        depth = wse.copy() if with_wse else wse
        _depth_in_place(depth, dem)
    if not with_wse:
        wse = None
    if slope is not None:
        _slope_in_flood(slope, flood)

    return {
        "flood": flood,
        "wse": wse,
        "depth": depth,
        "slope": slope,
        "stats_message": stats_message,
        "valid_point_mask": valid,
        "has_valid_points": True,
    }


_DEFAULT_OPTIONS = {
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
    "one_based_vdt_rc": False,
}


def multi_point_interpolation(
    E: np.ndarray,
    B,
    RR: np.ndarray,
    CC: np.ndarray,
    COMID_Unique_Flow: dict,
    CurveParamFileName: str | None,
    VDTDatabaseFileName: str | None,
    dx: float,
    dy: float,
    TopWidthPlausibleLimit: float,
    Set_Depth: float,
    flood_vdt_cells: bool,
    mapper_options: dict | None = None,
    with_depth: bool = True,
    with_slope: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, str]:
    """Multi-point interpolation mapper: ``(flood, depth, slope, message)`` for one flow event.

    1. Build one water-surface point per stream cell from the VDT database
       (preferred) or the curve file, at each reach's flow, with its top
       width as the table gives it (``TW_MultFact`` is the kernel-weighted
       mapper's; this one scales top widths with its ``MPI_*_TopWidth_Factor``
       settings); a reach without a flow above 0 has none.  With
       both tables, missing ``Elev``, ``Slope`` and ``XS_Angle`` values are
       borrowed from the curve file, and any ``Elev`` still missing is the
       DEM's at the cell.
    2. Optionally drop points with outlying depths.
    3. Add points across the cross sections of wide rivers.
    4. Interpolate and map the flood with ``create_fhs_flood_map_from_points``.

    With ``Set_Depth`` the points are the stream cells instead, their water
    that deep (see ``build_fhs_points_from_stream_cells``).  ``E`` is the DEM
    padded by one cell, ``RR``/``CC`` the stream cells' indices in it and
    ``B`` their reach ids (``B.values``, in the same order).
    ``mapper_options`` overrides the tuning defaults (``_DEFAULT_OPTIONS``);
    their input-file keys start with ``MPI_``.  With ``flood_vdt_cells``
    every stream cell floods.  ``depth`` is None without ``with_depth``, and
    ``slope`` without ``with_slope``.
    """
    options = dict(_DEFAULT_OPTIONS)
    if mapper_options:
        options.update({k: v for k, v in mapper_options.items() if v is not None})

    dem = E[1:-1, 1:-1]
    if Set_Depth > 0.0:
        point_data = build_fhs_points_from_stream_cells(E, B.values, RR, CC, Set_Depth, TopWidthPlausibleLimit, dx,
                                                        dy)
    else:
        one_based = bool(options["one_based_vdt_rc"])
        if VDTDatabaseFileName:
            scenario_df = build_fhs_scenario_dataframe_from_vdt(VDTDatabaseFileName, COMID_Unique_Flow, one_based)
            if CurveParamFileName:
                curve_metadata = read_curve2flood_table(
                    CurveParamFileName, ["COMID", "Row", "Col", "Elev", "DEM_Elev", "BaseElev", "Slope", "XS_Angle"])
                scenario_df = merge_missing_fhs_metadata(
                    scenario_df, _coerce_fhs_metadata_columns(curve_metadata, CurveParamFileName, one_based))
        elif CurveParamFileName:
            scenario_df = build_fhs_scenario_dataframe_from_curve_file(CurveParamFileName, COMID_Unique_Flow,
                                                                       one_based)
        else:
            raise ValueError("The multi-point mapper needs a VDT database or a curve file, or Set_Depth.")
        scenario_df = _fill_missing_ground(scenario_df, dem)
        if options["remove_hwm_outliers"]:
            scenario_df = filter_hwm_outliers_single_scenario(
                scenario_df,
                method=str(options["hwm_outlier_method"]),
                threshold=float(options["hwm_outlier_threshold"]),
                group_by=str(options["hwm_outlier_group_by"]),
                min_samples=int(options["hwm_outlier_min_samples"]),
            )
        point_data = create_fhs_points_from_scenario_dataframe(
            results_df=scenario_df,
            dem_array=dem,
            dx=dx,
            dy=dy,
            topwidth_threshold_m=options["topwidth_threshold_m"],
            xs_point_spacing_m=float(options["xs_point_spacing_m"]),
        )

    result = create_fhs_flood_map_from_points(
        dem=dem,
        point_rows=point_data["rows"],
        point_cols=point_data["cols"],
        point_x=point_data["x"],
        point_y=point_data["y"],
        point_wse=point_data["wse"],
        point_topwidth=point_data["topwidth"],
        point_ids=point_data["ids"],
        point_dem=point_data["dem"],
        point_types=point_data["types"],
        point_slope=point_data["slope"] if with_slope else None,
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
        fast_mode=bool(options["fast_mode"]),
        with_wse=False,
        with_depth=with_depth,
    )

    flood = result["flood"]
    if flood_vdt_cells:
        # Keep the stream cells wet even where the interpolation left them dry, as the other mappers do
        flood[RR - 1, CC - 1] = 1

    return flood, result["depth"], result["slope"], result["stats_message"]
