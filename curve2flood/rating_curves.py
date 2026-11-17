from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import polars.selectors as cs
from numba import njit

from curve2flood._log import LOG
from curve2flood.config import Config, is_parquet

if TYPE_CHECKING:
    import pandas as pd


def calculate_interpolated_curvefile(CurveParamFileName: str, COMID_Unique_Flow: dict, TW_MultFact: float) -> pd.DataFrame:
    """Evaluate the curve file's rating curves at each reach's flow.

    Depth, top width and velocity follow ``a * Q**b`` power laws, and WSE is
    the depth plus the cell's ``BaseElev``.  Cells more than two standard
    deviations from their reach's mean are dropped, and top widths are scaled
    by ``TW_MultFact``.
    """
    # Imported here: only curve files need pandas
    import pandas as pd

    from curve2flood.spreaders.multi_point import filter_outliers

    LOG.debug('\nOpening and Reading ' + CurveParamFileName)

    if is_parquet(CurveParamFileName):
        curve_df = pd.read_parquet(CurveParamFileName)
    else:
        curve_df = pd.read_csv(CurveParamFileName)

    # Add COMID flow information
    comid_flow_df = pd.DataFrame(COMID_Unique_Flow.items(), columns=['COMID', 'Flow'])
    curve_df = curve_df.merge(comid_flow_df, on="COMID", how="left")

    curve_df['Depth'] = curve_df['depth_a']*curve_df['Flow']**curve_df['depth_b']
    curve_df['TopWidth'] = curve_df['tw_a']*curve_df['Flow']**curve_df['tw_b']
    curve_df['Velocity'] = curve_df['vel_a']*curve_df['Flow']**curve_df['vel_b']
    curve_df = curve_df[curve_df['Depth']>0]
    curve_df = curve_df[curve_df['TopWidth']>0]
    curve_df['WSE'] = curve_df['Depth'] + curve_df['BaseElev']

    curve_df = filter_outliers(curve_df)

    curve_df['TopWidth'] = curve_df['TopWidth'].astype(np.float32) * np.float32(TW_MultFact)

    return curve_df

def Calculate_TW_D_V_ForEachCOMID_CurveFile(CurveParamFileName: str, COMID_Unique_Flow: dict, COMID_Unique, T_Rast, W_Rast, S_Rast, TW_MultFact):
    """Fill the top-width, WSE and slope rasters from the curve file for one event.

    Returns ``(COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, T_Rast,
    W_Rast, S_Rast)``: each reach's median top width and depth (0 for reaches
    without curves), and the largest of those top widths.
    """
    import pandas as pd

    curve_df = calculate_interpolated_curvefile(CurveParamFileName, COMID_Unique_Flow, TW_MultFact)

    T_Rast[curve_df['Row'], curve_df['Col']] = curve_df['TopWidth']
    W_Rast[curve_df['Row'], curve_df['Col']] = curve_df['WSE']
    if S_Rast is not None:
        S_Rast[curve_df['Row'], curve_df['Col']] = curve_df['Slope']

    median_values = curve_df.groupby('COMID').agg({'TopWidth': 'median', 'Depth': 'median'})

    # Map results back to the unique COMID list
    comid_result_df = pd.DataFrame({'COMID': COMID_Unique})
    comid_result_df = comid_result_df.merge(median_values, on='COMID', how='left').fillna(0)

    comid_result_df['TopWidth'] = comid_result_df['TopWidth'].astype(np.float32)
    comid_result_df['Depth'] = comid_result_df['Depth'].astype(np.float32)

    COMID_Unique_TW = comid_result_df.set_index('COMID')['TopWidth'].to_dict()
    COMID_Unique_Depth = comid_result_df.set_index('COMID')['Depth'].to_dict()

    TopWidthMax = comid_result_df['TopWidth'].max()

    return (
        COMID_Unique_TW,
        COMID_Unique_Depth,
        TopWidthMax,
        T_Rast,
        W_Rast,
        S_Rast,
    )

@njit("float32(float32, float32[:], float32[:])", cache=True)
def Find_TopWidth_at_Baseflow_when_using_VDT(QB, flow_values, top_width_values):
    """Top width at the first VDT flow that is at least the baseflow ``QB``.

    Returns the last top width when ``QB`` is above every flow.
    """
    idx = np.searchsorted(flow_values, QB, side="left")
    if idx >= len(flow_values):
        return top_width_values[-1]
    else:
        return top_width_values[idx]

@njit("float32(float32[:], float32[:], float32)", cache=True)
def interp1d_numba(x: np.ndarray, y: np.ndarray, xi: float) -> float:
    """Linearly interpolate ``y(x)`` at ``xi``, extrapolating from the end segments.

    ``x`` must be sorted.  Same as SciPy's ``interp1d(..., fill_value='extrapolate')``.
    """
    n = len(x)

    if xi <= x[0]:
        # extrapolate left
        return y[0] + (xi - x[0]) * (y[1] - y[0]) / (x[1] - x[0]) if (x[1] - x[0]) != 0 else y[0]
    elif xi >= x[n - 1]:
        # extrapolate right
        return y[n - 2] + (xi - x[n - 2]) * (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]) if (x[n - 1] - x[n - 2]) != 0 else y[n - 1]

    # binary search for correct interval
    low = 0
    high = n - 1
    while high - low > 1:
        mid = (high + low) // 2
        if x[mid] > xi:
            high = mid
        else:
            low = mid

    # linear interpolation
    x0 = x[low]
    x1 = x[high]
    y0 = y[low]
    y1 = y[high]

    return y0 + (xi - x0) * (y1 - y0) / (x1 - x0) if (x1 - x0) != 0 else y0

@njit(cache=True)
def vdt_interpolate(flow: np.ndarray,
                    qb: np.ndarray,
                    flow_values: np.ndarray,
                    top_width_values: np.ndarray,
                    wse_values: np.ndarray,
                    vel_values: np.ndarray,
                    elev: np.ndarray,
                    tw_mult_fact: np.ndarray) -> tuple[np.ndarray, ...]:
    """Interpolate each VDT row's curves at its flow.

    Returns ``(top_width, depth, wse, velocity)``.  Flows above a row's largest
    flow take its last values.  Smaller flows are interpolated linearly (and
    extrapolated below the first flow), with the WSE kept at or above the
    cell's elevation ``elev`` and the depth at least 1 mm.  Depths are
    measured from ``elev``.  The top width is at least the top
    width at baseflow ``qb``, and is then multiplied by ``tw_mult_fact``.
    """
    top_width = np.empty_like(flow)
    depth = np.empty_like(flow)
    wse = np.empty_like(flow)
    baseflow_tw = np.empty_like(flow)
    vel = np.empty_like(flow)

    for i in range(len(flow)):
        if flow[i] >= flow_values[i][-1]:
            # Above the maximum flow value
            top_width[i] = top_width_values[i][-1]
            wse[i] = wse_values[i][-1]
            depth[i] = wse[i] - elev[i]
            vel[i] = vel_values[i][-1]
        else:
            # Interpolate
            wse[i] = interp1d_numba(flow_values[i], wse_values[i], flow[i])
            top_width[i] = max(interp1d_numba(flow_values[i], top_width_values[i], flow[i]), 0.0)
            wse[i] = max(wse[i], elev[i])
            depth[i] = max(wse[i] - elev[i], 0.001)
            vel[i] = max(interp1d_numba(flow_values[i], vel_values[i], flow[i]), 0.0)

        baseflow_tw[i] = Find_TopWidth_at_Baseflow_when_using_VDT(qb[i], flow_values[i], top_width_values[i])

    # Ensure TopWidth respects baseflow and scale
    top_width = np.maximum(top_width, baseflow_tw) * tw_mult_fact

    return top_width, depth, wse, vel

def calculate_interpolated_vdt(
        VDTDatabaseFileName: str,
        COMID_Unique_Flow: dict,
        TW_MultFact: float,
        ) -> pl.DataFrame:
    """Interpolate the VDT database at each reach's flow.

    Returns one row per stream cell with ``TopWidth``, ``Depth``, ``WSE`` and
    ``Velocity`` (see ``vdt_interpolate``), measured from the VDT's ``Elev``,
    the channel bed under the cell.  Within each reach, cells outside the
    1st-99th percentile of top width or velocity are dropped.  WSE isn't
    filtered: it falls along a reach, so its extremes are the reach's two
    ends, and dropping them left gaps at every junction.
    """
    LOG.debug('\nOpening and Reading ' + VDTDatabaseFileName)

    if is_parquet(VDTDatabaseFileName):
        vdt_df = pl.read_parquet(VDTDatabaseFileName)
    else:
        vdt_df = pl.read_csv(VDTDatabaseFileName)

    if vdt_df.is_empty():
        raise ValueError("The VDT Database file is empty or could not be read properly.")
    if 'Elev' not in vdt_df.columns:
        raise ValueError(f"The VDT database {VDTDatabaseFileName} has no Elev column.")

    # Add COMID flow information
    comid_flow_df = pl.DataFrame(data=list(COMID_Unique_Flow.items()), schema=['COMID', 'Flow'], orient='row')
    vdt_df = vdt_df.join(comid_flow_df, on='COMID', how='inner')

    vdt_df = vdt_df.with_columns([
        pl.col('Row').cast(pl.Int32),
        pl.col('Col').cast(pl.Int32)
    ])

    flow = vdt_df['Flow'].to_numpy().astype(np.float32, copy=False)
    qb = vdt_df['QBaseflow'].to_numpy().astype(np.float32, copy=False)
    elev = vdt_df['Elev'].to_numpy().astype(np.float32, copy=False)

    # The flow, top-width, WSE and velocity curves, one row per cell
    flow_values = vdt_df.select(cs.starts_with('q_')).to_numpy().astype(np.float32, copy=False)
    top_width_values = vdt_df.select(cs.starts_with('t_')).to_numpy().astype(np.float32, copy=False)
    wse_values = vdt_df.select(cs.starts_with('wse_')).to_numpy().astype(np.float32, copy=False)
    vel_values = vdt_df.select(cs.starts_with('v_')).to_numpy().astype(np.float32, copy=False)

    tw_mult_fact = np.full(flow.size, TW_MultFact, dtype=np.float32)
    top_width, depth, wse, velocity = vdt_interpolate(flow, qb, flow_values, top_width_values, wse_values, vel_values, elev, tw_mult_fact)

    # Add the interpolated values back to the DataFrame.  The ~120 curve columns
    # they came from are dropped first, or every filter below would copy them.
    vdt_df = vdt_df.drop(cs.starts_with('q_', 't_', 'wse_', 'v_')).with_columns([
        pl.Series('TopWidth', top_width),
        pl.Series('Depth', depth),
        pl.Series('WSE', wse),
        pl.Series('Velocity', velocity),
    ])

    vdt_df = vdt_df.drop_nans(subset=['TopWidth', 'Depth', 'WSE', 'Velocity'])

    vdt_df = vdt_df.with_columns(
        pl.col('TopWidth').round(2),
        pl.col('WSE').round(2),
        pl.col('Velocity').round(2),
    )

    # Drop each reach's outliers (not by WSE: see above)
    cols = ['TopWidth', 'Velocity']
    for col in cols:
        vdt_df = vdt_df.with_columns(
            pl.col(col).quantile(0.01).over("COMID").alias("q01"),
            pl.col(col).quantile(0.99).over("COMID").alias("q99"),
        ).filter(
            pl.col(col).is_between(pl.col("q01"), pl.col("q99"))
        ).drop("q01", "q99")

    return vdt_df

def Calculate_TW_D_V_ForEachCOMID_VDTDatabase(VDTDatabaseFileName: str, COMID_Unique_Flow: dict, COMID_Unique, T_Rast, W_Rast, S_Rast, TW_MultFact, V_Rast=None):
    """Fill the top-width, WSE and slope rasters from the VDT database for one event.

    ``V_Rast``, when given, gets the VDT's valley slopes (``Valley_Slope``).

    Returns ``(COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, T_Rast,
    W_Rast, S_Rast)``: each reach's median top width and depth, and the
    largest of those top widths.
    """
    vdt_df = calculate_interpolated_vdt(VDTDatabaseFileName, COMID_Unique_Flow, TW_MultFact)

    T_Rast[vdt_df['Row'], vdt_df['Col']] = vdt_df['TopWidth']
    W_Rast[vdt_df['Row'], vdt_df['Col']] = vdt_df['WSE']
    if S_Rast is not None:
        S_Rast[vdt_df['Row'], vdt_df['Col']] = vdt_df['Slope']
    if V_Rast is not None:
        V_Rast[vdt_df['Row'], vdt_df['Col']] = vdt_df['Valley_Slope']

    median_values = vdt_df.group_by('COMID').agg([
        pl.col('TopWidth').median(),
        pl.col('Depth').median(),
    ])

    # Map results back to the unique COMID list
    comid_result_df = pl.DataFrame({'COMID': COMID_Unique})
    comid_result_df = comid_result_df.join(median_values, on='COMID', how='left').fill_nan(0)
    comid_result_df = comid_result_df.with_columns(
        pl.col('TopWidth').cast(pl.Float32),
        pl.col('Depth').cast(pl.Float32),
    )

    COMID_Unique_TW = dict(zip(comid_result_df['COMID'].to_numpy(), comid_result_df['TopWidth'].to_numpy()))
    COMID_Unique_Depth = dict(zip(comid_result_df['COMID'].to_numpy(), comid_result_df['Depth'].to_numpy()))

    TopWidthMax = comid_result_df['TopWidth'].max()

    return (
        COMID_Unique_TW,
        COMID_Unique_Depth,
        TopWidthMax,
        T_Rast,
        W_Rast,
        S_Rast
    )


def Calculate_Depth_TopWidth_TWMax_Velocity(config: Config, COMID_Unique_Flow, COMID_Unique, T_Rast, W_Rast, S_Rast, quiet,
                                            V_Rast=None):
    """Fill the per-cell rasters for one flow event and summarise each reach.

    Returns ``(COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, T_Rast,
    W_Rast, S_Rast)``: the median top width and depth of each reach, and the
    largest top width, capped at ``TopWidthPlausibleLimit``.  With
    ``Set_Depth`` every reach gets that depth and the plausible-limit top
    width instead, and the rasters are left untouched.  ``V_Rast``, when
    given, gets the VDT's valley slopes.
    """
    Set_Depth = config.set_depth
    TopWidthPlausibleLimit = config.topwidthplausiblelimit
    if Set_Depth > 0.0:
        top_width = TopWidthPlausibleLimit if TopWidthPlausibleLimit > 0 else -9999.0
        COMID_Unique_TW = {comid: top_width for comid in COMID_Unique}
        COMID_Unique_Depth = {comid: Set_Depth for comid in COMID_Unique}
        TopWidthMax = TopWidthPlausibleLimit
    elif config.print_vdt_database:
        # The VDT database is preferred over the curve file when both are given
        (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, T_Rast, W_Rast, S_Rast) = Calculate_TW_D_V_ForEachCOMID_VDTDatabase(
            config.print_vdt_database, COMID_Unique_Flow, COMID_Unique, T_Rast, W_Rast, S_Rast, config.tw_multfact,
            V_Rast)
    else:
        (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, T_Rast, W_Rast, S_Rast) = Calculate_TW_D_V_ForEachCOMID_CurveFile(
            config.print_curve_file, COMID_Unique_Flow, COMID_Unique, T_Rast, W_Rast, S_Rast, config.tw_multfact)

    LOG.info('Maximum Top Width = ' + str(TopWidthMax))

    if not quiet:
        for comid in COMID_Unique:
            if COMID_Unique_TW[comid]>TopWidthPlausibleLimit:
                LOG.warning(f"Reach {comid} (flow {COMID_Unique_Flow.get(comid)}, depth {COMID_Unique_Depth[comid]}) has a "
                            f"median top width of {COMID_Unique_TW[comid]}, above TopWidthPlausibleLimit; it is capped there.")

    TopWidthMax = min(TopWidthMax, TopWidthPlausibleLimit)

    return COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, T_Rast, W_Rast, S_Rast