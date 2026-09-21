"""
Python port of `fldpln_model_v5ram.m`.

This module keeps the MATLAB model's core behavior: it builds a floodplain
database for one stream segment using a filled DEM, a D8 flow-direction raster,
and segment metadata. The output is the FLDPLN table:

    column 1: flood source pixel (FSP)
    column 2: floodplain pixel
    column 3: depth to flood (DTF)
    column 4: DTF adjusted by fill depth

The seed-point branch from the MATLAB file is intentionally not implemented.
Use a segment id and segment metadata. Pixel ids default to the MATLAB/BIL
convention used by the source file: row-major, one-based linear ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scipy.io import loadmat, savemat
from osgeo import gdal


# Neighbor order used in the MATLAB code:
# 1 2 3
# 4 x 5
# 6 7 8
NEIGHBOR_DELTAS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

# Flow direction values in the MATLAB port's D8 convention.
INFLOW = np.asarray([2, 4, 8, 1, 16, 128, 64, 32], dtype=np.int32)
OUTFLOW = np.asarray([32, 64, 128, 16, 1, 8, 4, 2], dtype=np.int32)


@dataclass
class FldplnResult:
    fldpln: np.ndarray
    header: list[str]
    flood_map: np.ndarray
    processing_times: list[tuple[float, float]]


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_flat_raster(data: np.ndarray, shape: tuple[int, int] | None) -> tuple[np.ndarray, int, int]:
    arr = np.asarray(data)
    if arr.ndim == 2:
        nrows, ncols = arr.shape
        return arr.astype(np.float32, copy=False).ravel(order="C"), nrows, ncols
    if arr.ndim != 1:
        raise ValueError("Raster inputs must be 1D row-major vectors or 2D arrays.")
    if shape is None:
        raise ValueError("shape=(nrows, ncols) is required when raster inputs are 1D.")
    nrows, ncols = shape
    if arr.size != nrows * ncols:
        raise ValueError("1D raster length does not match shape.")
    return arr.astype(np.float32, copy=False), nrows, ncols


def _pixel_to_rc(pixel: int, ncols: int) -> tuple[int, int]:
    return pixel // ncols, pixel % ncols


def _rc_to_pixel(row: int, col: int, ncols: int) -> int:
    return row * ncols + col


def _valid_neighbors(pixel: int, nrows: int, ncols: int) -> list[tuple[int, int]]:
    row, col = _pixel_to_rc(pixel, ncols)
    out: list[tuple[int, int]] = []
    for pos, (dr, dc) in enumerate(NEIGHBOR_DELTAS):
        rr = row + dr
        cc = col + dc
        if 0 <= rr < nrows and 0 <= cc < ncols:
            out.append((pos, _rc_to_pixel(rr, cc, ncols)))
    return out


def _next_downstream(pixel: int, fdr: np.ndarray, nrows: int, ncols: int) -> int | None:
    fd = int(fdr[pixel])
    matches = np.where(OUTFLOW == fd)[0]
    if matches.size == 0:
        return None
    pos = int(matches[0])
    row, col = _pixel_to_rc(pixel, ncols)
    dr, dc = NEIGHBOR_DELTAS[pos]
    rr = row + dr
    cc = col + dc
    if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols:
        return None
    return _rc_to_pixel(rr, cc, ncols)


def _segment_pixels(seg_id: int, seg_info: np.ndarray, fdr: np.ndarray,
                    nrows: int, ncols: int) -> list[int]:
    """Return stream pixels for a segment id. The seed-point path is not ported."""
    row_idx = int(seg_id)
    if row_idx < 0 or row_idx >= seg_info.shape[0]:
        raise IndexError("seg0 is outside seg_info.")

    start = round(seg_info[row_idx, 0])
    length = round(seg_info[row_idx, 4])

    if length <= 0:
        return []

    pixels = [start]
    current = start
    for _ in range(1, length):
        nxt = _next_downstream(current, fdr, nrows, ncols)
        if nxt is None:
            break
        pixels.append(nxt)
        current = nxt
    return pixels


def _downstream_exclusion(seg_id: int, seg_info: np.ndarray, fdr: np.ndarray,
                          nrows: int, ncols: int) -> set[int]:
    """Pixels downstream of the segment end, excluded from spillover candidates."""
    row_idx = int(seg_id)
    end_pixel = int(seg_info[row_idx, 1])

    excluded: set[int] = set()
    current = end_pixel
    seen = {current}
    while True:
        nxt = _next_downstream(current, fdr, nrows, ncols)
        if nxt is None or nxt in seen:
            break
        excluded.add(nxt)
        seen.add(nxt)
        current = nxt
    return excluded


def _backfill_from_source(source: int, base_dtf: float, max_wse: float,
                          fil: np.ndarray, fdr: np.ndarray, nrows: int, ncols: int,
                          bg: float, flood_members: set[int] | None = None,
                          excluded: set[int] | None = None) -> list[tuple[int, float]]:
    """
    Backfill opposite the D8 flow direction from `source`.

    Returned DTF values include `base_dtf`. When `flood_members` is supplied,
    pixels already in the floodplain are not returned; this matches the first
    backfill pass in the MATLAB code. Spillover backfill passes leave it as None
    so existing pixels can be overwritten when the new DTF is lower.
    """
    source_elev = float(fil[source])
    result: list[tuple[int, float]] = []
    queue: list[int] = [source]
    visited = {source}
    excluded = excluded or set()

    while queue:
        center = queue.pop(0)
        for pos, nbr in _valid_neighbors(center, nrows, ncols):
            if nbr in visited or nbr in excluded:
                continue
            if flood_members is not None and nbr in flood_members:
                continue
            if float(fil[nbr]) == bg or float(fil[nbr]) > max_wse:
                continue
            if int(fdr[nbr]) != int(INFLOW[pos]):
                continue
            dtf = base_dtf + max(0.0, float(fil[nbr]) - source_elev)
            visited.add(nbr)
            result.append((nbr, dtf))
            queue.append(nbr)

    return result


def _boundary(records: dict[int, tuple[int, float]], fil: np.ndarray,
              nrows: int, ncols: int, bg: float) -> list[tuple[int, int, float]]:
    members = set(records)
    out: list[tuple[int, int, float]] = []
    for pixel, (fsp, dtf) in records.items():
        if float(fil[pixel]) == bg:
            continue
        for _, nbr in _valid_neighbors(pixel, nrows, ncols):
            if nbr not in members and float(fil[nbr]) != bg:
                out.append((fsp, pixel, dtf))
                break
    out.sort(key=lambda x: (x[2], -float(fil[x[1]])))
    return out


def _flood_map(records: dict[int, tuple[int, float]], n: int, fldmn: float) -> np.ndarray:
    flddat = np.zeros(n, dtype=np.float32)
    for pixel, (_, dtf) in records.items():
        flddat[pixel] = max(float(fldmn), float(dtf))
    return flddat


def _spill_candidates(boundary: list[tuple[int, int, float]], records: dict[int, tuple[int, float]],
                      flddat: np.ndarray, fil: np.ndarray, nrows: int, ncols: int,
                      fldht: float, mxht: float, bg: float, excluded: set[int]) -> list[tuple[int, float, int, float, float]]:
    members = set(records)
    best: dict[int, tuple[float, float, int, float, float]] = {}
    for fsp, bdy_pixel, bdy_dtf in boundary:
        bdy_elev = float(fil[bdy_pixel])
        if bdy_elev == bg:
            continue
        limit = min(mxht, bdy_elev + fldht - bdy_dtf)
        for _, nbr in _valid_neighbors(bdy_pixel, nrows, ncols):
            if nbr in members or nbr in excluded:
                continue
            nbr_elev = float(fil[nbr])
            if nbr_elev == bg or nbr_elev > limit:
                continue
            spill_dtf = bdy_dtf + max(0.0, nbr_elev - bdy_elev)
            available_depth = fldht - bdy_dtf - max(0.0, nbr_elev - bdy_elev)
            previous = best.get(nbr)
            if previous is None:
                best[nbr] = (available_depth, bdy_elev, fsp, spill_dtf, nbr_elev)
            else:
                old_available, old_bdy_elev, *_ = previous
                if available_depth > old_available or (
                    available_depth == old_available and bdy_elev > old_bdy_elev
                ):
                    best[nbr] = (available_depth, bdy_elev, fsp, spill_dtf, nbr_elev)

    candidates = [
        (fsp, spill_dtf, pixel, pixel_elev, bdy_elev)
        for pixel, (_, bdy_elev, fsp, spill_dtf, pixel_elev) in best.items()
    ]
    candidates.sort(key=lambda x: (x[1], -x[4]))
    return candidates


def _forward_path(start: int, spill_dtf: float, fil: np.ndarray, fdr: np.ndarray,
                  flddat: np.ndarray, nrows: int, ncols: int, bg: float,
                  excluded: set[int]) -> list[int]:
    path: list[int] = []
    seen: set[int] = set()
    current = start
    while True:
        if current in seen:
            break
        seen.add(current)
        path.append(current)
        nxt = _next_downstream(current, fdr, nrows, ncols)
        if nxt is None:
            break
        if nxt in excluded:
            path.append(nxt)
            break
        if float(fil[nxt]) == bg:
            break
        if flddat[nxt] > 0 and spill_dtf >= float(flddat[nxt]):
            break
        current = nxt
    return path


def _assimilate(records: dict[int, tuple[int, float]], fsp: int, pixel: int, dtf: float) -> bool:
    old = records.get(pixel)
    if old is None or old[1] > dtf:
        records[pixel] = (fsp, dtf)
        return True
    return False


def fldpln_model_v5ram(seg0: int, inp: Any, fildat: np.ndarray, fdrdat: np.ndarray,
                       fldmn: float, fldmx: float, dh: float, mxht0: float,
                       ssflg: int | bool, bg: float = 0.0, *,
                       shape: tuple[int, int] | None = None,
                       demdat: np.ndarray | None = None,
                       save_mat: str | Path | None = None) -> FldplnResult:
    """
    Build an FLDPLN floodplain table for a stream segment.

    Parameters mirror the MATLAB function where practical. `inp` may be a dict or
    object with a `seg` field containing the segment information matrix. If
    `fildat` and `fdrdat` are 2D arrays, shape is inferred; if they are 1D
    row-major arrays, pass `shape=(nrows, ncols)`.

    The seed-point mode from the MATLAB code is not supported.
    """
    if not np.isscalar(seg0):
        raise NotImplementedError("Seed-point mode is intentionally not ported.")
    if dh <= 0:
        raise ValueError("dh must be positive.")

    fil, nrows, ncols = _as_flat_raster(fildat, shape)
    fdr, fdr_rows, fdr_cols = _as_flat_raster(fdrdat, (nrows, ncols))
    if (fdr_rows, fdr_cols) != (nrows, ncols):
        raise ValueError("fildat and fdrdat shapes differ.")
    fdr = fdr.astype(np.int32, copy=False)

    if demdat is None:
        dem = fil
    else:
        dem, dem_rows, dem_cols = _as_flat_raster(demdat, (nrows, ncols))
        if (dem_rows, dem_cols) != (nrows, ncols):
            raise ValueError("demdat shape differs from fildat.")

    seg_info_obj = _get_field(inp, "seg")
    if isinstance(seg_info_obj, (str, Path)):
        try:
            from scipy.io import loadmat
        except Exception as exc:  # pragma: no cover
            raise ImportError("scipy is required to load MATLAB segment files.") from exc
        loaded = loadmat(seg_info_obj)
        seg_info = np.asarray(loaded["seg_info"])
    else:
        seg_info = np.asarray(seg_info_obj)
    if seg_info.ndim != 2 or seg_info.shape[1] < 5:
        raise ValueError("inp.seg must provide a segment info matrix with at least 5 columns.")

    mxht = np.finfo(np.float32).max if not mxht0 else 0.99999 * float(mxht0)
    strpts = _segment_pixels(int(seg0), seg_info, fdr, nrows, ncols)
    excluded = _downstream_exclusion(int(seg0), seg_info, fdr, nrows, ncols)

    records: dict[int, tuple[int, float]] = {}
    for pixel in strpts:
        if float(fil[pixel]) != bg:
            records[pixel] = (pixel, 0.0)

    fldht = 0.0
    iterations = int(np.ceil(float(fldmx) / float(dh)))
    proctime: list[tuple[float, float]] = []

    for _ in range(iterations):
        import time

        t0 = time.perf_counter()
        fldht += min(float(dh), float(fldmx) - fldht)

        bdy = _boundary(records, fil, nrows, ncols, bg)
        flddat = _flood_map(records, fil.size, fldmn)
        flood_members = set(records)

        for fsp, boundary_pixel, boundary_dtf in bdy:
            boundary_elev = float(fil[boundary_pixel])
            max_wse = min(mxht, boundary_elev + fldht - boundary_dtf)
            additions = _backfill_from_source(
                boundary_pixel, boundary_dtf, max_wse, fil, fdr, nrows, ncols,
                bg, flood_members=flood_members
            )
            for pixel, dtf in additions:
                if _assimilate(records, fsp, pixel, dtf):
                    flddat[pixel] = max(float(fldmn), float(dtf))
                    flood_members.add(pixel)

        bdy = _boundary(records, fil, nrows, ncols, bg)
        spill = True
        while spill:
            before = len(records)
            flddat = _flood_map(records, fil.size, fldmn)
            candidates = _spill_candidates(
                bdy, records, flddat, fil, nrows, ncols, fldht, mxht, bg, excluded
            )

            for fsp, spill_dtf, pixel, pixel_elev, _ in candidates:
                path = _forward_path(pixel, spill_dtf, fil, fdr, flddat, nrows, ncols, bg, excluded)
                if not path:
                    continue
                path_set = set(path)
                path_stage = min(mxht, pixel_elev + fldht - spill_dtf)
                path_depth = path_stage - pixel_elev

                pending: list[tuple[int, float]] = [(p, 0.0) for p in path]
                for source in path:
                    source_wse = float(fil[source]) + path_depth
                    pending.extend(
                        _backfill_from_source(
                            source, 0.0, source_wse, fil, fdr, nrows, ncols,
                            bg, flood_members=None, excluded=path_set
                        )
                    )

                for new_pixel, local_dtf in pending:
                    _assimilate(records, fsp, new_pixel, local_dtf + spill_dtf)

            bdy = _boundary(records, fil, nrows, ncols, bg)
            if ssflg:
                if len(records) > before and any(dtf < fldht for _, _, dtf in bdy):
                    bdy = [row for row in bdy if row[2] < fldht]
                    spill = len(bdy) > 0
                else:
                    spill = False
            else:
                spill = False

        proctime.append((fldht, time.perf_counter() - t0))

    rows: list[list[float]] = []
    for pixel, (fsp, dtf) in records.items():
        out_dtf = max(fldmn, dtf)
        # fill_adjusted = out_dtf + fil[pixel] - max(0.0, dem[pixel])
        fill_adjusted = fil[pixel] - max(0.0, dem[pixel])
        rows.append([fsp, pixel, out_dtf, fill_adjusted])

    fldpln = np.asarray(rows, dtype=np.float64)
    header = [
        "reference stream pixel",
        "floodplain pixel",
        "flood height",
        "fill depth",
    ]
    flood_map = _flood_map(records, fil.size, fldmn).reshape((nrows, ncols))

    if save_mat is not None:
        savemat(str(save_mat), {"fldpln_py": fldpln, "header": header})

    return FldplnResult(fldpln=fldpln, header=header, flood_map=flood_map, processing_times=proctime)


__all__ = ["FldplnResult", "fldpln_model_v5ram"]

if __name__ == "__main__":
    fldpln_model_v5ram(
        1,
        {"seg": loadmat(r"C:\Users\lrr43\Downloads\wildcat_10m_3dep\segs\seg_info.mat")['seg_info']},
        gdal.Open(r"C:\Users\lrr43\Downloads\wildcat_10m_3dep\bil\fil.bil").ReadAsArray(),
        gdal.Open(r"C:\Users\lrr43\Downloads\wildcat_10m_3dep\bil\fdr.bil").ReadAsArray(),
        0.01,
        10,
        1,
        0,
        True,
        save_mat=r"C:\Users\lrr43\Downloads\wildcat_10m_3dep\test.mat",
        demdat=gdal.Open(r"C:\Users\lrr43\Downloads\wildcat_10m_3dep\bil\dem.bil").ReadAsArray(),
    )


    import argparse

    parser = argparse.ArgumentParser(description="Run the fldpln_model_v5ram function with .npy inputs.")
    parser.add_argument("seg0", type=int, help="Segment id to process.")
    parser.add_argument("inp_seg_info", type=str, help="Path to .npy file containing segment info matrix.")
    parser.add_argument("fildat", type=str, help="Path to .npy file containing filled DEM data.")
    parser.add_argument("fdrdat", type=str, help="Path to .npy file containing flow direction data.")
    parser.add_argument("fldmn", type=float, help="Minimum flood depth.")
    parser.add_argument("fldmx", type=float, help="Maximum flood depth.")
    parser.add_argument("dh", type=float, help="Flood depth increment.")
    parser.add_argument("mxht0", type=float, help="Maximum water surface elevation (0 for no limit).")
    parser.add_argument("--ssflg", action="store_true", help="Enable spillover iterations.")
    parser.add_argument("--bg", type=float, default=0.0, help="Background value in rasters indicating no data.")
    parser.add_argument("--shape", type=int, nargs=2, metavar=("NROWS", "NCOLS"), help="Shape of raster inputs if they are 1D.")
    parser.add_argument("--demdat", type=str, default=None, help="Optional path to .npy file containing DEM data (if different from filled DEM).")
    parser.add_argument("--save_mat", type=str, default=None, help="Optional path to save output as a MATLAB .mat file.")

    args = parser.parse_args()

    demdat_array = gdal.Open(args.demdat).ReadAsArray() if args.demdat else None
    result = fldpln_model_v5ram(
        seg0=args.seg0,
        inp={"seg": loadmat(args.inp_seg_info)["seg_info"]},
        fildat=gdal.Open(args.fildat).ReadAsArray(),
        fdrdat=gdal.Open(args.fdrdat).ReadAsArray(),
        fldmn=args.fldmn,
        fldmx=args.fldmx,
        dh=args.dh,
        mxht0=args.mxht0,
        ssflg=args.ssflg,
        bg=args.bg,
        shape=tuple(args.shape) if args.shape else None,
        demdat=demdat_array,
        save_mat=args.save_mat,
    )
    print("FLDPLN table:")
    print(result.fldpln)
    print(len(result.fldpln), "floodplain pixels")
    print("Processing times (flood height, seconds):")
    for height, seconds in result.processing_times:
        print(f"{height:.2f}, {seconds:.4f}")