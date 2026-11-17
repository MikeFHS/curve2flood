"""Benchmark Curve2Flood's three mappers on synthetic valleys, and on real sites when you have them.

    python benchmarks/mappers/run_benchmark.py                       # the synthetic valleys
    python benchmarks/mappers/run_benchmark.py --site my_site.yaml   # and a real site too (as many as you like)

Nothing is stored but this script and its results: each synthetic valley is made in a temporary folder as it runs.
A valley is a DEM, its streams, a VDT database whose rating curves come from the valley's own cross section
(Manning's equation for uniform flow, the bed slope along the stream), the flows of three events (about 1, 3 and 5 m
deep), and, for FLDPLN, the filled DEM, its D8 flow directions, the stream table and the library built from them.

The truth for an event is the flood those rating curves imply: each cell's water surface is its nearest stream
cell's (interpolated in the VDT, as the mappers interpolate it), and the cell floods where that is above the ground
and it connects to a stream. This is a consistency benchmark, of how faithfully each mapper turns the same rating
curves into a flood map, and not a hydraulic one: the water surface of a real flood isn't its nearest stream cell's.

A real site is a YAML file of Curve2Flood's input keys (``DEM_File``, ``Print_VDT_Database``, ``COMID_Flow_File``,
and for FLDPLN ``Flow_Direction_File``, ``Stream_Info_File`` and ``FLDPLN_Library``; the flow file's first event is
mapped) and ``reference``, a raster of the observed flood on the DEM's grid (above 0 is wet), with an optional
``boundary`` raster (above 0 is scored). Each mapper whose inputs are there is scored against the reference.

Results go to results.md and results.json beside this script (or ``--out``): CSI and MCC of the flood extent,
the depth's bias and RMSE over the cells both call wet, and the mapping time of an event, warm.
"""
import argparse
import heapq
import json
import math
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from osgeo import gdal, osr
from scipy import ndimage

import curve2flood
from curve2flood import Curve2Flood_MainFunction, build_fldpln_library

gdal.UseExceptions()

MAPPERS = {
    "kernel-weighted": "Curve2Flood-Kernel Weighted",
    "multi-point": "Curve2Flood-Multi-Point Interpolation",
    "FLDPLN": "Curve2Flood-FLDPLNpy",
}
CELL = 10.0  # m
MANNING_N = 0.035
# Depths (m) of the VDT's entries, and of the events mapped
VDT_DEPTHS = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0])
EVENT_DEPTHS = (1.0, 3.0, 5.0)
# Whitebox D8 codes by (row, column) step
D8 = {(-1, 1): 1, (0, 1): 2, (1, 1): 4, (1, 0): 8, (1, -1): 16, (0, -1): 32, (-1, -1): 64, (-1, 0): 128}


# --------------------------------------------------------------------------
# Synthetic valleys
# --------------------------------------------------------------------------

def profile(d: np.ndarray, valley: dict) -> np.ndarray:
    """Ground above the bed at a distance ``d`` (m) from the stream: a trapezoidal channel, then a floodplain rising
    at ``floodplain_slope`` and, past ``floodplain_width``, valley walls at ``wall_slope``."""
    bottom, top, depth = valley["bottom_width"] / 2, valley["top_width"] / 2, valley["channel_depth"]
    plain = valley["floodplain_width"] / 2
    z = np.where(d <= bottom, 0.0, depth * (d - bottom) / (top - bottom))
    z = np.where(d > top, depth + valley["floodplain_slope"] * (d - top), z)
    return np.where(d > plain, depth + valley["floodplain_slope"] * (plain - top) + valley["wall_slope"] * (d - plain),
                    z)


def rating_curve(valley: dict, slope: float) -> dict[str, np.ndarray]:
    """The cross section's flow, top width and velocity at each of ``VDT_DEPTHS``, from Manning's equation."""
    d = np.arange(0.0, 4000.0, 0.25) + 0.125
    z = profile(d, valley)
    flows, widths, velocities = [], [], []
    for h in VDT_DEPTHS:
        wet = z < h
        area = 2 * np.sum(h - z[wet]) * 0.25
        perimeter = 2 * np.sum(np.hypot(0.25, np.diff(np.r_[0.0, z])[wet]))
        q = area * (area / perimeter) ** (2 / 3) * math.sqrt(slope) / MANNING_N
        flows.append(q)
        widths.append(2 * d[wet].max() + 0.125 if wet.any() else 0.0)
        velocities.append(q / area)
    return {"q": np.array(flows), "t": np.array(widths), "v": np.array(velocities)}


def _line(points: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """The cells of a polyline of (row, column) points, each the D8 neighbour of the one before."""
    cells = []
    for (r0, c0), (r1, c1) in zip(points[:-1], points[1:]):
        steps = int(max(abs(r1 - r0), abs(c1 - c0), 1) * 2)
        for t in np.linspace(0, 1, steps, endpoint=False):
            cell = (int(round(r0 + t * (r1 - r0))), int(round(c0 + t * (c1 - c0))))
            if not cells or cells[-1] != cell:
                cells.append(cell)
    cells.append((int(round(points[-1][0])), int(round(points[-1][1]))))
    return [cell for i, cell in enumerate(cells) if i == 0 or cell != cells[i - 1]]


def _priority_flood(dem: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """The DEM with its pits filled, each cell at least ``epsilon`` above the one it drains to (Barnes et al. 2014)."""
    nrows, ncols = dem.shape
    filled = dem.astype(np.float64).copy()
    closed = np.zeros(dem.shape, bool)
    heap = []
    for r in range(nrows):
        for c in range(ncols):
            if r in (0, nrows - 1) or c in (0, ncols - 1):
                heapq.heappush(heap, (filled[r, c], r, c))
                closed[r, c] = True
    while heap:
        z, r, c = heapq.heappop(heap)
        for dr, dc in D8:
            a, b = r + dr, c + dc
            if 0 <= a < nrows and 0 <= b < ncols and not closed[a, b]:
                closed[a, b] = True
                filled[a, b] = max(filled[a, b], z + epsilon)
                heapq.heappush(heap, (filled[a, b], a, b))
    return filled


def _d8(filled: np.ndarray) -> np.ndarray:
    """Whitebox D8 flow directions down the steepest drop; 0 where no neighbour is lower."""
    nrows, ncols = filled.shape
    padded = np.pad(filled, 1, constant_values=np.inf)
    best = np.zeros(filled.shape)
    fdr = np.zeros(filled.shape, np.uint8)
    for (dr, dc), code in D8.items():
        drop = (filled - padded[1 + dr:1 + dr + nrows, 1 + dc:1 + dc + ncols]) / math.hypot(dr, dc)
        better = drop > best
        best[better] = drop[better]
        fdr[better] = code
    return fdr


def make_valley(folder: Path, name: str) -> dict:
    """Write a synthetic valley's inputs into ``folder``; returns its inputs, truths and description."""
    nrows, ncols = 240, 160
    valley = {"bottom_width": 30.0, "top_width": 50.0, "channel_depth": 2.0, "floodplain_width": 900.0,
              "floodplain_slope": 0.01, "wall_slope": 0.15}
    slope = 0.001
    middle = ncols / 2
    if name == "straight":
        reaches = [[(0, middle), (nrows - 1, middle)]]
    elif name == "meander":
        rows = np.arange(0, nrows, 2.0)
        reaches = [list(zip(rows, middle + 25 * np.sin(rows / 240 * 4 * np.pi)))]
    elif name == "confluence":
        reaches = [[(0, middle), (nrows - 1, middle)], [(0, 15), (60, 40), (120, middle)]]
    elif name == "levee_pond":
        reaches = [[(0, 50), (nrows - 1, 50)]]
    elif name == "steep":
        slope = 0.02
        valley.update(bottom_width=10.0, top_width=20.0, channel_depth=1.0, floodplain_width=200.0,
                      floodplain_slope=0.05, wall_slope=0.4)
        reaches = [[(0, middle), (nrows - 1, middle)]]
    else:
        raise ValueError(name)

    # The streams, and each cell's distance downstream along them, from which its bed falls at the slope
    lines = [_line(points) for points in reaches]
    if len(lines) > 1:
        junction = lines[0].index(lines[1][-1])
        lines[1] = lines[1][:-1]  # the tributary ends where it meets the main stream
    along = []
    for i, line in enumerate(lines):
        steps = np.r_[0.0, np.hypot(*np.diff(np.array(line, float), axis=0).T) * CELL]
        distance = np.cumsum(steps)
        if i == 1:  # the tributary's bed meets the main stream's at the junction
            meet = along[0][junction] - (distance[-1] + math.hypot(*np.subtract(lines[0][junction], line[-1])) * CELL)
            distance = distance + meet
        along.append(distance)
    stream_id = np.zeros((nrows, ncols), np.int64)
    downstream = np.full((nrows, ncols), np.nan)
    for i, (line, distance) in enumerate(zip(lines, along)):
        for (r, c), s in zip(line, distance):
            if stream_id[r, c] == 0:
                stream_id[r, c], downstream[r, c] = i + 1, s

    # The ground: the nearest stream cell's bank plus the cross section's rise from the bank
    distance, (near_r, near_c) = ndimage.distance_transform_edt(stream_id == 0, sampling=CELL, return_indices=True)
    bed = 100.0 - slope * downstream[near_r, near_c]
    dem = (bed + profile(distance, valley)).astype(np.float32)
    if name == "levee_pond":
        # A levee 8 m above the bed along the whole valley, 150 m from the stream, and behind it a pond whose floor
        # is below the stream's bed: ground below every event's water that none of it reaches
        rows, cols = np.mgrid[0:nrows, 0:ncols]
        dem = np.where((np.abs(cols - 76) <= 4) & (np.abs(rows - 120) <= 30), dem - 6.0, dem)
        dem = np.where(np.abs(cols - 65) <= 1, bed + 8.0, dem).astype(np.float32)

    # Flow directions down the filled DEM, but along the streams on them
    filled = _priority_flood(dem)
    fdr = _d8(filled)
    for line in lines:
        for (r, c), (r2, c2) in zip(line[:-1], line[1:]):
            fdr[r, c] = D8[(r2 - r, c2 - c)]
    if len(lines) > 1:
        r, c = lines[1][-1]
        r2, c2 = lines[0][junction]
        fdr[r, c] = D8[(int(np.sign(r2 - r)), int(np.sign(c2 - c)))]
    last_r, last_c = lines[0][-1]
    fdr[last_r, last_c] = D8[(1, 0)]

    # The VDT: each stream cell's rating curve, from its cross section and the bed slope
    curve = rating_curve(valley, slope)
    rows_vdt = []
    for i, line in enumerate(lines):
        cells = np.array(line)
        heading = np.gradient(cells.astype(float), axis=0)
        for k, (r, c) in enumerate(line):
            if stream_id[r, c] != i + 1:
                continue
            row = {"COMID": i + 1, "Row": r, "Col": c, "Elev": float(dem[r, c]), "QBaseflow": float(curve["q"][1]),
                   "Slope": slope, "XS_Angle": math.atan2(-heading[k, 1], heading[k, 0]) % math.pi}
            for j in range(len(VDT_DEPTHS)):
                row.update({f"q_{j + 1}": curve["q"][j], f"t_{j + 1}": curve["t"][j],
                            f"wse_{j + 1}": float(dem[r, c]) + VDT_DEPTHS[j], f"v_{j + 1}": curve["v"][j]})
            rows_vdt.append(row)
    vdt = pd.DataFrame(rows_vdt)

    # The events: the flows about 1, 3 and 5 m deep, for every reach
    flows = np.interp(EVENT_DEPTHS, VDT_DEPTHS, curve["q"])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32612)
    paths = {}
    for raster, array, kind in (("dem", dem, gdal.GDT_Float32), ("filled", filled.astype(np.float32), gdal.GDT_Float32),
                                ("flowdir", fdr, gdal.GDT_Byte)):
        paths[raster] = folder / f"{raster}.tif"
        ds = gdal.GetDriverByName("GTiff").Create(str(paths[raster]), ncols, nrows, 1, kind)
        ds.SetGeoTransform((400000.0, CELL, 0.0, 4500000.0, 0.0, -CELL))
        ds.SetProjection(srs.ExportToWkt())
        ds.GetRasterBand(1).WriteArray(array)
        ds = None
    vdt.to_parquet(folder / "vdt.parquet")
    events = {f"q{i + 1}": [q] * len(lines) for i, q in enumerate(flows)}
    for event, values in events.items():
        pd.DataFrame({"COMID": range(1, len(lines) + 1), event: values}).to_csv(folder / f"{event}.csv", index=False)
    info = [(line[0][0] * ncols + line[0][1], line[-1][0] * ncols + line[-1][1], len(line), i + 1)
            for i, line in enumerate(lines)]
    pd.DataFrame(info, columns=["start_pixel", "end_pixel", "length", "stream_id"]).to_parquet(folder / "info.parquet")

    # Each event's truth
    vdt_cells = vdt.set_index(["Row", "Col"])
    q_cols, wse_cols = [f"q_{j + 1}" for j in range(len(VDT_DEPTHS))], [f"wse_{j + 1}" for j in range(len(VDT_DEPTHS))]
    truths = {}
    for event, q in zip(events, flows):
        wse_stream = np.full((nrows, ncols), np.nan)
        for (r, c), cell in vdt_cells.iterrows():
            wse_stream[r, c] = np.interp(q, cell[q_cols].to_numpy(float), cell[wse_cols].to_numpy(float))
        wse = wse_stream[near_r, near_c]
        wet = (wse > dem)
        labels, _ = ndimage.label(wet, structure=np.ones((3, 3)))
        kept = np.isin(labels, np.unique(labels[(stream_id > 0) & wet]))
        truths[event] = {"wet": kept & (labels > 0), "depth": np.where(kept, wse - dem, np.nan)}

    inputs = {"DEM_File": paths["dem"], "Print_VDT_Database": folder / "vdt.parquet",
              "Flow_Direction_File": paths["flowdir"], "Stream_Info_File": folder / "info.parquet",
              "FLDPLN_Library": folder / "library.parquet", "Make_Output_GPKG": False}
    started = time.perf_counter()
    build_fldpln_library(dem=str(paths["dem"]), filled_dem=str(paths["filled"]),
                         stream_info_file=str(folder / "info.parquet"), flow_direction_file=str(paths["flowdir"]),
                         library_file=str(folder / "library.parquet"), vdt_file=str(folder / "vdt.parquet"),
                         fldmx=12.0, pbar=False)
    build_seconds = time.perf_counter() - started
    return {"inputs": inputs, "events": list(events), "truths": truths, "build_seconds": build_seconds,
            "folder": folder, "cells": nrows * ncols}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def score(wet: np.ndarray, truth: np.ndarray, depth: np.ndarray | None = None, true_depth: np.ndarray | None = None,
          inside: np.ndarray | None = None) -> dict:
    """CSI and MCC of ``wet`` against ``truth`` (inside ``inside``), and the depth's bias and RMSE where both are
    wet."""
    if inside is not None:
        wet, truth = wet[inside], truth[inside]
        depth = None if depth is None else depth[inside]
        true_depth = None if true_depth is None else true_depth[inside]
    tp = float(np.sum(wet & truth))
    fp = float(np.sum(wet & ~truth))
    fn = float(np.sum(~wet & truth))
    tn = float(np.sum(~wet & ~truth))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    result = {"csi": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
              "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
              "wet_km2": None, "truth_km2": None}
    if depth is not None and true_depth is not None:
        both = wet & truth & np.isfinite(depth) & np.isfinite(true_depth)
        error = depth[both] - true_depth[both]
        result.update(depth_bias=float(error.mean()) if error.size else None,
                      depth_rmse=float(np.sqrt((error ** 2).mean())) if error.size else None)
    return result


def run_mapper(mapper: str, inputs: dict, flows: Path, out: Path, with_depth: bool) -> tuple[float, np.ndarray,
                                                                                             np.ndarray | None]:
    """Map one event with ``mapper``: its time (the second of two runs, warm), the flood and the depth."""
    args = {**inputs, "mapper": MAPPERS[mapper], "COMID_Flow_File": flows, "OutFLD": out / "flood.tif"}
    if mapper != "FLDPLN":
        for key in ("Flow_Direction_File", "Stream_Info_File", "FLDPLN_Library"):
            args.pop(key, None)
    if with_depth:
        args["OutDEP"] = out / "depth.tif"
    seconds = []
    for _ in range(2):
        started = time.perf_counter()
        Curve2Flood_MainFunction(args=args, quiet=True)
        seconds.append(time.perf_counter() - started)
    wet = gdal.Open(str(out / "flood.tif")).ReadAsArray() > 0
    depth = None
    if with_depth:
        depth = gdal.Open(str(out / "depth.tif")).ReadAsArray().astype(np.float64)
        depth[depth == -9999.0] = np.nan
    return seconds[-1], wet, depth


def synthetic(names: list[str]) -> list[dict]:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            folder = Path(tmp) / name
            folder.mkdir()
            valley = make_valley(folder, name)
            for event, depth_m in zip(valley["events"], EVENT_DEPTHS):
                truth = valley["truths"][event]
                for mapper in MAPPERS:
                    seconds, wet, depth = run_mapper(mapper, valley["inputs"], folder / f"{event}.csv", folder, True)
                    result = score(wet, truth["wet"], depth, truth["depth"])
                    result.update(site=name, event=f"{depth_m:g} m deep", mapper=mapper, seconds=seconds,
                                  wet_km2=float(wet.sum() * CELL * CELL / 1e6),
                                  truth_km2=float(truth["wet"].sum() * CELL * CELL / 1e6))
                    rows.append(result)
            rows.append({"site": name, "event": "library build", "mapper": "FLDPLN",
                         "seconds": valley["build_seconds"]})
    return rows


def real_sites(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path) as f:
            site = yaml.safe_load(f)
        reference = gdal.Open(site.pop("reference")).ReadAsArray() > 0
        boundary = site.pop("boundary", None)
        inside = gdal.Open(boundary).ReadAsArray() > 0 if boundary else None
        name = site.pop("name", Path(path).stem)
        with tempfile.TemporaryDirectory() as tmp:
            flows = Path(tmp) / "first_event.csv"
            table = site.pop("COMID_Flow_File")
            table = pd.read_parquet(table) if str(table).lower().endswith((".parquet", ".pq")) else pd.read_csv(table)
            table.iloc[:, :2].to_csv(flows, index=False)
            for mapper in MAPPERS:
                needs = ("Flow_Direction_File", "Stream_Info_File", "FLDPLN_Library") if mapper == "FLDPLN" else ()
                if not all(site.get(key) for key in needs):
                    continue
                seconds, wet, _ = run_mapper(mapper, {**site, "Make_Output_GPKG": False}, flows, Path(tmp), False)
                result = score(wet, reference, inside=inside)
                result.update(site=name, event="first event", mapper=mapper, seconds=seconds)
                rows.append(result)
    return rows


def write_results(rows: list[dict], out: Path):
    out.mkdir(parents=True, exist_ok=True)
    meta = {"curve2flood": getattr(curve2flood, "__version__", None) or _version(), "python": platform.python_version(),
            "machine": platform.processor() or platform.machine()}
    (out / "results.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1), encoding="utf-8")

    def fmt(value, digits=3):
        return "" if value is None else f"{value:.{digits}f}"

    lines = ["# Mapper benchmark", "",
             "Made by `run_benchmark.py` (see its docstring for what is measured and how).", "",
             f"Curve2Flood {meta['curve2flood']}, Python {meta['python']}.", "",
             "| Site | Event | Mapper | CSI | MCC | Wet km² (truth) | Depth bias (m) | Depth RMSE (m) | Seconds |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        wet = "" if row.get("wet_km2") is None else f"{row['wet_km2']:.3f} ({row['truth_km2']:.3f})"
        lines.append(f"| {row['site']} | {row['event']} | {row['mapper']} | {fmt(row.get('csi'))} | "
                     f"{fmt(row.get('mcc'))} | {wet} | {fmt(row.get('depth_bias'), 2)} | "
                     f"{fmt(row.get('depth_rmse'), 2)} | {fmt(row.get('seconds'), 2)} |")
    scored = pd.DataFrame([row for row in rows if "csi" in row])
    if not scored.empty:
        means = scored.groupby("mapper")[["csi", "mcc", "seconds"]].mean()
        lines += ["", "Means over every site and event:", "", "| Mapper | CSI | MCC | Seconds |",
                  "| --- | --- | --- | --- |"]
        lines += [f"| {mapper} | {r.csi:.3f} | {r.mcc:.3f} | {r.seconds:.2f} |" for mapper, r in means.iterrows()]
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("curve2flood")
    except PackageNotFoundError:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--valleys", nargs="*", default=["straight", "meander", "confluence", "levee_pond", "steep"],
                        help="The synthetic valleys to map (none for only real sites).")
    parser.add_argument("--site", action="append", default=[], help="A real site's YAML file (see the docstring).")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent, help="Where to write the results.")
    args = parser.parse_args()
    rows = synthetic(args.valleys) + real_sites(args.site)
    write_results(rows, args.out)
    print((args.out / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
