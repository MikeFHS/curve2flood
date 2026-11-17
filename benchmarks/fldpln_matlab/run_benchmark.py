"""
Benchmark building an FLDPLN library with the University of Kansas's MATLAB model, its line-for-line Python port
and Curve2Flood, for time and memory, on the same segments with the same settings.

    python benchmarks/fldpln_matlab/run_benchmark.py

It needs MATLAB on the PATH, and writes results.json and results.md beside this script (docs/fldpln.md includes
results.md). It takes about a quarter of an hour.

The three build the same table: the MATLAB model (tests/fldpln_matlab/fldpln_model_v5ram.m), the port
(tests/fldpln_model_v5ram.py) and Curve2Flood's ``build_fldpln_library`` with ``solver="iterative"`` flood the same
cells from each segment with the same DTFs (the port's bit for bit, Curve2Flood's to the centimetre its libraries
keep), as the tests check on a few segments; the benchmark checks it on all of them.

Each run is a fresh process: MATLAB (``matlab -batch``) or Python. It loads its inputs, warms up on one short
segment (so MATLAB's and numba's compiling isn't timed), and builds every segment in turn, writing marker files
along the way. The time is the build's, as the process measures it. The memory is the process's working set (all
its processes'), sampled every 10 ms: "used" is the most it held while warming up and building less what it held
once its inputs were loaded, and "peak" is the most it held at all, MATLAB's or Python's own memory included.
Each implementation runs ``--repeats`` times, and the median of each figure is reported.
"""
import argparse
import importlib.util
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "tests" / "data" / "scottsbluff"
MATLAB_MODEL = ROOT / "tests" / "fldpln_matlab"
PORT = ROOT / "tests" / "fldpln_model_v5ram.py"
sys.path.insert(0, str(ROOT))  # this checkout's Curve2Flood

# Curve2Flood's defaults, but for the deepest flood, which keeps the MATLAB model's runs to a few minutes
FLDMN, FLDMX, DH = 0.1, 10.0, 0.5
ITERATIVE_SPILL = False  # spill once a step (MATLAB's ssflg 0)
BG = -9999.0
# The segments of tests/data/scottsbluff whose floods, this deep, stay on the raster: where a spill runs onto the
# raster's edge the MATLAB model stops with an error (see tests/fldpln_matlab/README.md), as it does for the other
# 31 of its 101
SEGMENTS = [
    760330156, 760410026, 760388395, 760395048, 760434982, 760366761, 760385061, 760363429, 760360101, 760361765,
    760353447, 760380068, 760072227, 760380062, 760439964, 760444958, 760493214, 760494878, 760496542, 760506527,
    760419996, 760649626, 760498203, 760534813, 760353434, 760461597, 760711193, 760388378, 760340120, 760599704,
    760416664, 760546458, 760413336, 760478233, 760599705, 760601369, 760415000, 760473237, 760536470, 760546454,
    760534808, 760484884, 760448278, 760433298, 760463251, 760446614, 760489875, 760493202, 760308499, 760310163,
    760568090, 760418328, 760478238, 760355111, 760563095, 760474901, 760536469, 760438300, 760541466, 760381733,
    760499864, 760549788, 760434985, 760583064, 760461599, 760651290, 760346788, 760491552, 760523165, 760464915,
]
WARMUP = 760363429  # a short one
THREADS = 4

IMPLEMENTATIONS = {
    "matlab": "MATLAB model",
    "port": "Python port of the MATLAB model",
    "curve2flood_1": "Curve2Flood, 1 thread",
    f"curve2flood_{THREADS}": f"Curve2Flood, {THREADS} threads",
}


# --- The runs -------------------------------------------------------------------------------------------------------

def mark(work: Path, run: str, stage: str, text: str = "") -> None:
    (work / f"{run}.{stage}").write_text(text)


def run_python(name: str, work: Path, run: str) -> None:
    """One run of a Python implementation, in this process (started by ``measure``)."""
    from osgeo import gdal
    gdal.UseExceptions()
    stream_info = pd.read_parquet(DATA / "fabdem_stream_info.parquet")
    if name == "port":
        spec = importlib.util.spec_from_file_location("fldpln_model_v5ram", PORT)
        port = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = port
        spec.loader.exec_module(port)
        fil = gdal.Open(str(DATA / "fabdem_filled.tif")).ReadAsArray().astype(np.float32)
        fdr = gdal.Open(str(DATA / "fabdem_flowdir.tif")).ReadAsArray().astype(np.int32)
        esri = esri_d8(fdr)
        seg = {"seg": port_stream_table(stream_info)}
        row = {stream_id: i for i, stream_id in enumerate(stream_info["stream_id"])}

        def build(stream_ids):
            return [np.asarray(port.fldpln_model_v5ram(row[stream_id], seg, fil, esri, FLDMN, FLDMX, DH, 0.0,
                                                       ITERATIVE_SPILL, BG, shape=fil.shape).fldpln)
                    for stream_id in stream_ids]
    else:
        from curve2flood import build_fldpln_library
        threads = int(name.rsplit("_", 1)[1])

        def build(stream_ids, library="library.parquet"):
            build_fldpln_library(str(DATA / "fabdem_filled.tif"), str(DATA / "fabdem_filled.tif"),
                                 str(DATA / "fabdem_stream_info.parquet"), str(DATA / "fabdem_flowdir.tif"),
                                 str(work / f"{run}_{library}"), dh=DH, fldmn=FLDMN, fldmx=FLDMX,
                                 iterative_spill=ITERATIVE_SPILL, stream_ids=stream_ids, bg=BG,
                                 parallel=threads > 1, processes=threads, pbar=False, max_reach_length=None,
                                 max_profile_deviation=None)
    mark(work, run, "loaded")
    if name == "port":
        build([WARMUP])
    else:
        build([WARMUP], "warmup.parquet")
    mark(work, run, "ready")
    start = time.perf_counter()
    tables = build(SEGMENTS)
    seconds = time.perf_counter() - start
    mark(work, run, "done", json.dumps({"seconds": seconds}))
    if name == "port":
        np.save(work / f"{run}_rows.npy", np.concatenate([np.column_stack([table[:, :3], np.full(len(table), k)])
                                                          for k, table in enumerate(tables)]))


def esri_d8(whitebox: np.ndarray) -> np.ndarray:
    """WhiteboxTools D8 codes as the MATLAB model's (ESRI's): each direction's code halved, north-east's 1 to 128."""
    return np.where(whitebox == 1, 128, whitebox >> 1).astype(whitebox.dtype)


def port_stream_table(stream_info: pd.DataFrame) -> np.ndarray:
    """The stream table as the MATLAB model and its port take it: start and end pixels in columns 0 and 1, the
    length in column 4."""
    table = np.zeros((len(stream_info), 5))
    table[:, 0] = stream_info["start_pixel"]
    table[:, 1] = stream_info["end_pixel"]
    table[:, 4] = stream_info["length"]
    return table


def matlab_inputs(work: Path) -> None:
    """The MATLAB model's inputs, as the University of Kansas's scripts give them: the filled DEM as float32 (single)
    with a BIL header, ESRI D8 codes, and pixels and segments counted from 1."""
    from osgeo import gdal
    from scipy.io import savemat
    gdal.UseExceptions()
    fil = gdal.Open(str(DATA / "fabdem_filled.tif")).ReadAsArray().astype(np.float32)
    fdr = gdal.Open(str(DATA / "fabdem_flowdir.tif")).ReadAsArray().astype(np.uint8)
    stream_info = pd.read_parquet(DATA / "fabdem_stream_info.parquet")
    fil.astype("<f4").tofile(work / "fil.bil")
    (work / "fil.hdr").write_text(f"BYTEORDER I\nLAYOUT BIL\nNROWS {fil.shape[0]}\nNCOLS {fil.shape[1]}\nNBANDS 1\n"
                                  f"NBITS 32\nPIXELTYPE FLOAT\n")
    table = port_stream_table(stream_info)
    table[:, :2] += 1
    row = {stream_id: i + 1 for i, stream_id in enumerate(stream_info["stream_id"])}
    savemat(work / "inputs.mat", {
        "fildat": fil.reshape(-1, 1), "fdrdat": esri_d8(fdr).reshape(-1, 1), "seg_info": table,
        "segs": np.array([row[s] for s in SEGMENTS], dtype=float), "warmup": float(row[WARMUP]),
        "fldmn": FLDMN, "fldmx": FLDMX, "dh": DH, "ssflg": float(ITERATIVE_SPILL), "bg": BG})


def command(name: str, work: Path, run: str) -> tuple[list[str], dict]:
    env = dict(os.environ)
    if name == "matlab":
        matlab = shutil.which("matlab")
        if matlab is None:
            sys.exit("MATLAB is not on the PATH.")
        call = (f"addpath('{MATLAB_MODEL.as_posix()}'); addpath('{HERE.as_posix()}'); "
                f"run_matlab('{work.as_posix()}', '{run}')")
        return [matlab, "-batch", call], env
    if name == "curve2flood_1":
        env["NUMBA_NUM_THREADS"] = "1"
    return [sys.executable, str(Path(__file__).resolve()), "--run", name, str(work), run], env


def tree_memory(process: psutil.Process) -> int:
    """The working set of a process and all its children (MATLAB's launcher starts MATLAB itself), in bytes."""
    try:
        processes = [process, *process.children(recursive=True)]
    except psutil.Error:  # it has just ended
        return 0
    total = 0
    for p in processes:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total


def measure(name: str, work: Path, run: str) -> dict:
    """Run one implementation in a new process, and its time and memory."""
    args, env = command(name, work, run)
    with open(work / f"{run}.log", "w") as log:
        process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
        watched = psutil.Process(process.pid)
        loaded = used = peak = 0
        building = False
        while process.poll() is None:
            if not building and not loaded and (work / f"{run}.loaded").exists():
                building = True
            memory = tree_memory(watched)
            peak = max(peak, memory)
            if building:
                loaded = loaded or memory
                used = max(used, memory - loaded)
                if (work / f"{run}.done").exists():
                    building = False
            time.sleep(0.01)
    if process.returncode != 0 or not (work / f"{run}.done").exists():
        sys.exit(f"{name} failed: see {work / f'{run}.log'}\n{(work / f'{run}.log').read_text()[-3000:]}")
    done = json.loads((work / f"{run}.done").read_text())
    return {"seconds": done["seconds"], "used_mb": used / 2**20, "peak_mb": peak / 2**20,
            **({"matlab": done["version"]} if "version" in done else {})}


# --- The check ------------------------------------------------------------------------------------------------------

def rows(name: str, work: Path, run: str) -> pd.DataFrame:
    """An implementation's rows, ``FSP`` and ``DTF`` by segment (stream ID) and ``FPP``, pixels counted from 0."""
    if name == "matlab":
        from scipy.io import loadmat
        table = loadmat(work / f"{run}_rows.mat")["fldpln"]
        fsp, fpp, dtf = table[:, 0] - 1, table[:, 1] - 1, table[:, 2]
        segment = np.asarray(SEGMENTS)[table[:, 3].astype(np.int64) - 1]
    elif name == "port":
        table = np.load(work / f"{run}_rows.npy")
        fsp, fpp, dtf = table[:, 0], table[:, 1], table[:, 2]
        segment = np.asarray(SEGMENTS)[table[:, 3].astype(np.int64)]
    else:
        library = pd.read_parquet(work / f"{run}_library.parquet", columns=["FSP", "FPP", "DTF", "segment"])
        fsp, fpp, dtf = library["FSP"].values, library["FPP"].values, library["DTF"].astype(float).values
        # Curve2Flood numbers the segments from 0 in the stream table's order
        info = pd.read_parquet(DATA / "fabdem_stream_info.parquet")
        segment = info["stream_id"][info["stream_id"].isin(SEGMENTS)].to_numpy()[library["segment"].values]
    found = pd.DataFrame({"segment": segment, "FPP": fpp.astype(np.int64), "FSP": fsp.astype(np.int64),
                          "DTF": dtf})
    return found.set_index(["segment", "FPP"]).sort_index()


def compare(name: str, found: pd.DataFrame, matlab: pd.DataFrame) -> tuple[bool, int]:
    """Whether an implementation floods MATLAB's cells from each segment with MATLAB's DTFs (the port's bit for bit,
    Curve2Flood's to the centimetre, with DTFs below fldmn raised to it, as MATLAB raises them), and in how many rows
    it credits the cell to another stream pixel of the segment: always one that reaches it at the same depth."""
    if not found.index.equals(matlab.index):
        return False, 0
    if name == "port":
        same = np.array_equal(found["DTF"].values, matlab["DTF"].values)
    else:
        same = np.array_equal(np.maximum(found["DTF"].values.astype(np.float32), np.float32(FLDMN)),
                              np.round(matlab["DTF"].values.astype(np.float32), 2))
    return same, int((found["FSP"].values != matlab["FSP"].values).sum())


# --- The report -----------------------------------------------------------------------------------------------------

def cpu_name() -> str:
    if sys.platform == "win32":
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def machine(matlab: str) -> dict:
    import numba
    return {"cpu": cpu_name(), "cores": psutil.cpu_count(logical=False), "threads": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / 2**30), "os": f"{platform.system()} {platform.release()}",
            "matlab": matlab, "python": platform.python_version(), "numpy": np.__version__,
            "numba": numba.__version__}


def report(results: dict) -> str:
    settings, m = results["settings"], results["machine"]
    lines = [
        f"{len(SEGMENTS)} segments of `tests/data/scottsbluff` (FABDEM, {settings['ncols']} by {settings['nrows']} "
        f"cells), {settings['rows']:,} library rows, with `dh` {DH:g} m, `fldmn` {FLDMN:g} m, `fldmx` {FLDMX:g} m "
        f"and spilling once a step. Median of {settings['repeats']} runs each.",
        "",
        "| | Time (s) | Faster than MATLAB | Memory used (MB) | Peak memory (MB) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    matlab = results["runs"]["matlab"]["seconds"]
    for name, label in IMPLEMENTATIONS.items():
        run = results["runs"][name]
        lines.append(f"| {label} | {run['seconds']:.1f} | {matlab / run['seconds']:.0f}x | {run['used_mb']:.0f} | "
                     f"{run['peak_mb']:.0f} |")
    others = {results["runs"][name]["other_source_rows"] for name in IMPLEMENTATIONS if name != "matlab"}
    lines.append("")
    if others == {0}:
        lines.append("All of them give the same rows.")
    else:
        counts = " or ".join(f"{n:,}" for n in sorted(others))
        lines.append(f"All of them flood the same cells from each segment with the same DTFs. The port and Curve2Flood "
                     f"credit {counts} of the rows to another stream pixel of the segment than MATLAB does, one that "
                     f"reaches the cell at the same depth, which changes no flood map.")
    lines += ["", f"On {m['cpu']} ({m['cores']} cores, {m['threads']} threads) with {m['memory_gb']} GB, "
                  f"{m['os']}: MATLAB {m['matlab']}, Python {m['python']}, NumPy {m['numpy']}, "
                  f"numba {m['numba']}."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=3, help="runs of each implementation (default 3)")
    parser.add_argument("--keep", action="store_true", help="keep the work folder")
    parser.add_argument("--run", nargs=3, metavar=("NAME", "WORK", "RUN"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.run:
        run_python(args.run[0], Path(args.run[1]), args.run[2])
        return

    work = Path(tempfile.mkdtemp(prefix="fldpln_benchmark_"))
    try:
        matlab_inputs(work)
        runs, version, matlab_rows = {}, "", None
        for name, label in IMPLEMENTATIONS.items():
            measured = []
            for i in range(args.repeats):
                run = f"{name}_{i}"
                measured.append(measure(name, work, run))
                print(f"{label}, run {i + 1}: {measured[-1]['seconds']:.1f} s, {measured[-1]['used_mb']:.0f} MB "
                      f"used, {measured[-1]['peak_mb']:.0f} MB peak", flush=True)
            version = measured[0].get("matlab", version)
            runs[name] = {key: statistics.median(m[key] for m in measured)
                          for key in ("seconds", "used_mb", "peak_mb")}
            found = rows(name, work, run)
            if name == "matlab":
                matlab_rows = found
                continue
            same, other_source = compare(name, found, matlab_rows)
            if not same:
                sys.exit(f"{label} doesn't flood the MATLAB model's cells with its DTFs.")
            runs[name]["other_source_rows"] = other_source
        from osgeo import gdal
        gdal.UseExceptions()
        ds = gdal.Open(str(DATA / "fabdem_filled.tif"))
        results = {
            "settings": {"segments": SEGMENTS, "warmup": WARMUP, "fldmn": FLDMN, "fldmx": FLDMX, "dh": DH,
                         "iterative_spill": ITERATIVE_SPILL, "repeats": args.repeats, "rows": len(matlab_rows),
                         "nrows": ds.RasterYSize, "ncols": ds.RasterXSize},
            "machine": machine(version),
            "runs": runs,
        }
        (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        (HERE / "results.md").write_text(report(results))
        print(report(results))
    finally:
        if args.keep:
            print(f"The runs are in {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
