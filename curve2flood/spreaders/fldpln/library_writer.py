import base64
import json
import os
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
from numba import njit
import pyarrow.parquet as pq

from curve2flood.config import is_parquet

# Segments whose rows may wait for the library's writer thread
_WRITES_AHEAD = 4

# Columns stored as 2-decimal numbers.  A fill can be deep (a pit filled to its
# rim), so it gets more digits than a DTF, which is at most the library's fldmx.
DECIMAL_COLS = {"DTF": pa.decimal32(5, 2), "fill": pa.decimal32(7, 2)}

# Library rows written and read at a time: one row group of a library written by
# save_library.
_LIBRARY_SLICE_ROWS = 1 << 20

# zstd's level for a Parquet library.  On a flat tile's library of a billion rows, 3 wrote it in two thirds of 6's
# time, 4% larger, and read as fast; 1 and lz4 were read slower.
_ZSTD_LEVEL = 3

# The key of a Parquet library's metadata (see library_metadata)
METADATA_KEY = b"curve2flood:fldpln_library"
# The version of that metadata's layout
METADATA_VERSION = 1
# The key of a Parquet library's footer metadata that keeps its cells' fill (see fill_footer), and the version of its
# layout
FILL_KEY = b"curve2flood:fldpln_fill"
FILL_VERSION = 1


def library_metadata(path: str | os.PathLike) -> dict | None:
    """What a Parquet library says about itself, or None for a CSV library or one built before it said anything.

    ``nrows``, ``ncols``, ``geotransform`` and ``crs`` (WKT) are the grid
    its pixel ids number, so a flood map needs no DEM to know where it is;
    ``cell_size_m`` is its cells' size in metres, ``build`` the build's
    settings and ``curve2flood`` the version that built it.  A library
    built by ``build_fldpln_library`` keeps its cells' fill elsewhere in
    its footer (see ``library_fill``).
    """
    if not is_parquet(path):
        return None
    
    metadata = pq.read_schema(path).metadata or {}
    if METADATA_KEY not in metadata:
        return None
    
    return json.loads(metadata[METADATA_KEY])


def fill_footer(cells: np.ndarray, fill: np.ndarray) -> dict[bytes, bytes]:
    """The footer metadata that keeps a library's fill once a cell, rather than in a column: ``fill`` (metres) at
    ``cells`` (flat pixel ids, ascending, each once), in hundredths of a metre as a fill column stores it
    (``DECIMAL_COLS``), and only where that isn't 0.

    A cell's fill is the same in every row of it, but a cell is in many
    rows, millions of rows apart, and Parquet compresses each page on its
    own: a column of them took a third of a flat tile's library (416 of
    1268 MB, for 80 rows a cell).  Once a cell, its 3.8 million fills took
    5 MB.

    The value is JSON: ``version`` (``FILL_VERSION``), ``cells``, how
    many cells have a fill, and ``data``, the base64 of zstd's
    compression of their int32 gaps (from 0 to the first cell, and then
    from each to the next) and then of their int32 fills in hundredths,
    each array split into its four little-endian byte streams, as
    Parquet's BYTE_STREAM_SPLIT does.  Base64, because polars won't open
    a Parquet file whose metadata isn't UTF-8.
    """
    hundredths, bad = _hundredths(np.ascontiguousarray(fill, dtype=np.float32),
                                  10.0 ** DECIMAL_COLS["fill"].precision)
    if bad >= 0:
        raise ValueError(f"A fill doesn't fit {DECIMAL_COLS['fill']}: {fill[bad:bad + 3]}.")

    kept = hundredths != 0
    cells = np.asarray(cells, dtype=np.int64)[kept]
    gaps = np.diff(cells, prepend=0)
    if cells.size and (cells[0] < 0 or cells[-1] >= 2 ** 31 or (gaps[1:] <= 0).any()):
        raise ValueError("A library's fill needs its cells' ids ascending, each once, from 0 up to 2**31 - 1.")

    data = _split(gaps) + _split(hundredths[kept])
    value = {"version": FILL_VERSION, "cells": int(cells.size),
             "data": base64.b64encode(pa.Codec("zstd", compression_level=_ZSTD_LEVEL).compress(data, asbytes=True))
             .decode("ascii")}
    return {FILL_KEY: json.dumps(value).encode()}


def library_fill(path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray] | None:
    """A Parquet library's fill kept in its footer (``fill_footer``): the cells whose fill isn't 0, as int64
    flat pixel ids, ascending, and their fills, int32 hundredths of a metre.  None for a library without it, whose
    fill is its ``fill`` column (a CSV library, or one written by ``save_library``)."""
    if not is_parquet(path):
        return None

    value = (pq.ParquetFile(path).metadata.metadata or {}).get(FILL_KEY)
    if value is None:
        return None

    fill = json.loads(value)
    if fill.get("version") != FILL_VERSION:
        raise ValueError(f"{path} keeps its fill in layout version {fill.get('version')}; this curve2flood reads "
                         f"version {FILL_VERSION}.")

    n = int(fill["cells"])
    if n == 0:
        return np.empty(0, np.int64), np.empty(0, np.int32)

    data = pa.Codec("zstd").decompress(base64.b64decode(fill["data"]), decompressed_size=8 * n, asbytes=True)
    return np.cumsum(_unsplit(data[:4 * n]), dtype=np.int64), _unsplit(data[4 * n:])


def _split(values: np.ndarray) -> bytes:
    """``values`` as little-endian int32, their bytes split into four streams: each value's first byte, then
    second, third and fourth."""
    return np.ascontiguousarray(values, dtype="<i4").view(np.uint8).reshape(-1, 4).T.tobytes()


def _unsplit(data: bytes) -> np.ndarray:
    """The int32 values of ``_split``'s bytes."""
    return np.frombuffer(data, np.uint8).reshape(4, -1).T.copy().view("<i4").ravel().astype(np.int32, copy=False)


class LibraryWriter:
    """Writes a library a part at a time: Parquet as ``save_library`` saves it, in row groups of
    ``_LIBRARY_SLICE_ROWS`` rows whatever the parts' sizes, or CSV.

    The file is written beside ``path`` and takes its name only once closed
    without an error, so a build that fails or is killed never leaves a
    library that looks finished.  ``parquet`` is from ``path``'s suffix when
    None.  With ``background``, ``write_later`` makes and writes the rows in
    a thread of the writer's own, at most ``_WRITES_AHEAD`` parts behind.
    ``metadata`` (see ``library_metadata``) is stored in a Parquet library;
    a CSV library has nowhere to keep it.  ``footer``, if given, is called
    once all the rows are written, and its key-value metadata is added to a
    Parquet library's footer (see ``fill_footer``); it isn't called for a
    CSV library.
    """

    def __init__(self, path: str | os.PathLike, parquet: bool | None = None, background: bool = False,
                 metadata: dict | None = None, footer: Callable[[], dict[bytes, bytes]] | None = None):
        self.path = Path(path)
        self._footer = footer
        self._thread = ThreadPoolExecutor(1) if background else None
        self._queued = deque()
        self.parquet = is_parquet(self.path) if parquet is None else parquet
        self._metadata = None if metadata is None else {METADATA_KEY: json.dumps(metadata).encode()}
        self._partial = self.path.with_name(self.path.name + ".partial")
        self._begun = False
        self._writer = None
        self._pending: list[dict[str, np.ndarray]] = []
        self._n_pending = 0
        self._empty = None

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        finished = False
        try:
            if self._thread is not None:
                try:
                    while self._queued and kind is None:
                        self._queued.popleft().result()
                finally:
                    self._thread.shutdown(wait=True, cancel_futures=True)
            if kind is None:
                self._flush(final=True)
                if self._footer is not None and self._writer is not None:
                    self._writer.add_key_value_metadata(self._footer())
                finished = True
        finally:
            if self._writer is not None:
                self._writer.close()
            if finished:
                self._replace()
            else:
                self._partial.unlink(missing_ok=True)

    def _replace(self):
        """The finished file takes the library's name.  On Windows a file another program has open (QGIS, GDAL, a
        Python process reading it) can't be replaced, and Windows only says "Access is denied": the error says so, and
        the new library is kept beside it, as ``.partial``, rather than built again."""
        try:
            os.replace(self._partial, self.path)
        except PermissionError as error:
            raise PermissionError(
                f"The library was built, but couldn't replace {self.path}: another program may have it open (on "
                f"Windows, a file open in QGIS, GDAL or another Python process can't be replaced). The new library is "
                f"{self._partial}: close that program and rename it to {self.path.name}.") from error

    def write_later(self, make_rows, *args):
        """``write(make_rows(*args))``, in the writer's thread if it has one: the rows are made and compressed
        while the caller goes on.  An error there is raised by a later call, or on closing."""
        if self._thread is None:
            self.write(make_rows(*args))
            return
        while len(self._queued) >= _WRITES_AHEAD:
            self._queued.popleft().result()
        self._queued.append(self._thread.submit(lambda: self.write(make_rows(*args))))

    def write(self, columns: dict[str, np.ndarray]):
        """Add rows: each column's values by name, the same columns every time."""
        if self._empty is None:
            self._empty = {c: v[:0] for c, v in columns.items()}
        n = len(next(iter(columns.values()))) if columns else 0
        if not n:
            return
        self._pending.append(columns)
        self._n_pending += n
        if self._n_pending >= _LIBRARY_SLICE_ROWS:
            self._flush(final=False)

    def _flush(self, final: bool):
        """Write the pending rows: every whole row group of them, and the rest too if ``final``."""
        if not self._pending:
            if final and not self._begun and self._empty is not None:
                self._write_part(self._empty)  # a library without rows still has its columns
            return
        names = list(self._pending[0])
        if len(self._pending) == 1:
            rows = self._pending[0]
        else:
            rows = {c: np.concatenate([part[c] for part in self._pending]) for c in names}
        n = self._n_pending
        stop = n if final else n - n % _LIBRARY_SLICE_ROWS
        step = _LIBRARY_SLICE_ROWS if self.parquet else max(stop, 1)
        for start in range(0, stop, step):
            self._write_part({c: v[start:min(start + step, stop)] for c, v in rows.items()})
        # A copy, so that the rows written are freed
        self._pending = [{c: v[stop:].copy() for c, v in rows.items()}] if stop < n else []
        self._n_pending = n - stop

    def _write_part(self, part: dict[str, np.ndarray]):
        if not self.parquet:
            import pandas as pd  # Imported here: only a CSV library needs it

            pd.DataFrame(part).to_csv(self._partial, index=False, mode="a" if self._begun else "w",
                                      header=not self._begun)
            self._begun = True
            return
        
        arrays = {}
        for c, values in part.items():
            arrays[c] = _as_decimal(values, DECIMAL_COLS[c]) if c in DECIMAL_COLS else pa.array(values)

        table = pa.table(arrays)
        if self._metadata is not None:
            table = table.replace_schema_metadata(self._metadata)
            
        if not self._begun:
            encoding = {"FPP": "DELTA_BINARY_PACKED"} | {c: "BYTE_STREAM_SPLIT" for c in DECIMAL_COLS if c in part}
            self._writer = pq.ParquetWriter(
                self._partial, table.schema, use_dictionary=[c for c in ("FSP", "segment") if c in part],
                column_encoding=encoding, store_decimal_as_integer=True, compression="zstd",
                compression_level=_ZSTD_LEVEL)
            self._begun = True
        self._writer.write_table(table, row_group_size=_LIBRARY_SLICE_ROWS)


def _as_decimal(values: np.ndarray, kind: pa.Decimal32Type) -> pa.Array:
    """``values`` (metres) as ``kind``, a decimal32 of hundredths, as Arrow's cast of them as float32 rounded to two
    places, at a fraction of its cost."""
    hundredths, bad = _hundredths(np.ascontiguousarray(values, dtype=np.float32), 10.0 ** kind.precision)
    if bad >= 0:
        raise ValueError(f"A value doesn't fit {kind}: {values[bad:bad + 3]}.")

    return pa.Array.from_buffers(kind, hundredths.size, [None, pa.py_buffer(hundredths)])


@njit(cache=True, nogil=True)
def _hundredths(values: np.ndarray, limit: float) -> tuple[np.ndarray, int]:
    """float32 ``values`` as int32 hundredths, in one pass: NumPy's float32 ``round(2)`` (times 100, to the nearest
    even, divided by 100, all in float32), then times 100 in float64 and to the nearest even again.  Also where the
    first that isn't below ``limit`` in size (or isn't a number) is, or -1."""
    out = np.empty(values.size, dtype=np.int32)
    hundred = np.float32(100.0)
    for i in range(values.size):
        rounded = np.float32(np.rint(values[i] * hundred) / hundred)
        h = np.rint(np.float64(rounded) * 100.0)
        if not abs(h) < limit:
            return out, i
        out[i] = np.int32(h)

    return out, -1
