import json
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
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

# zstd's level for a Parquet library.
_ZSTD_LEVEL = 6

# The key of a Parquet library's metadata (see library_metadata)
METADATA_KEY = b"curve2flood:fldpln_library"
# The version of that metadata's layout
METADATA_VERSION = 1


def library_metadata(path: str | os.PathLike) -> dict | None:
    """What a Parquet library says about itself, or None for a CSV library or one built before it said anything.

    ``nrows``, ``ncols``, ``geotransform`` and ``crs`` (WKT) are the grid
    its pixel ids number, so a flood map needs no DEM to know where it is;
    ``cell_size_m`` is its cells' size in metres, ``build`` the build's
    settings and ``curve2flood`` the version that built it.
    """
    if not is_parquet(path):
        return None
    
    metadata = pq.read_schema(path).metadata or {}
    if METADATA_KEY not in metadata:
        return None
    
    return json.loads(metadata[METADATA_KEY])


class LibraryWriter:
    """Writes a library a part at a time: Parquet as ``save_library`` saves it, in row groups of
    ``_LIBRARY_SLICE_ROWS`` rows whatever the parts' sizes, or CSV.

    The file is written beside ``path`` and takes its name only once closed
    without an error, so a build that fails or is killed never leaves a
    library that looks finished.  ``parquet`` is from ``path``'s suffix when
    None.  With ``background``, ``write_later`` makes and writes the rows in
    a thread of the writer's own, at most ``_WRITES_AHEAD`` parts behind.
    ``metadata`` (see ``library_metadata``) is stored in a Parquet library;
    a CSV library has nowhere to keep it.
    """

    def __init__(self, path: str | os.PathLike, parquet: bool | None = None, background: bool = False,
                 metadata: dict | None = None):
        self.path = Path(path)
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
                finished = True
        finally:
            if self._writer is not None:
                self._writer.close()
            if finished:
                os.replace(self._partial, self.path)
            else:
                self._partial.unlink(missing_ok=True)

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
    places, at a quarter of its cost."""
    hundredths = np.round(values.astype(np.float32).round(2).astype(np.float64) * 100.0)
    limit = 10.0 ** kind.precision
    if not (np.abs(hundredths) < limit).all():
        raise ValueError(f"A value doesn't fit {kind}: {values[~(np.abs(hundredths) < limit)][:3]}.")
    
    return pa.Array.from_buffers(kind, hundredths.size, [None, pa.py_buffer(hundredths.astype(np.int32))])
