# Python API

Everything on this page is importable from `curve2flood`.

## Running a model

`Curve2Flood_MainFunction` runs Curve2Flood from Python, as the `curve2flood` command does.

::: curve2flood.Curve2Flood_MainFunction

`Curve2Flood_Batch` runs several input files at once, as `Curve2Flood_MainFunction` runs each. The FLDPLN flood maps
among them that share a library are mapped together: reading the library takes most of a flood map's time, so it is
read once for all of their flow events (as many as fit in memory) rather than twice for each run. On a flat tile, six
return periods took 52 s rather than 231 s.

```python
from curve2flood import Curve2Flood_Batch

Curve2Flood_Batch([f"inputs_rp{rp}.yaml" for rp in (2, 5, 10, 25, 50, 100)])
```

::: curve2flood.Curve2Flood_Batch

## Settings

`Config`'s fields are the input-file keys, and [Input File](input_file.md) describes each one. A `Config` can't be
changed, but `dataclasses.replace` makes a copy with some fields changed (and checks it again):

```python
from dataclasses import replace
from curve2flood import Config, Curve2Flood_MainFunction

config = Config.from_file("inputs.yaml")
Curve2Flood_MainFunction(replace(config, outdep="depth.tif"))
```

::: curve2flood.Config
    options:
      merge_init_into_class: false
      group_by_category: false
      separate_signature: false

## FLDPLN library

`build_fldpln_library` builds the floodplain library the FLDPLN mapper reads (see [FLDPLN Mapper](fldpln.md)).

::: curve2flood.build_fldpln_library

## Helpers

::: curve2flood.remove_cells_not_connected
