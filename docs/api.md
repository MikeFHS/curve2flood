# Python API

Everything on this page is importable from `curve2flood`.

## Running a model

`Curve2Flood_MainFunction` runs Curve2Flood from Python, as the `curve2flood` command does.

::: curve2flood.Curve2Flood_MainFunction

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
