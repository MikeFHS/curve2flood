"""Curve2Flood's inputs: the input file, as YAML, as text, or as a dictionary."""
import difflib
import math
import os
from dataclasses import dataclass, fields
from pathlib import Path
from types import UnionType
from typing import Literal, Union, get_args, get_origin, get_type_hints

import yaml

from curve2flood._log import LOG

Mapper = Literal["Curve2Flood-Kernel Weighted", "Curve2Flood-Multi-Point Interpolation", "Curve2Flood-FLDPLNpy"]
MAPPERS = get_args(Mapper)
KERNEL_WEIGHTED, MULTI_POINT, FLDPLN = MAPPERS

# Other names for some keys, from FloodSpreader and ARC (see Config._normalize_keys)
_ALIASES = {'floodspreader_specifydepth', 'floodlocalonly', 'aroutbathy'}
# FloodSpreader's keys that Curve2Flood doesn't use, beside those it reads under other names (_ALIASES): its input
# files still run
_IGNORED_KEYS = {'outshp', 'fs_adjust_flow_by_fraction'}
# Curve2Flood's retired keys, with why: ignored with a warning, so that older input files still run
_REMOVED_KEYS = {
    'max_wse_rise': "the FLDPLN mapper no longer caps how fast the water surface rises",
    'fldpln_max_wse_rise': "the FLDPLN mapper no longer caps how fast the water surface rises",
    'median_filter_size': "the FLDPLN mapper no longer smooths the stages with a running median",
    'fldpln_median_filter_size': "the FLDPLN mapper no longer smooths the stages with a running median",
    'max_drop_below_source': "FLDPLN no longer limits how far below its stream pixel a cell may lie",
    'fldpln_max_drop_below_source': "FLDPLN no longer limits how far below its stream pixel a cell may lie",
    'filled_dem_file': "the FLDPLN mapper takes the filled DEM at the stream from the VDT's Elev and the library's fill",
}

_TRUE = {'true', 'yes', 'y', 'on', '1'}
_FALSE = {'false', 'no', 'n', 'off', '0'}

# The multi-point mapper's numeric settings, by input-file key: the least each may be, and whether it may be that
_MPI_LOWER_BOUNDS = {
    'MPI_TopWidth_Threshold_m': (0.0, True),
    'MPI_XS_Point_Spacing_m': (0.0, False),
    'MPI_HWM_Outlier_Threshold': (0.0, False),
    'MPI_HWM_Outlier_Min_Samples': (1, True),
    'MPI_Corridor_Buffer_m': (0.0, True),
    'MPI_Anchor_Buffer_m': (0.0, True),
    'MPI_Corridor_TopWidth_Factor': (0.0, True),
    'MPI_Anchor_TopWidth_Factor': (0.0, True),
    'MPI_K': (1, True),
    'MPI_Power': (0.0, True),
    'MPI_Max_Distance_m': (0.0, False),
    'MPI_MaxDist_TopWidth_Factor': (0.0, False),
    'MPI_Min_TopWidth_m': (0.0, False),
    'MPI_Fallback_TopWidth_m': (0.0, False),
    'MPI_Smooth_Sigma_Pixels': (0.0, True),
}


def path_exists(path: str | os.PathLike | None) -> bool:
    """Whether a path is given and exists."""
    return bool(path) and Path(path).exists()


def is_parquet(path: str | os.PathLike) -> bool:
    """Whether a table's path names a Parquet file (.parquet or .pq, in any case) rather than CSV."""
    return Path(path).suffix.lower() in ('.parquet', '.pq')


@dataclass(kw_only=True, frozen=True)
class Config:
    """Curve2Flood's inputs, checked and typed.

    Make one with `from_file` (YAML, or text with a key and its value on each line) or `from_mapping` (a dictionary).
    Each field is its input-file key in lower case, and keys are case-insensitive, so DEM_File and dem_file are the
    same. A key set to nothing (an empty value, or null in YAML) takes its default.
    """
    # Input files
    dem_file: str | None = None
    print_vdt_database: str | None = None
    print_curve_file: str | None = None
    comid_flow_file: str | None = None
    # Read only with set_depth; otherwise the stream cells are the VDT's or the curve file's
    stream_file: str | None = None
    lu_raster_sameres: str | None = None
    lu_manning_n: str | None = None

    # Outputs
    outfld: str | None = None
    outdep: str | None = None
    outwse: str | None = None
    outvel: str | None = None
    make_output_gpkg: bool = True
    compression: str = "LZW"

    # Mapping
    mapper: Mapper = KERNEL_WEIGHTED
    tw_multfact: float = 3.0
    topwidthplausiblelimit: float = 1000.0
    localfloodoption: bool = False
    flood_waterlc_and_strm_cells: bool = False
    flood_bathy_mask: bool = False
    land_watervalue: int = 80
    # Flood every stream cell of stream_file this deep instead of mapping the flows; off at 0 or less
    set_depth: float = -1.1

    # FLDPLN mapper
    flow_direction_file: str | None = None
    stream_info_file: str | None = None
    fldpln_library: str | None = None
    # Water shallower than this (m) floods nothing: a stream's water that just tops a rise beside it leaves a film of
    # a few millimetres over everything downhill (see _make_fldpln_flood_map)
    fldpln_min_flood_depth: float = 0.01

    # Multi-point interpolation mapper (see multi_point_options)
    mpi_topwidth_threshold_m: float = 200.0
    mpi_xs_point_spacing_m: float = 100.0
    mpi_remove_hwm_outliers: bool = True
    mpi_hwm_outlier_method: Literal["mad", "iqr"] = "mad"
    mpi_hwm_outlier_threshold: float = 3.5
    mpi_hwm_outlier_group_by: Literal["comid", "global"] = "comid"
    mpi_hwm_outlier_min_samples: int = 5
    mpi_corridor_buffer_m: float = 500.0
    mpi_anchor_buffer_m: float = 5.0
    mpi_use_topwidth_buffers: bool = True
    mpi_corridor_topwidth_factor: float = 1.0
    mpi_anchor_topwidth_factor: float = 0.1
    mpi_connectivity: Literal[4, 8] = 4
    mpi_k: int = 12
    mpi_power: float = 2.0
    mpi_max_distance_m: float = 1000.0
    mpi_use_topwidth_max_distance: bool = True
    mpi_maxdist_topwidth_factor: float = 1.0
    mpi_min_topwidth_m: float = 5.0
    mpi_max_topwidth_m: float = 500.0
    mpi_fallback_topwidth_m: float = 20.0
    mpi_smooth_sigma_pixels: float = 0.25
    mpi_apply_wse_sanity_filter: bool = True
    mpi_wse_sanity_tolerance_m: float = 0.0
    mpi_fast_mode: bool = False
    mpi_one_based_vdt_rc: bool = False

    # Bathymetry, written only when both bathy_out_file (ARC's) and fsoutbathy are given
    bathy_out_file: str | None = None
    fsoutbathy: str | None = None
    # Where the water is (its cells above 0), on the DEM's grid: flooded with flood_bathy_mask. ARC puts its banks
    # where its water ends; the bathymetry isn't clipped to it.
    bathywatermask: str | None = None
    bathy_use_banks: bool = False

    def __post_init__(self) -> None:
        self._coerce_types()
        self.validate_options()


    @classmethod
    def from_mapping(cls, params: dict):
        """The inputs from a dictionary of the input file's keys and values. Paths may be strings or Path objects."""
        return cls(**cls._normalize_keys(params))

    @classmethod
    def from_file(cls, file_path: str | os.PathLike):
        """The inputs from an input file: YAML for a .yaml or .yml file, and otherwise text."""
        if Path(file_path).suffix.lower() in ('.yaml', '.yml'):
            return cls.from_yaml(file_path)
        else:
            return cls.from_text_file(file_path)

    @classmethod
    def from_yaml(cls, yaml_name: str | os.PathLike):
        """The inputs from a YAML input file."""
        with open(yaml_name, 'r') as o_input_file:
            params = yaml.load(o_input_file, Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))
        return cls.from_mapping(params or {})

    @classmethod
    def from_text_file(cls, file_name: str | os.PathLike):
        """The inputs from a text file with a key, spaces or a tab, and its value on each line. A value may contain
        spaces. Blank lines and lines starting with # are skipped, and a key on its own takes its default."""
        params = {}
        with open(file_name, 'r') as o_input_file:
            for line in o_input_file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue  # Skip empty lines and comments
                parts = line.split(maxsplit=1)
                params[parts[0]] = parts[1] if len(parts) == 2 else ''

        return cls.from_mapping(params)

    @classmethod
    def _normalize_keys(cls, params: dict) -> dict:
        # Keys are case-insensitive, so DEM_File is dem_file
        params = {str(k).lower(): v for k, v in params.items()}

        own = {f.name for f in fields(cls) if f.init} | _ALIASES
        unknown = sorted(set(params) - own - _IGNORED_KEYS - set(_REMOVED_KEYS))
        if unknown:
            described = []
            for key in unknown:
                close = difflib.get_close_matches(key, sorted(own), n=1)
                described.append(f'{key!r}' + (f' (did you mean {close[0]!r}?)' if close else ''))
            raise ValueError(f'Unrecognized config key(s): {", ".join(described)}')
        for key in sorted(set(params) & set(_REMOVED_KEYS)):
            LOG.warning(f'{key} is no longer used ({_REMOVED_KEYS[key]}); ignoring it.')

        # A key set to nothing takes its default
        params = {key: value for key, value in params.items() if value is not None and value != ''}

        # FloodSpreader_SpecifyDepth is FloodSpreader's name for Set_Depth, which wins when both are given
        specify_depth = params.pop('floodspreader_specifydepth', None)
        if specify_depth is not None:
            params.setdefault('set_depth', specify_depth)
        # FloodLocalOnly is FloodSpreader's name for LocalFloodOption: either one turns it on
        local_only = params.pop('floodlocalonly', None)
        if local_only is not None:
            params['localfloodoption'] = (_to_bool('floodlocalonly', local_only)
                                          or _to_bool('localfloodoption', params.get('localfloodoption', False)))
        # AROutBATHY is another name for BATHY_Out_File, and wins when both are given, as it does in ARC
        aroutbathy = params.pop('aroutbathy', None)
        if aroutbathy is not None:
            params['bathy_out_file'] = aroutbathy

        # The ignored and removed keys are allowed in the inputs, but aren't fields
        known = {f.name for f in fields(cls) if f.init}
        return {key: value for key, value in params.items() if key in known}

    @classmethod
    def check_keys(cls, params: dict) -> dict:
        """Check some of an input file's keys and values without the rest of a run's: each key must be Curve2Flood's
        and each value of its type. Returns them by field name, typed. For programs that write Curve2Flood's input
        files, to check the settings a user gives them before the inputs exist; `from_mapping` checks a whole run."""
        params = cls._normalize_keys(params)
        hints = get_type_hints(cls)
        return {key: _coerce(key, value, hints[key]) for key, value in params.items()}

    def _coerce_types(self) -> None:
        # Text input files give every value as a string, so convert each one to its field's type
        hints = get_type_hints(type(self))
        for f in fields(self):
            if f.init:
                object.__setattr__(self, f.name, _coerce(f.name, getattr(self, f.name), hints[f.name]))

    def validate_options(self) -> None:
        """Check the run has every input it needs and that no two settings conflict. The messages name the
        input-file keys. Whether the files exist is `check_paths`'s to check, when the run starts."""
        if self.needs_dem and not self.dem_file:
            raise ValueError("Missing required parameters: DEM_File")
        for key, value in (('TW_MultFact', self.tw_multfact), ('TopWidthPlausibleLimit', self.topwidthplausiblelimit)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{key} must be finite and positive, not {value!r}.')
        if not math.isfinite(self.fldpln_min_flood_depth) or self.fldpln_min_flood_depth < 0.0:
            raise ValueError(f'FLDPLN_Min_Flood_Depth must be finite and 0 or more, not {self.fldpln_min_flood_depth!r}.')
        for key, (lowest, inclusive) in _MPI_LOWER_BOUNDS.items():
            value = getattr(self, key.lower())
            if not math.isfinite(value) or value < lowest or (value == lowest and not inclusive):
                least = f'{lowest} or more' if inclusive else f'more than {lowest}'
                raise ValueError(f'{key} must be {least}, not {value!r}.')
            
        if self.mpi_max_topwidth_m < self.mpi_min_topwidth_m:
            raise ValueError(f'MPI_Max_TopWidth_m ({self.mpi_max_topwidth_m}) is less than MPI_Min_TopWidth_m '
                             f'({self.mpi_min_topwidth_m}).')
        
        if not math.isfinite(self.mpi_wse_sanity_tolerance_m):
            raise ValueError(f'MPI_WSE_Sanity_Tolerance_m must be finite, not {self.mpi_wse_sanity_tolerance_m!r}.')
        
        if not self.needs_flood_map:
            return

        fldpln = self.mapper == FLDPLN
        if self.uses_set_depth and fldpln:
            raise ValueError("Set_Depth (FloodSpreader_SpecifyDepth) is not supported by the FLDPLN mapper.")
        if self.uses_set_depth and self.outvel:
            raise ValueError("OutVEL needs the slopes in Print_VDT_Database or Print_Curve_File, "
                             "so it cannot be used with Set_Depth (FloodSpreader_SpecifyDepth).")

        required = {}  # input-file key -> its value
        if self.uses_set_depth:
            required['Stream_File'] = self.stream_file
        else:
            required['COMID_Flow_File'] = self.comid_flow_file
            if fldpln:
                required['Print_VDT_Database'] = self.print_vdt_database
            else:
                required['Print_VDT_Database or Print_Curve_File'] = self.print_vdt_database or self.print_curve_file
        if fldpln:
            required.update({
                'Flow_Direction_File': self.flow_direction_file,
                'Stream_Info_File': self.stream_info_file,
                'FLDPLN_Library': self.fldpln_library,
            })
        if self.outvel:
            required['LU_Manning_n'] = self.lu_manning_n
        if self.outvel or self.flood_waterlc_and_strm_cells:
            required['LU_Raster_SameRes'] = self.lu_raster_sameres
        if self.flood_bathy_mask:
            required['BathyWaterMask'] = self.bathywatermask

        missing = [key for key, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")
        if self.flood_bathy_mask and not path_exists(self.bathywatermask):
            raise ValueError(f"Flood_Bathy_Mask floods the cells of BathyWaterMask, and {self.bathywatermask} "
                             f"doesn't exist.")

    @property
    def uses_set_depth(self) -> bool:
        """Whether every stream cell floods to set_depth, instead of the flows being mapped."""
        return self.set_depth > 0

    @property
    def makes_bathymetry(self) -> bool:
        """Whether the run burns ARC's bathymetry (bathy_out_file) into the DEM and writes it to fsoutbathy."""
        return bool(self.bathy_out_file and self.fsoutbathy)

    @property
    def needs_flood_map(self) -> bool:
        """Whether a flood map is needed, for an output raster."""
        return bool(self.outfld or self.outdep or self.outwse or self.outvel)

    @property
    def needs_dem(self) -> bool:
        """Whether the run needs the DEM: all but an FLDPLN flood map without OutWSE or bathymetry, which takes its
        grid from the library (or the flow directions) and its depths from the library."""
        return self.makes_bathymetry or not (self.mapper == FLDPLN and self.needs_flood_map and not self.outwse)

    @property
    def gpkg_file(self) -> str | None:
        """Where the flood's polygons are written: OutFLD's folder and name, as a GeoPackage; None without them."""
        if not (self.make_output_gpkg and self.outfld):
            return None
        flood = Path(self.outfld)
        return str(flood.with_name(f'{flood.stem}.gpkg'))

    def input_paths(self) -> dict[str, str]:
        """The files the run reads, by input-file key."""
        paths = {'DEM_File': self.dem_file}
        if self.needs_flood_map:
            if self.uses_set_depth:
                paths['Stream_File'] = self.stream_file
            else:
                paths['COMID_Flow_File'] = self.comid_flow_file
                paths['Print_VDT_Database'] = self.print_vdt_database
                # The VDT database is used instead of the curve file, but the multi-point mapper borrows from it
                if not self.print_vdt_database or self.mapper == MULTI_POINT:
                    paths['Print_Curve_File'] = self.print_curve_file

            if self.mapper == FLDPLN:
                paths['Flow_Direction_File'] = self.flow_direction_file
                paths['Stream_Info_File'] = self.stream_info_file
                paths['FLDPLN_Library'] = self.fldpln_library

            if self.outvel or self.flood_waterlc_and_strm_cells:
                paths['LU_Raster_SameRes'] = self.lu_raster_sameres
            if self.outvel:
                paths['LU_Manning_n'] = self.lu_manning_n
            if self.flood_bathy_mask:
                paths['BathyWaterMask'] = self.bathywatermask

        if self.makes_bathymetry:
            paths['BATHY_Out_File'] = self.bathy_out_file

        return {key: path for key, path in paths.items() if path}

    def output_paths(self) -> dict[str, str]:
        """The files the run writes, by input-file key."""
        paths = {}
        if self.needs_flood_map:
            paths = {'OutFLD': self.outfld, 'OutDEP': self.outdep, 'OutWSE': self.outwse, 'OutVEL': self.outvel,
                     'Make_Output_GPKG': self.gpkg_file}
        if self.makes_bathymetry:
            paths['FSOutBATHY'] = self.fsoutbathy
            
        return {key: path for key, path in paths.items() if path}

    def check_paths(self) -> None:
        """Check, before the run does anything, that every file it reads exists and that every file it writes can
        be: its folder exists and can be written to. Raises FileNotFoundError naming every one that fails."""
        problems = [f"{key} {path} doesn't exist" for key, path in self.input_paths().items() if not path_exists(path)]
        for key, path in self.output_paths().items():
            path = Path(path)
            folder = path.parent
            if path.is_dir():
                problems.append(f"{key} {path} is a folder")
            elif not folder.is_dir():
                problems.append(f"{key}'s folder {folder} doesn't exist")
            elif not os.access(folder, os.W_OK):
                problems.append(f"{key}'s folder {folder} can't be written to")
        if problems:
            raise FileNotFoundError("; ".join(problems) + ".")

    @property
    def multi_point_options(self) -> dict:
        """The multi-point mapper's tuning options (the MPI_ keys), named as multi_point_interpolation names them."""
        return {f.name.removeprefix('mpi_'): getattr(self, f.name) for f in fields(self) if f.name.startswith('mpi_')}


def _to_bool(name: str, value) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise ValueError(f'{name} must be True or False, not {value!r}.')
    return bool(value)


def _choice(name: str, value, choices: tuple):
    """The one of ``choices`` that ``value`` is: text matches whatever its case, and a number matches its text."""
    for choice in choices:
        if isinstance(choice, str) and str(value).strip().lower() == choice.lower():
            return choice
        if isinstance(choice, int) and str(value).strip() == str(choice):
            return choice
    close = difflib.get_close_matches(str(value), [str(choice) for choice in choices], n=1)
    raise ValueError(f"{name} must be one of {', '.join(map(str, choices))}, not {value!r}"
                     + (f" (did you mean {close[0]!r}?)." if close else "."))


def _coerce(name: str, value, annotation):
    """Convert a raw input value to the type given by its field annotation."""
    if get_origin(annotation) is Literal:
        return _choice(name, value, get_args(annotation))
    members = get_args(annotation) if get_origin(annotation) in (Union, UnionType) else (annotation,)
    if type(None) in members and (value is None or value == ''):
        return None
    if bool in members:
        return _to_bool(name, value)
    for kind, description in ((int, 'an integer'), (float, 'a number')):
        if kind in members:
            try:
                if kind is int and isinstance(value, float) and not value.is_integer():
                    raise ValueError
                return kind(value)
            except (TypeError, ValueError):
                raise ValueError(f'{name} must be {description}, not {value!r}.') from None
    if str in members and value is not None:
        return os.fspath(value) if isinstance(value, os.PathLike) else str(value)
    return value
