#!/usr/bin/env python3
"""
Create 1-time-step sample NetCDF files from ENSO input datasets and run basic QC.

What it does
------------
1) Scans a directory for .nc files.
2) Chooses one time index per file (default: middle).
3) Writes `sample_<original>.nc` with only that one time step preserved.
4) Runs structure / missing-value / time-axis QC and writes:
   - qc_file_summary.csv      : one row per file
   - qc_variable_summary.csv  : one row per file-variable
   - qc_report.md             : human-readable report

This script is deliberately dataset-agnostic, so it can handle:
- one variable per file, or
- multiple variables in a single NetCDF file,
provided there is a time dimension.

Examples
--------
python make_samples_and_qc_enso_nc.py /mnt/d/project/01_ENSO/01_data/03_input
python make_samples_and_qc_enso_nc.py /mnt/d/project/01_ENSO/01_data/03_input \
    --outdir /mnt/d/project/01_ENSO/01_data/03_input/_sample_qc \
    --time-select middle
python make_samples_and_qc_enso_nc.py /path/to/input --time-select first
python make_samples_and_qc_enso_nc.py /path/to/input --time-index 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

try:
    import cftime  # type: ignore
except Exception:  # pragma: no cover
    cftime = None

TIME_CANDIDATES = ["time", "Time", "TIME"]
LAT_CANDIDATES = [
    "lat", "latitude", "nav_lat", "LAT", "Latitude", "yt_ocean", "y", "Y",
]
LON_CANDIDATES = [
    "lon", "longitude", "nav_lon", "LON", "Longitude", "xt_ocean", "x", "X",
]
BOUNDS_CANDIDATES = ["time_bnds", "time_bounds", "bounds_time", "tbnds", "bnds_time"]
ATMOS_KEYWORDS = (
    "slp", "psl", "msl", "tauu", "tauv", "wind", "uas", "vas", "u10", "v10",
    "taux", "tauy", "stress", "air", "atm"
)


@dataclass
class TimeCheck:
    monotonic: bool
    duplicated: bool
    gap_check_method: str
    n_gap_steps: int
    first_gap_detail: str


@dataclass
class BoundsCheck:
    has_time_bounds: bool
    time_bounds_name: str
    n_bounds_discontinuity: int
    first_bounds_detail: str


@dataclass
class VariableQC:
    file: str
    variable: str
    category: str
    dims: str
    shape: str
    dtype: str
    nt: int
    total_cells_per_time: int
    n_time_any_nan: int
    n_time_all_nan: int
    min_nan_count: int
    max_nan_count: int
    ref_nan_count: int
    n_time_nan_changed: int
    fraction_changed_cells: float
    changed_cells: int
    max_flip_count: int
    mean_flip_on_changed_cells: float
    nan_count_range: int
    monotonic_time: bool
    duplicated_time: bool
    gap_check_method: str
    n_gap_steps: int
    first_gap_detail: str
    has_time_bounds: bool
    time_bounds_name: str
    n_bounds_discontinuity: int
    first_bounds_detail: str
    status: str
    note: str


@dataclass
class FileQC:
    file: str
    sample_file: str
    status: str
    chosen_time_index: int
    chosen_time_value: str
    time_name: str
    time_len: int
    lat_name: str
    lon_name: str
    lat_min: float | str
    lat_max: float | str
    lon_min: float | str
    lon_max: float | str
    has_expected_120x180_grid: bool | str
    data_vars: str
    n_time_dependent_vars: int
    n_pass: int
    n_warn: int
    n_fail: int
    n_error: int
    note: str


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


def infer_name(ds: xr.Dataset, candidates: Sequence[str], *, search_dims: bool = True) -> str:
    for name in candidates:
        if name in ds.coords:
            return name
    if search_dims:
        for name in candidates:
            if name in ds.dims:
                return name
    lower_map = {k.lower(): k for k in list(ds.coords) + list(ds.dims)}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return ""



def infer_time_name(ds: xr.Dataset) -> str:
    name = infer_name(ds, TIME_CANDIDATES)
    if name:
        return name
    # fallback: any coord or dim containing 'time'
    for name in list(ds.coords) + list(ds.dims):
        if "time" in name.lower():
            return name
    return ""



def infer_lat_lon(ds: xr.Dataset) -> Tuple[str, str]:
    lat = infer_name(ds, LAT_CANDIDATES)
    lon = infer_name(ds, LON_CANDIDATES)
    return lat, lon



def infer_time_bounds_name(ds: xr.Dataset, time_name: str) -> str:
    if time_name in ds.coords:
        bounds_attr = ds[time_name].attrs.get("bounds", "")
        if isinstance(bounds_attr, str) and bounds_attr in ds.variables:
            return bounds_attr
    for name in BOUNDS_CANDIDATES:
        if name in ds.variables:
            return name
    for name, da in ds.variables.items():
        if name == time_name:
            continue
        if time_name in da.dims and da.ndim == 2 and any(str(d).lower().startswith("bnd") for d in da.dims):
            return name
    return ""



def time_dependent_vars(ds: xr.Dataset, time_name: str) -> list[str]:
    return [name for name, da in ds.data_vars.items() if time_name in da.dims]



def infer_primary_var(ds: xr.Dataset, time_name: str) -> str:
    vars_with_time = time_dependent_vars(ds, time_name)
    if not vars_with_time:
        return ""
    return sorted(vars_with_time, key=lambda v: ds[v].size, reverse=True)[0]



def classify_variable(varname: str) -> str:
    s = varname.lower()
    return "atmos" if any(k in s for k in ATMOS_KEYWORDS) else "ocean"



def choose_time_index(n: int, mode: str, explicit_index: Optional[int]) -> int:
    if n <= 0:
        raise ValueError("time length must be positive")
    if explicit_index is not None:
        idx = explicit_index if explicit_index >= 0 else n + explicit_index
        if idx < 0 or idx >= n:
            raise IndexError(f"time index out of range: {explicit_index} for n={n}")
        return idx
    if mode == "first":
        return 0
    if mode == "last":
        return n - 1
    return n // 2



def stringify_scalar_time(v: object) -> str:
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    if hasattr(v, "isoformat"):
        try:
            return str(v.isoformat())
        except Exception:
            return str(v)
    if isinstance(v, np.datetime64):
        try:
            return np.datetime_as_string(v, unit="s")
        except Exception:
            return str(v)
    return str(v)



def month_key(v: object) -> Optional[Tuple[int, int]]:
    # cftime / datetime-like
    if hasattr(v, "year") and hasattr(v, "month"):
        try:
            return int(v.year), int(v.month)
        except Exception:
            pass
    # numpy datetime64
    if isinstance(v, np.datetime64):
        try:
            ts = pd.Timestamp(v)
            return int(ts.year), int(ts.month)
        except Exception:
            return None
    # pandas Timestamp
    if isinstance(v, pd.Timestamp):
        return int(v.year), int(v.month)
    return None



def next_month_key(key: Tuple[int, int]) -> Tuple[int, int]:
    y, m = key
    return (y + 1, 1) if m == 12 else (y, m + 1)



def check_time_axis(time_values: np.ndarray) -> TimeCheck:
    n = len(time_values)
    if n <= 1:
        return TimeCheck(True, False, "single_step", 0, "")

    # duplicate check via string representation for maximum robustness
    as_strings = [stringify_scalar_time(v) for v in time_values]
    duplicated = len(set(as_strings)) != len(as_strings)

    # monotonic check
    monotonic = True
    month_keys: list[Optional[Tuple[int, int]]] = [month_key(v) for v in time_values]
    if all(k is not None for k in month_keys):
        mk = [k for k in month_keys if k is not None]
        for a, b in zip(mk[:-1], mk[1:]):
            if not (a < b):
                monotonic = False
                break
        gaps = 0
        first_gap = ""
        if monotonic:
            for i, (a, b) in enumerate(zip(mk[:-1], mk[1:])):
                exp = next_month_key(a)
                if b != exp:
                    gaps += 1
                    if not first_gap:
                        first_gap = f"index {i}->{i+1}: expected {exp[0]:04d}-{exp[1]:02d}, got {b[0]:04d}-{b[1]:02d}"
        return TimeCheck(monotonic, duplicated, "monthly_calendar", gaps, first_gap)

    # numeric fallback
    try:
        vals = np.asarray(time_values, dtype=np.float64)
        diffs = np.diff(vals)
        monotonic = bool(np.all(diffs > 0))
        # for numeric time we can only detect obvious non-regularity
        if len(diffs) == 0:
            return TimeCheck(monotonic, duplicated, "numeric_single_diff", 0, "")
        ref = np.median(diffs)
        bad = np.where(~np.isclose(diffs, ref, rtol=1e-6, atol=1e-9))[0]
        if len(bad) > 0:
            i = int(bad[0])
            detail = f"index {i}->{i+1}: step={diffs[i]!r}, median_step={ref!r}"
            return TimeCheck(monotonic, duplicated, "numeric_regular_step", int(len(bad)), detail)
        return TimeCheck(monotonic, duplicated, "numeric_regular_step", 0, "")
    except Exception:
        return TimeCheck(monotonic, duplicated, "unknown", 0, "")



def check_time_bounds(ds: xr.Dataset, time_name: str) -> BoundsCheck:
    bname = infer_time_bounds_name(ds, time_name)
    if not bname:
        return BoundsCheck(False, "", 0, "")
    try:
        b = ds[bname]
        if time_name not in b.dims or b.ndim != 2 or b.shape[1] < 2:
            return BoundsCheck(True, bname, 0, "unsupported bounds shape")
        vals = np.asarray(b.values)
        # assume [:, 0] = start, [:, 1] = end
        n_bad = 0
        first = ""
        for i in range(vals.shape[0] - 1):
            prev_end = vals[i, 1]
            next_start = vals[i + 1, 0]
            if stringify_scalar_time(prev_end) != stringify_scalar_time(next_start):
                n_bad += 1
                if not first:
                    first = (
                        f"index {i}->{i+1}: prev_end={stringify_scalar_time(prev_end)}, "
                        f"next_start={stringify_scalar_time(next_start)}"
                    )
        return BoundsCheck(True, bname, n_bad, first)
    except Exception as e:
        return BoundsCheck(True, bname, 0, f"bounds check skipped: {type(e).__name__}: {e}")



def safe_float(v: object) -> float | str:
    try:
        x = float(np.asarray(v).min())
        return x
    except Exception:
        return ""



def minmax_coord(ds: xr.Dataset, name: str) -> Tuple[float | str, float | str]:
    if not name or name not in ds.variables:
        return "", ""
    try:
        arr = np.asarray(ds[name].values)
        return float(np.nanmin(arr)), float(np.nanmax(arr))
    except Exception:
        return "", ""



def has_expected_grid(ds: xr.Dataset, lat_name: str, lon_name: str) -> bool | str:
    try:
        h = ds.sizes.get(lat_name, None) if lat_name else None
        w = ds.sizes.get(lon_name, None) if lon_name else None
        if h is None or w is None:
            return ""
        return bool(h == 120 and w == 180)
    except Exception:
        return ""



def dims_to_string(da: xr.DataArray) -> str:
    return ",".join(f"{d}:{int(da.sizes[d])}" for d in da.dims)



def write_netcdf_safe(ds: xr.Dataset, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    clean = ds.copy(deep=False)
    for name in list(clean.variables):
        try:
            clean[name].encoding = {}
        except Exception:
            pass
    try:
        clean.to_netcdf(outpath)
    except Exception:
        # last fallback: strip attrs that sometimes break writing
        clean2 = clean.copy(deep=True)
        clean2.attrs = {}
        for name in list(clean2.variables):
            try:
                clean2[name].attrs = {}
                clean2[name].encoding = {}
            except Exception:
                pass
        clean2.to_netcdf(outpath)


# -----------------------------------------------------------------------------
# QC
# -----------------------------------------------------------------------------


def analyze_variable(
    ds: xr.Dataset,
    path: Path,
    varname: str,
    time_name: str,
    time_check: TimeCheck,
    bounds_check: BoundsCheck,
) -> VariableQC:
    da = ds[varname]
    if time_name not in da.dims:
        return VariableQC(
            file=str(path),
            variable=varname,
            category=classify_variable(varname),
            dims=dims_to_string(da),
            shape=str(tuple(int(da.sizes[d]) for d in da.dims)),
            dtype=str(da.dtype),
            nt=0,
            total_cells_per_time=0,
            n_time_any_nan=0,
            n_time_all_nan=0,
            min_nan_count=0,
            max_nan_count=0,
            ref_nan_count=0,
            n_time_nan_changed=0,
            fraction_changed_cells=np.nan,
            changed_cells=0,
            max_flip_count=0,
            mean_flip_on_changed_cells=0.0,
            nan_count_range=0,
            monotonic_time=time_check.monotonic,
            duplicated_time=time_check.duplicated,
            gap_check_method=time_check.gap_check_method,
            n_gap_steps=time_check.n_gap_steps,
            first_gap_detail=time_check.first_gap_detail,
            has_time_bounds=bounds_check.has_time_bounds,
            time_bounds_name=bounds_check.time_bounds_name,
            n_bounds_discontinuity=bounds_check.n_bounds_discontinuity,
            first_bounds_detail=bounds_check.first_bounds_detail,
            status="SKIP",
            note="variable has no time dimension",
        )

    other_dims = [d for d in da.dims if d != time_name]
    nt = int(da.sizes[time_name])
    total_cells = int(math.prod(int(da.sizes[d]) for d in other_dims)) if other_dims else 1

    count_per_time = da.count(dim=other_dims) if other_dims else da.notnull().astype(np.int64)
    valid_counts = np.asarray(count_per_time.values).astype(np.int64).reshape(-1)
    nan_counts = total_cells - valid_counts

    n_time_any_nan = int(np.sum(nan_counts > 0))
    n_time_all_nan = int(np.sum(valid_counts == 0))
    ref_nan_count = int(nan_counts[0]) if len(nan_counts) else 0
    n_time_nan_changed = int(np.sum(nan_counts != ref_nan_count))
    min_nan_count = int(np.min(nan_counts)) if len(nan_counts) else 0
    max_nan_count = int(np.max(nan_counts)) if len(nan_counts) else 0
    nan_count_range = max_nan_count - min_nan_count

    changed_cells = 0
    fraction_changed_cells = np.nan
    max_flip_count = 0
    mean_flip_on_changed_cells = 0.0
    if nt > 1 and other_dims:
        valid_mask = da.notnull().astype(np.int8)
        mask_change = valid_mask.std(dim=time_name) > 0
        flip_count = np.abs(valid_mask.diff(time_name)).sum(dim=time_name)
        changed_cells = int(mask_change.sum().item())
        fraction_changed_cells = changed_cells / mask_change.size if mask_change.size else np.nan
        max_flip_count = int(flip_count.max().item()) if flip_count.size else 0
        if changed_cells > 0:
            changed_flip = flip_count.where(flip_count > 0)
            mean_flip_on_changed_cells = float(changed_flip.mean().item())

    category = classify_variable(varname)
    status = "PASS"
    notes: list[str] = []

    if n_time_all_nan > 0:
        status = "FAIL"
        notes.append("all-NaN time step exists")
    if category == "atmos" and n_time_any_nan > 0:
        status = "FAIL"
        notes.append("atmos variable has NaN")
    if not time_check.monotonic or time_check.duplicated:
        status = "FAIL"
        notes.append("invalid time coordinate")
    if time_check.n_gap_steps > 0:
        status = "FAIL"
        notes.append("time gap/overlap detected")
    if bounds_check.n_bounds_discontinuity > 0:
        status = "FAIL"
        notes.append("time bounds continuity broken")

    if status == "PASS":
        if category == "ocean" and n_time_nan_changed > 0:
            status = "WARN"
            notes.append("ocean mask or valid-cell count changes over time")
        elif category == "ocean" and np.isfinite(fraction_changed_cells) and (fraction_changed_cells >= 0.01 or nan_count_range > 20):
            status = "WARN"
            notes.append("mask variation larger than heuristic threshold")
        elif category == "atmos":
            notes.append("no NaN in atmos variable")
        else:
            notes.append("no obvious issue")

    return VariableQC(
        file=str(path),
        variable=varname,
        category=category,
        dims=dims_to_string(da),
        shape=str(tuple(int(da.sizes[d]) for d in da.dims)),
        dtype=str(da.dtype),
        nt=nt,
        total_cells_per_time=total_cells,
        n_time_any_nan=n_time_any_nan,
        n_time_all_nan=n_time_all_nan,
        min_nan_count=min_nan_count,
        max_nan_count=max_nan_count,
        ref_nan_count=ref_nan_count,
        n_time_nan_changed=n_time_nan_changed,
        fraction_changed_cells=float(fraction_changed_cells) if np.isfinite(fraction_changed_cells) else np.nan,
        changed_cells=changed_cells,
        max_flip_count=max_flip_count,
        mean_flip_on_changed_cells=mean_flip_on_changed_cells,
        nan_count_range=nan_count_range,
        monotonic_time=time_check.monotonic,
        duplicated_time=time_check.duplicated,
        gap_check_method=time_check.gap_check_method,
        n_gap_steps=time_check.n_gap_steps,
        first_gap_detail=time_check.first_gap_detail,
        has_time_bounds=bounds_check.has_time_bounds,
        time_bounds_name=bounds_check.time_bounds_name,
        n_bounds_discontinuity=bounds_check.n_bounds_discontinuity,
        first_bounds_detail=bounds_check.first_bounds_detail,
        status=status,
        note="; ".join(notes),
    )



def worst_status(statuses: Iterable[str]) -> str:
    order = {"ERROR": 4, "FAIL": 3, "WARN": 2, "PASS": 1, "SKIP": 0}
    best = "PASS"
    score = -1
    for s in statuses:
        if order.get(s, -1) > score:
            best = s
            score = order.get(s, -1)
    return best



def analyze_file(
    path: Path,
    outdir: Path,
    time_select: str,
    time_index: Optional[int],
) -> tuple[FileQC, list[VariableQC]]:
    sample_path = outdir / "samples" / f"sample_{path.name}"

    try:
        ds = xr.open_dataset(path, decode_times=True)
    except Exception:
        ds = xr.open_dataset(path, decode_times=False)

    try:
        time_name = infer_time_name(ds)
        if not time_name:
            file_qc = FileQC(
                file=str(path),
                sample_file=str(sample_path),
                status="ERROR",
                chosen_time_index=-1,
                chosen_time_value="",
                time_name="",
                time_len=0,
                lat_name="",
                lon_name="",
                lat_min="",
                lat_max="",
                lon_min="",
                lon_max="",
                has_expected_120x180_grid="",
                data_vars=",".join(ds.data_vars),
                n_time_dependent_vars=0,
                n_pass=0,
                n_warn=0,
                n_fail=0,
                n_error=1,
                note="time coordinate/dimension not found",
            )
            return file_qc, []

        ntime = int(ds.sizes[time_name])
        idx = choose_time_index(ntime, time_select, time_index)
        chosen_time_value = stringify_scalar_time(np.asarray(ds[time_name].values)[idx])

        # sample write
        ds_sample = ds.isel({time_name: slice(idx, idx + 1)})
        write_netcdf_safe(ds_sample, sample_path)

        time_values = np.asarray(ds[time_name].values)
        time_check = check_time_axis(time_values)
        bounds_check = check_time_bounds(ds, time_name)

        lat_name, lon_name = infer_lat_lon(ds)
        lat_min, lat_max = minmax_coord(ds, lat_name)
        lon_min, lon_max = minmax_coord(ds, lon_name)
        grid_ok = has_expected_grid(ds, lat_name, lon_name)

        vars_with_time = time_dependent_vars(ds, time_name)
        var_rows: list[VariableQC] = [
            analyze_variable(ds, path, var, time_name, time_check, bounds_check)
            for var in vars_with_time
        ]
        status = worst_status(v.status for v in var_rows) if var_rows else "WARN"

        n_pass = sum(v.status == "PASS" for v in var_rows)
        n_warn = sum(v.status == "WARN" for v in var_rows)
        n_fail = sum(v.status == "FAIL" for v in var_rows)
        n_error = sum(v.status == "ERROR" for v in var_rows)

        note_parts: list[str] = []
        if isinstance(grid_ok, bool) and not grid_ok:
            note_parts.append("grid is not 120x180")
        if time_check.gap_check_method == "unknown":
            note_parts.append("time gap check unavailable")
        if not var_rows:
            note_parts.append("no time-dependent data variable found")
        if not note_parts:
            note_parts.append("sample created and QC completed")

        file_qc = FileQC(
            file=str(path),
            sample_file=str(sample_path),
            status=status,
            chosen_time_index=idx,
            chosen_time_value=chosen_time_value,
            time_name=time_name,
            time_len=ntime,
            lat_name=lat_name,
            lon_name=lon_name,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            has_expected_120x180_grid=grid_ok,
            data_vars=",".join(ds.data_vars),
            n_time_dependent_vars=len(vars_with_time),
            n_pass=n_pass,
            n_warn=n_warn,
            n_fail=n_fail,
            n_error=n_error,
            note="; ".join(note_parts),
        )
        return file_qc, var_rows
    finally:
        ds.close()


# -----------------------------------------------------------------------------
# reporting
# -----------------------------------------------------------------------------


def write_markdown_report(file_df: pd.DataFrame, var_df: pd.DataFrame, outpath: Path) -> None:
    lines: list[str] = []
    lines.append("# ENSO NetCDF sample/QC report")
    lines.append("")
    if file_df.empty:
        lines.append("No files were processed.")
        outpath.write_text("\n".join(lines), encoding="utf-8")
        return

    counts = file_df["status"].value_counts(dropna=False).to_dict()
    lines.append("## File-level status summary")
    lines.append("")
    for key in ["PASS", "WARN", "FAIL", "ERROR"]:
        lines.append(f"- {key}: {counts.get(key, 0)}")
    lines.append("")

    flagged = file_df[file_df["status"].isin(["WARN", "FAIL", "ERROR"])]
    if flagged.empty:
        lines.append("All files are PASS.")
        lines.append("")
    else:
        lines.append("## Flagged files")
        lines.append("")
        cols = [
            "status", "file", "sample_file", "chosen_time_index", "chosen_time_value",
            "n_warn", "n_fail", "note",
        ]
        lines.append(flagged[cols].to_markdown(index=False))
        lines.append("")

    flagged_vars = var_df[var_df["status"].isin(["WARN", "FAIL", "ERROR"])] if not var_df.empty else pd.DataFrame()
    if not flagged_vars.empty:
        lines.append("## Flagged variables")
        lines.append("")
        cols = [
            "status", "file", "variable", "category", "n_time_any_nan", "n_time_all_nan",
            "n_time_nan_changed", "n_gap_steps", "n_bounds_discontinuity",
            "fraction_changed_cells", "nan_count_range", "note",
        ]
        lines.append(flagged_vars[cols].to_markdown(index=False))
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Atmos variables are treated strictly: any NaN -> FAIL.")
    lines.append("- Ocean variables with time-varying valid/missing masks -> WARN.")
    lines.append("- Heuristic mask warning threshold: changed-cell fraction >= 0.01 or NaN-count range > 20.")
    lines.append("- Sample files preserve the time dimension with length 1.")
    lines.append("")

    outpath.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Create one-time-step sample NetCDF files and run QC.")
    p.add_argument("input_dir", type=Path, help="Directory containing .nc files")
    p.add_argument("--pattern", default="*.nc", help="Glob pattern under input_dir (default: *.nc)")
    p.add_argument("--outdir", type=Path, default=None, help="Output directory (default: <input_dir>/_sample_qc)")
    p.add_argument("--time-select", choices=["first", "middle", "last"], default="middle",
                   help="Which time index to select when --time-index is not given")
    p.add_argument("--time-index", type=int, default=None,
                   help="Explicit time index to select. Negative values allowed.")
    args = p.parse_args()

    input_dir = args.input_dir.resolve()
    outdir = (args.outdir.resolve() if args.outdir is not None else input_dir / "_sample_qc")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "samples").mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files found under {input_dir} with pattern {args.pattern}")

    file_rows: list[dict] = []
    var_rows: list[dict] = []

    for i, path in enumerate(files, start=1):
        if not path.is_file():
            continue
        try:
            file_qc, vars_qc = analyze_file(path, outdir, args.time_select, args.time_index)
        except Exception as e:
            sample_path = outdir / "samples" / f"sample_{path.name}"
            file_qc = FileQC(
                file=str(path),
                sample_file=str(sample_path),
                status="ERROR",
                chosen_time_index=-1,
                chosen_time_value="",
                time_name="",
                time_len=0,
                lat_name="",
                lon_name="",
                lat_min="",
                lat_max="",
                lon_min="",
                lon_max="",
                has_expected_120x180_grid="",
                data_vars="",
                n_time_dependent_vars=0,
                n_pass=0,
                n_warn=0,
                n_fail=0,
                n_error=1,
                note=f"{type(e).__name__}: {e}",
            )
            vars_qc = []

        file_rows.append(asdict(file_qc))
        var_rows.extend(asdict(v) for v in vars_qc)
        print(f"[{i}/{len(files)}] {Path(file_qc.file).name}: {file_qc.status} -> {file_qc.sample_file}")

    file_df = pd.DataFrame(file_rows)
    var_df = pd.DataFrame(var_rows)

    file_csv = outdir / "qc_file_summary.csv"
    var_csv = outdir / "qc_variable_summary.csv"
    report_md = outdir / "qc_report.md"

    file_df.to_csv(file_csv, index=False)
    var_df.to_csv(var_csv, index=False)
    write_markdown_report(file_df, var_df, report_md)

    print("\n=== File summary ===")
    if not file_df.empty:
        print(file_df[["status", "file", "sample_file", "chosen_time_index", "chosen_time_value", "note"]].to_string(index=False))
    print(f"\nSaved: {file_csv}")
    print(f"Saved: {var_csv}")
    print(f"Saved: {report_md}")


if __name__ == "__main__":
    main()
