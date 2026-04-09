from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from preprocess_config import (
    ATM_VARS,
    INPUT_ROOT,
    INTERIM_ROOT,
    LAT_CANDIDATES,
    LON_CANDIDATES,
    OCEAN_VARS,
    TEST_END,
    TEST_START,
    TIME_CANDIDATES,
    VALIDATION_END,
    VALIDATION_START,
    VAR_ORDER,
    is_anomaly_file,
    select_preferred_path,
)


def first_existing(items: Iterable[str], collection: Iterable[str]) -> str | None:
    pool = set(collection)
    for item in items:
        if item in pool:
            return item
    return None


def guess_time_name(ds: xr.Dataset) -> str | None:
    name = first_existing(TIME_CANDIDATES, ds.dims) or first_existing(TIME_CANDIDATES, ds.coords)
    if name:
        return name

    for name in ds.coords:
        c = ds.coords[name]
        std = str(c.attrs.get("standard_name", "")).lower()
        axis = str(c.attrs.get("axis", "")).upper()
        units = str(c.attrs.get("units", "")).lower()
        if std == "time" or axis == "T" or "since" in units:
            return name

    return None


def normalize_monthly_time(ds: xr.Dataset) -> xr.Dataset:
    vals = ds["time"].values
    out = []
    for t in vals:
        if hasattr(t, "year") and hasattr(t, "month"):
            y = int(t.year)
            m = int(t.month)
        else:
            ts = pd.Timestamp(t)
            y = int(ts.year)
            m = int(ts.month)
        out.append(np.datetime64(f"{y:04d}-{m:02d}-01"))
    ds = ds.assign_coords(time=("time", np.array(out, dtype="datetime64[ns]")))
    ds = ds.sortby("time")
    return ds


def _time_to_yyyymm(t) -> str:
    if hasattr(t, "year") and hasattr(t, "month"):
        return f"{int(t.year):04d}{int(t.month):02d}"
    return pd.Timestamp(t).strftime("%Y%m")


def infer_var_from_name(path: Path) -> str | None:
    name = path.name
    m = re.match(r"^([A-Za-z0-9]+)\.", name)
    if m and m.group(1) in VAR_ORDER:
        return m.group(1)
    m = re.match(r"^([A-Za-z0-9]+)_", name)
    if m and m.group(1) in VAR_ORDER:
        return m.group(1)
    return None


def detect_main_var(ds: xr.Dataset, expected: str | None = None) -> str:
    if expected and expected in ds.data_vars:
        return expected
    data_vars = list(ds.data_vars)
    if expected:
        for v in data_vars:
            if v.lower() == expected.lower():
                return v
    filtered = [
        v for v in data_vars if v.lower() not in {"time_bnds", "lat_bnds", "lon_bnds", "bounds", "bnds"}
    ]
    if expected:
        for v in filtered:
            if expected in v.lower():
                return v
    if len(filtered) == 1:
        return filtered[0]
    raise ValueError(f"Cannot determine main variable from {data_vars}")


def rename_dims_coords(ds: xr.Dataset) -> xr.Dataset:
    rename_map: dict[str, str] = {}
    time_name = guess_time_name(ds)
    lat_name = first_existing(LAT_CANDIDATES, ds.dims) or first_existing(LAT_CANDIDATES, ds.coords)
    lon_name = first_existing(LON_CANDIDATES, ds.dims) or first_existing(LON_CANDIDATES, ds.coords)
    if time_name and time_name != "time":
        rename_map[time_name] = "time"
    if lat_name and lat_name != "lat":
        rename_map[lat_name] = "lat"
    if lon_name and lon_name != "lon":
        rename_map[lon_name] = "lon"
    if rename_map:
        ds = ds.rename(rename_map)
    return ds


def sort_lon_if_needed(ds: xr.Dataset) -> xr.Dataset:
    if "lon" not in ds.coords:
        return ds
    lon = ds["lon"]
    if lon.ndim != 1:
        return ds
    if np.any(np.diff(lon.values) < 0):
        return ds.sortby("lon")
    return ds


def load_one(path: Path, expected_var: str, fillna_mode: str = "none") -> xr.Dataset:
    with xr.open_dataset(path, decode_times=True) as ds_in:
        ds = rename_dims_coords(ds_in)
        ds = sort_lon_if_needed(ds)
        if "time" in ds.coords or "time" in ds.dims:
            ds = normalize_monthly_time(ds)
        main = detect_main_var(ds, expected_var)
        da = ds[main]
        if main != expected_var:
            da = da.rename(expected_var)
        missing_dims = [d for d in ["time", "lat", "lon"] if d not in da.dims]
        if missing_dims:
            raise ValueError(f"{path} missing dims {missing_dims}; got {da.dims}")
        da = da.transpose("time", "lat", "lon").load()
    out = da.to_dataset(name=expected_var)
    if fillna_mode == "paper":
        if expected_var in OCEAN_VARS:
            out[expected_var] = out[expected_var].fillna(0.0)
    elif fillna_mode == "all":
        out[expected_var] = out[expected_var].fillna(0.0)
    return out


def assert_same_grid(ds: xr.Dataset, ref: xr.Dataset | None, label: str) -> None:
    if ref is None:
        return
    for coord in ["lat", "lon"]:
        a = ds[coord].values
        b = ref[coord].values
        if a.shape != b.shape or not np.allclose(a, b, equal_nan=True):
            raise ValueError(f"Grid mismatch in {label} for coord={coord}: {a.shape} vs {b.shape}")


def assert_monthly_time(ds: xr.Dataset, label: str) -> None:
    vals = ds["time"].values
    if vals.size == 0:
        raise ValueError(f"No timesteps found in {label}")
    if vals.size > 1 and np.any(np.diff(vals).astype("timedelta64[D]") <= np.timedelta64(0, "D")):
        raise ValueError(f"Time coordinate is not strictly increasing in {label}")


def assert_requested_time_covered(ds: xr.Dataset, time_start: str | None, time_end: str | None, label: str) -> None:
    first = pd.Timestamp(ds["time"].values[0])
    last = pd.Timestamp(ds["time"].values[-1])
    if time_start is not None and first > pd.Timestamp(time_start):
        raise ValueError(f"{label} starts at {first:%Y-%m}, later than requested {time_start}")
    if time_end is not None and last < pd.Timestamp(time_end):
        raise ValueError(f"{label} ends at {last:%Y-%m}, earlier than requested {time_end}")


def get_time_label(ds: xr.Dataset) -> str:
    vals = ds["time"].values
    return f"{_time_to_yyyymm(vals[0])}-{_time_to_yyyymm(vals[-1])}"


def apply_time_slice(ds: xr.Dataset, time_start: str | None, time_end: str | None) -> xr.Dataset:
    if time_start is None and time_end is None:
        return ds
    assert_requested_time_covered(ds, time_start, time_end, label="merged dataset")
    sliced = ds.sel(time=slice(time_start, time_end))
    if sliced.sizes.get("time", 0) == 0:
        raise ValueError(f"No timesteps after time slice: start={time_start}, end={time_end}")
    return sliced


def missing_var_hint(missing: list[str]) -> str:
    if "uasmax" in missing:
        return " Missing 'uasmax' usually means make_anom_uasmax.py has not been run yet."
    return ""


def merge_files(
    file_map: dict[str, Path],
    out_path: Path,
    *,
    var_order: list[str],
    fillna_mode: str = "none",
    time_start: str | None = None,
    time_end: str | None = None,
) -> None:
    missing = [v for v in var_order if v not in file_map]
    if missing:
        available = ", ".join(sorted(file_map))
        raise FileNotFoundError(
            f"Missing variables: {missing}. Available: [{available}]." + missing_var_hint(missing)
        )
    datasets = []
    ref = None
    for var in var_order:
        ds = load_one(file_map[var], var, fillna_mode=fillna_mode)
        assert_same_grid(ds, ref, str(file_map[var]))
        assert_monthly_time(ds, str(file_map[var]))
        if ref is None:
            ref = ds
        datasets.append(ds)
    merged = xr.merge(
        datasets,
        join="inner",
        compat="override",
        combine_attrs="drop_conflicts",
    ).transpose("time", "lat", "lon")
    if merged.sizes["time"] == 0:
        raise ValueError("No overlapping timesteps after inner merge")
    merged = apply_time_slice(merged, time_start=time_start, time_end=time_end)
    assert_monthly_time(merged, str(out_path))
    merged.attrs["merged_variables"] = ",".join(var_order)
    merged.attrs["time_range"] = get_time_label(merged)
    merged.attrs["note"] = (
        "Merged monthly anomaly fields for ENSO model input. "
        f"Variable order target: {','.join(var_order)}"
    )
    encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in var_order}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_netcdf(out_path, format="NETCDF4", encoding=encoding)
    print(f"[OK] {out_path}  ({merged.sizes['time']} steps; {merged.attrs['time_range']})")


def find_cmip6_groups(
    inroot: Path,
    *,
    only_experiments: set[str] | None = None,
) -> list[tuple[str, str, str, Path]]:
    groups: list[tuple[str, str, str, Path]] = []
    for exp_dir in sorted([p for p in inroot.iterdir() if p.is_dir()]):
        if only_experiments is not None and exp_dir.name not in only_experiments:
            continue
        for model_dir in sorted([p for p in exp_dir.iterdir() if p.is_dir()]):
            for member_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
                monthly = member_dir / "monthly"
                if monthly.is_dir():
                    groups.append((exp_dir.name, model_dir.name, member_dir.name, monthly))
    return groups


def collect_candidate_files(directory: Path, allowed_vars: set[str], *, allow_raw_fallback: bool) -> dict[str, Path]:
    buckets: dict[str, list[Path]] = {}
    seen: set[Path] = set()
    patterns = ["*.anom_1x2.nc"]
    if allow_raw_fallback:
        patterns.append("*.nc")
    for pattern in patterns:
        for path in sorted(directory.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            var = infer_var_from_name(path)
            if var in allowed_vars:
                buckets.setdefault(var, []).append(path)
    selected: dict[str, Path] = {}
    for var, paths in buckets.items():
        chosen = select_preferred_path(paths)
        if chosen is None:
            continue
        if allow_raw_fallback and not is_anomaly_file(chosen):
            print(f"[warn] using non-anomaly fallback for {var}: {chosen}")
        selected[var] = chosen
    return selected


def describe_file_map(label: str, file_map: dict[str, Path], var_order: list[str]) -> None:
    chosen = ", ".join(f"{var}={file_map[var].name}" for var in var_order if var in file_map)
    print(f"[map] {label}: {chosen}")


def run_cmip6(
    cmip6_root: Path,
    out_root: Path,
    *,
    var_order: list[str],
    fillna_mode: str,
    allow_raw_fallback: bool,
    only_experiments: set[str] | None = None,
) -> None:
    groups = find_cmip6_groups(cmip6_root, only_experiments=only_experiments)
    if not groups:
        exp_msg = ""
        if only_experiments:
            exp_msg = f" (filtered experiments: {sorted(only_experiments)})"
        raise FileNotFoundError(f"No CMIP6 monthly directories found under {cmip6_root}{exp_msg}")
    out_root.mkdir(parents=True, exist_ok=True)
    allowed = set(var_order)
    for exp, model, member, monthly_dir in groups:
        fmap = collect_candidate_files(monthly_dir, allowed, allow_raw_fallback=allow_raw_fallback)
        if not fmap:
            print(f"[SKIP] no compatible input files: {monthly_dir}")
            continue
        try:
            describe_file_map(str(monthly_dir), fmap, var_order)
            out_name = f"input_{model}_{exp}_{member}.nc"
            out_path = out_root / out_name
            merge_files(fmap, out_path, var_order=var_order, fillna_mode=fillna_mode)
        except Exception as e:
            print(f"[FAIL] {monthly_dir}: {type(e).__name__}: {e}")


def build_reanalysis_file_map(
    ocean_dir: Path,
    atm_dir: Path,
    *,
    ocean_vars: set[str],
    atm_vars: set[str],
    allow_raw_fallback: bool,
) -> dict[str, Path]:
    fmap: dict[str, Path] = {}
    fmap.update(collect_candidate_files(ocean_dir, ocean_vars, allow_raw_fallback=allow_raw_fallback))
    fmap.update(collect_candidate_files(atm_dir, atm_vars, allow_raw_fallback=allow_raw_fallback))
    return fmap


def run_reanalysis(
    base_root: Path,
    out_root: Path,
    *,
    var_order: list[str],
    fillna_mode: str,
    validation_start: str,
    validation_end: str,
    test_start: str,
    test_end: str,
    allow_raw_fallback: bool,
    validation_ocean_subdir: str = "soda",
    validation_atm_subdir: str = "20crv2",
    validation_out: str = "validation_dataset.nc",
    test_ocean_subdir: str = "godas",
    test_atm_subdir: str = "era5",
    test_out: str = "test_dataset.nc",
) -> None:
    validation_ocean_dir = base_root / validation_ocean_subdir
    validation_atm_dir = base_root / validation_atm_subdir
    test_ocean_dir = base_root / test_ocean_subdir
    test_atm_dir = base_root / test_atm_subdir

    out_root.mkdir(parents=True, exist_ok=True)
    ocean_vars = {v for v in var_order if v in OCEAN_VARS}
    atm_vars = {v for v in var_order if v in ATM_VARS}

    if validation_ocean_dir.is_dir() and validation_atm_dir.is_dir():
        try:
            fmap = build_reanalysis_file_map(
                validation_ocean_dir,
                validation_atm_dir,
                ocean_vars=ocean_vars,
                atm_vars=atm_vars,
                allow_raw_fallback=allow_raw_fallback,
            )
            describe_file_map(validation_out, fmap, var_order)
            out_path = out_root / validation_out
            merge_files(
                fmap,
                out_path,
                var_order=var_order,
                fillna_mode=fillna_mode,
                time_start=validation_start,
                time_end=validation_end,
            )
        except Exception as e:
            print(
                f"[FAIL] {validation_ocean_subdir}+{validation_atm_subdir}: "
                f"{type(e).__name__}: {e}"
            )
    else:
        print(
            f"[SKIP] validation directories not found: "
            f"{validation_ocean_dir} | {validation_atm_dir}"
        )

    if test_ocean_dir.is_dir() and test_atm_dir.is_dir():
        try:
            fmap = build_reanalysis_file_map(
                test_ocean_dir,
                test_atm_dir,
                ocean_vars=ocean_vars,
                atm_vars=atm_vars,
                allow_raw_fallback=allow_raw_fallback,
            )
            describe_file_map(test_out, fmap, var_order)
            out_path = out_root / test_out
            merge_files(
                fmap,
                out_path,
                var_order=var_order,
                fillna_mode=fillna_mode,
                time_start=test_start,
                time_end=test_end,
            )
        except Exception as e:
            print(f"[FAIL] {test_ocean_subdir}+{test_atm_subdir}: {type(e).__name__}: {e}")
    else:
        print(
            f"[SKIP] test directories not found: "
            f"{test_ocean_dir} | {test_atm_dir}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create ENSO multivariate input files from 02_interim anomaly fields")
    p.add_argument(
        "--inroot",
        type=Path,
        default=INTERIM_ROOT,
        help="Root directory containing cmip6/, soda/, 20crv2/, godas/, era5/, oras5/ ...",
    )
    p.add_argument("--outroot", type=Path, default=INPUT_ROOT, help="Output directory for input datasets")
    p.add_argument("--mode", choices=["all", "cmip6", "reanalysis"], default="all", help="What to build")
    p.add_argument(
        "--vars",
        nargs="+",
        choices=VAR_ORDER,
        default=VAR_ORDER,
        help="Variable order to include. Default is the canonical 10-variable configuration.",
    )
    p.add_argument(
        "--fillna",
        choices=["none", "paper", "all"],
        default="none",
        help="none: keep NaN; paper: fill only ocean vars with 0; all: fill all variables with 0",
    )
    p.add_argument(
        "--allow-raw-fallback",
        action="store_true",
        help="Allow fallback to non-anomaly *.nc files when anomaly files are missing.",
    )
    p.add_argument(
        "--only-experiments",
        nargs="+",
        default=None,
        help="Only build CMIP6 inputs for these experiment directories under inroot/cmip6 (e.g. historical_ssp370)",
    )
    p.add_argument("--validation-start", default=VALIDATION_START, help="Start month for validation dataset")
    p.add_argument("--validation-end", default=VALIDATION_END, help="End month for validation dataset")
    p.add_argument("--test-start", default=TEST_START, help="Start month for test dataset")
    p.add_argument("--test-end", default=TEST_END, help="End month for test dataset")
    p.add_argument(
        "--validation-ocean-subdir",
        default="soda",
        help="Subdirectory under --inroot for validation ocean variables (e.g. soda, oras5)",
    )
    p.add_argument(
        "--validation-atm-subdir",
        default="20crv2",
        help="Subdirectory under --inroot for validation atmospheric variables (e.g. 20crv2, era5)",
    )
    p.add_argument(
        "--validation-out",
        default="validation_dataset.nc",
        help="Output filename for validation dataset",
    )
    p.add_argument(
        "--test-ocean-subdir",
        default="godas",
        help="Subdirectory under --inroot for test ocean variables",
    )
    p.add_argument(
        "--test-atm-subdir",
        default="era5",
        help="Subdirectory under --inroot for test atmospheric variables",
    )
    p.add_argument(
        "--test-out",
        default="test_dataset.nc",
        help="Output filename for test dataset",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    var_order = list(dict.fromkeys(args.vars))
    only_experiments = set(args.only_experiments) if args.only_experiments else None

    if args.mode in {"all", "cmip6"}:
        run_cmip6(
            args.inroot / "cmip6",
            args.outroot,
            var_order=var_order,
            fillna_mode=args.fillna,
            allow_raw_fallback=args.allow_raw_fallback,
            only_experiments=only_experiments,
        )

    if args.mode in {"all", "reanalysis"}:
        run_reanalysis(
            args.inroot,
            args.outroot,
            var_order=var_order,
            fillna_mode=args.fillna,
            validation_start=args.validation_start,
            validation_end=args.validation_end,
            test_start=args.test_start,
            test_end=args.test_end,
            allow_raw_fallback=args.allow_raw_fallback,
            validation_ocean_subdir=args.validation_ocean_subdir,
            validation_atm_subdir=args.validation_atm_subdir,
            validation_out=args.validation_out,
            test_ocean_subdir=args.test_ocean_subdir,
            test_atm_subdir=args.test_atm_subdir,
            test_out=args.test_out,
        )


if __name__ == "__main__":
    main()
