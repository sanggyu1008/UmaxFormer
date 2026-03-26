from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

# Final multivariate input variable order
VAR_ORDER = ["tos", "ohc300", "mlotst", "sos", "psl", "uos", "vos", "uas", "vas"]
OCEAN_VARS = {"tos", "ohc300", "mlotst", "sos", "uos", "vos"}
ATM_VARS = {"psl", "uas", "vas"}

LAT_CANDIDATES = ["lat", "latitude", "nav_lat", "y"]
LON_CANDIDATES = ["lon", "longitude", "nav_lon", "x"]
TIME_CANDIDATES = ["time", "time_counter", "t", "valid_time"]


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
    ts = pd.Timestamp(t)
    return ts.strftime("%Y%m")


def infer_var_from_name(path: Path) -> str | None:
    name = path.name

    # era5/oras5/godas style: var.xxx.nc, var.mon.xxx.nc
    m = re.match(r"^([A-Za-z0-9]+)\.", name)
    if m and m.group(1) in VAR_ORDER:
        return m.group(1)

    # cmip6 style: var_Table_Model_...
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

    if len(data_vars) == 1:
        return data_vars[0]

    filtered = [
        v
        for v in data_vars
        if v.lower() not in {"time_bnds", "lat_bnds", "lon_bnds", "bounds", "bnds"}
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

    vals = lon.values
    if np.any(np.diff(vals) < 0):
        return ds.sortby("lon")

    return ds


def load_one(path: Path, expected_var: str, fillna_mode: str = "none") -> xr.Dataset:
    ds = xr.open_dataset(path, decode_times=True)
    ds = rename_dims_coords(ds)
    ds = sort_lon_if_needed(ds)

    if "time" in ds.coords or "time" in ds.dims:
        ds = normalize_monthly_time(ds)

    main = detect_main_var(ds, expected_var)
    if main != expected_var:
        ds = ds.rename({main: expected_var})

    ds = ds[[expected_var]]

    missing_dims = [d for d in ["time", "lat", "lon"] if d not in ds[expected_var].dims]
    if missing_dims:
        raise ValueError(f"{path} missing dims {missing_dims}; got {ds[expected_var].dims}")

    if fillna_mode == "paper":
        if expected_var in OCEAN_VARS:
            ds[expected_var] = ds[expected_var].fillna(0.0)
    elif fillna_mode == "all":
        ds[expected_var] = ds[expected_var].fillna(0.0)

    return ds


def assert_same_grid(ds: xr.Dataset, ref: xr.Dataset | None, label: str) -> None:
    if ref is None:
        return

    for coord in ["lat", "lon"]:
        a = ds[coord].values
        b = ref[coord].values
        if a.shape != b.shape or not np.allclose(a, b, equal_nan=True):
            raise ValueError(f"Grid mismatch in {label} for coord={coord}: {a.shape} vs {b.shape}")


def get_time_label(ds: xr.Dataset) -> str:
    vals = ds["time"].values
    t0 = _time_to_yyyymm(vals[0])
    t1 = _time_to_yyyymm(vals[-1])
    return f"{t0}-{t1}"


def apply_time_slice(ds: xr.Dataset, time_start: str | None, time_end: str | None) -> xr.Dataset:
    if time_start is None and time_end is None:
        return ds

    sliced = ds.sel(time=slice(time_start, time_end))
    if sliced.sizes.get("time", 0) == 0:
        raise ValueError(f"No timesteps after time slice: start={time_start}, end={time_end}")
    return sliced


def merge_files(
    file_map: dict[str, Path],
    out_path: Path,
    fillna_mode: str = "none",
    time_start: str | None = None,
    time_end: str | None = None,
) -> None:
    missing = [v for v in VAR_ORDER if v not in file_map]
    if missing:
        raise FileNotFoundError(f"Missing variables: {missing}")

    datasets = []
    ref = None

    for var in VAR_ORDER:
        ds = load_one(file_map[var], var, fillna_mode=fillna_mode)
        assert_same_grid(ds, ref, str(file_map[var]))
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

    merged.attrs["merged_variables"] = ",".join(VAR_ORDER)
    merged.attrs["time_range"] = get_time_label(merged)
    merged.attrs["note"] = (
        "Merged monthly anomaly fields for ENSO model input. "
        f"Variable order target: {','.join(VAR_ORDER)}"
    )

    encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in VAR_ORDER}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_netcdf(out_path, format="NETCDF4", encoding=encoding)

    print(f"[OK] {out_path}  ({merged.sizes['time']} steps; {merged.attrs['time_range']})")


def find_cmip6_groups(inroot: Path) -> list[tuple[str, str, str, Path]]:
    groups: list[tuple[str, str, str, Path]] = []
    for exp_dir in sorted([p for p in inroot.iterdir() if p.is_dir()]):
        for model_dir in sorted([p for p in exp_dir.iterdir() if p.is_dir()]):
            for member_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
                monthly = member_dir / "monthly"
                if monthly.is_dir():
                    groups.append((exp_dir.name, model_dir.name, member_dir.name, monthly))
    return groups


def build_cmip6_file_map(monthly_dir: Path) -> dict[str, Path]:
    fmap: dict[str, Path] = {}
    for path in sorted(monthly_dir.glob("*.nc")):
        var = infer_var_from_name(path)
        if var in VAR_ORDER:
            fmap[var] = path
    return fmap


def run_cmip6(cmip6_root: Path, out_root: Path, fillna_mode: str) -> None:
    groups = find_cmip6_groups(cmip6_root)
    if not groups:
        raise FileNotFoundError(f"No CMIP6 monthly directories found under {cmip6_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    for exp, model, member, monthly_dir in groups:
        fmap = build_cmip6_file_map(monthly_dir)
        if not fmap:
            print(f"[SKIP] no nc files: {monthly_dir}")
            continue

        try:
            out_name = f"input_{model}_{exp}_{member}.nc"
            out_path = out_root / out_name
            merge_files(fmap, out_path, fillna_mode=fillna_mode)
        except Exception as e:
            print(f"[FAIL] {monthly_dir}: {type(e).__name__}: {e}")


def build_reanalysis_file_map(ocean_dir: Path, atm_dir: Path) -> dict[str, Path]:
    fmap: dict[str, Path] = {}

    ocean_files = sorted(ocean_dir.glob("*_1x2.nc"))
    if not ocean_files:
        ocean_files = sorted(ocean_dir.glob("*.nc"))

    atm_files = sorted(atm_dir.glob("*_1x2.nc"))
    if not atm_files:
        atm_files = sorted(atm_dir.glob("*.nc"))

    for path in ocean_files:
        var = infer_var_from_name(path)
        if var in OCEAN_VARS:
            fmap[var] = path

    for path in atm_files:
        var = infer_var_from_name(path)
        if var in ATM_VARS:
            fmap[var] = path

    return fmap


def run_reanalysis(
    base_root: Path,
    out_root: Path,
    fillna_mode: str,
    validation_start: str,
    validation_end: str,
    test_start: str,
    test_end: str,
) -> None:
    oras5_dir = base_root / "oras5"
    godas_dir = base_root / "godas"
    era5_dir = base_root / "era5"

    out_root.mkdir(parents=True, exist_ok=True)

    if oras5_dir.is_dir() and era5_dir.is_dir():
        try:
            fmap = build_reanalysis_file_map(oras5_dir, era5_dir)
            out_path = out_root / "validation_dataset.nc"
            merge_files(
                fmap,
                out_path,
                fillna_mode=fillna_mode,
                time_start=validation_start,
                time_end=validation_end,
            )
        except Exception as e:
            print(f"[FAIL] ORAS5+ERA5: {type(e).__name__}: {e}")

    if godas_dir.is_dir() and era5_dir.is_dir():
        try:
            fmap = build_reanalysis_file_map(godas_dir, era5_dir)
            out_path = out_root / "test_dataset.nc"
            merge_files(
                fmap,
                out_path,
                fillna_mode=fillna_mode,
                time_start=test_start,
                time_end=test_end,
            )
        except Exception as e:
            print(f"[FAIL] GODAS+ERA5: {type(e).__name__}: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create ENSO multivariate input files from 02_interim anomaly fields"
    )
    p.add_argument(
        "--inroot",
        type=Path,
        default=Path("/mnt/d/project/01_ENSO/01_data/02_interim"),
        help="Root directory containing cmip6/, era5/, godas/, oras5/",
    )
    p.add_argument(
        "--outroot",
        type=Path,
        default=Path("/mnt/d/project/01_ENSO/01_data/03_input"),
        help="Output directory for input datasets",
    )
    p.add_argument(
        "--mode",
        choices=["all", "cmip6", "reanalysis"],
        default="all",
        help="What to build",
    )
    p.add_argument(
        "--fillna",
        choices=["none", "paper", "all"],
        default="none",
        help=(
            "none: keep NaN; "
            "paper: fill only ocean vars with 0; "
            "all: fill all variables with 0"
        ),
    )
    p.add_argument(
        "--validation-start",
        default="1958-01",
        help="Start month for validation_dataset.nc",
    )
    p.add_argument(
        "--validation-end",
        default="1976-12",
        help="End month for validation_dataset.nc",
    )
    p.add_argument(
        "--test-start",
        default="1980-01",
        help="Start month for test_dataset.nc",
    )
    p.add_argument(
        "--test-end",
        default="2025-12",
        help="End month for test_dataset.nc",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode in {"all", "cmip6"}:
        run_cmip6(args.inroot / "cmip6", args.outroot, fillna_mode=args.fillna)

    if args.mode in {"all", "reanalysis"}:
        run_reanalysis(
            args.inroot,
            args.outroot,
            fillna_mode=args.fillna,
            validation_start=args.validation_start,
            validation_end=args.validation_end,
            test_start=args.test_start,
            test_end=args.test_end,
        )


if __name__ == "__main__":
    main()
