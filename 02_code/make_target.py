from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

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


def detect_main_var(ds: xr.Dataset, expected: str = "tos") -> str:
    if expected in ds.data_vars:
        return expected

    data_vars = list(ds.data_vars)
    if len(data_vars) == 1:
        return data_vars[0]

    for v in data_vars:
        if v.lower() == expected.lower():
            return v

    for v in data_vars:
        if expected in v.lower():
            return v

    raise ValueError(f"Cannot determine main variable from {data_vars}")


def load_tos(path: Path) -> xr.DataArray:
    ds = xr.open_dataset(path, decode_times=True)
    ds = rename_dims_coords(ds)

    if "time" in ds.coords or "time" in ds.dims:
        ds = normalize_monthly_time(ds)

    main = detect_main_var(ds, expected="tos")
    if main != "tos":
        ds = ds.rename({main: "tos"})

    da = ds["tos"]

    missing_dims = [d for d in ["time", "lat", "lon"] if d not in da.dims]
    if missing_dims:
        raise ValueError(f"{path} missing dims {missing_dims}; got {da.dims}")

    return da.transpose("time", "lat", "lon")


def normalize_lon_to_360(da: xr.DataArray) -> xr.DataArray:
    lon = da["lon"]
    if lon.ndim != 1:
        raise ValueError("Only 1D lon coordinate is supported")

    lon360 = xr.where(lon < 0, lon + 360, lon)
    order = np.argsort(lon360.values)
    da = da.assign_coords(lon=("lon", lon360.values))
    da = da.isel(lon=order)
    return da


def sel_lat_band(da: xr.DataArray, lat_min: float, lat_max: float) -> xr.DataArray:
    lat = da["lat"].values
    if lat[0] <= lat[-1]:
        return da.sel(lat=slice(lat_min, lat_max))
    return da.sel(lat=slice(lat_max, lat_min))


def infer_time_range(da: xr.DataArray) -> tuple[str, str]:
    start = pd.Timestamp(da["time"].values[0]).strftime("%Y-%m")
    end = pd.Timestamp(da["time"].values[-1]).strftime("%Y-%m")
    return start, end


def compute_nino34(da: xr.DataArray, time_start: str | None = None, time_end: str | None = None) -> xr.Dataset:
    if time_start is None or time_end is None:
        inferred_start, inferred_end = infer_time_range(da)
        if time_start is None:
            time_start = inferred_start
        if time_end is None:
            time_end = inferred_end

    da = da.sel(time=slice(time_start, time_end))
    if da.sizes.get("time", 0) == 0:
        raise ValueError(f"No timesteps after time slice: {time_start} to {time_end}")

    da = normalize_lon_to_360(da)
    da = sel_lat_band(da, -5.0, 5.0)
    da = da.sel(lon=slice(190.0, 240.0))

    if da.sizes.get("lat", 0) == 0 or da.sizes.get("lon", 0) == 0:
        raise ValueError("Niño3.4 region selection returned empty lat/lon dimensions")

    weights = xr.DataArray(
        np.cos(np.deg2rad(da["lat"].values)),
        dims=("lat",),
        coords={"lat": da["lat"]},
        name="weights",
    )

    nino34 = da.weighted(weights).mean(dim=("lat", "lon"), skipna=True)
    out = nino34.to_dataset(name="nino34")
    out["nino34"].attrs.update(
        {
            "long_name": "Niño3.4 index",
            "units": str(da.attrs.get("units", "")),
            "source_variable": "tos",
            "region": "5S-5N, 170W-120W",
            "note": "Area-weighted mean of monthly SST anomaly over Niño3.4 region",
        }
    )
    out.attrs["time_range"] = (
        f"{str(out['time'].dt.strftime('%Y%m').values[0])}-"
        f"{str(out['time'].dt.strftime('%Y%m').values[-1])}"
    )
    return out


def save_target(ds: xr.Dataset, out_path: Path, source_path: Path) -> None:
    out = ds.copy()
    out.attrs["source_file"] = str(source_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {"nino34": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    out.to_netcdf(out_path, format="NETCDF4", encoding=encoding)
    print(f"[OK] {out_path}  ({out.sizes['time']} steps; {out.attrs['time_range']})")


def infer_var_from_name(path: Path) -> str | None:
    name = path.name
    m = re.match(r"^([A-Za-z0-9]+)\.", name)
    if m:
        return m.group(1)
    m = re.match(r"^([A-Za-z0-9]+)_", name)
    if m:
        return m.group(1)
    return None


def find_cmip6_groups(inroot: Path) -> list[tuple[str, str, str, Path]]:
    groups: list[tuple[str, str, str, Path]] = []
    for exp_dir in sorted([p for p in inroot.iterdir() if p.is_dir()]):
        for model_dir in sorted([p for p in exp_dir.iterdir() if p.is_dir()]):
            for member_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
                monthly = member_dir / "monthly"
                if monthly.is_dir():
                    groups.append((exp_dir.name, model_dir.name, member_dir.name, monthly))
    return groups


def find_tos_file(monthly_dir: Path) -> Path | None:
    candidates = sorted(monthly_dir.glob("*.nc"))
    for path in candidates:
        if infer_var_from_name(path) == "tos":
            return path
    return None


def build_validation_target(
    validation_src: Path,
    outroot: Path,
    validation_start: str,
    validation_end: str,
    validation_out: str,
) -> None:
    da = load_tos(validation_src)
    ds = compute_nino34(da, validation_start, validation_end)
    save_target(ds, outroot / validation_out, validation_src)


def build_test_target(
    test_src: Path,
    outroot: Path,
    test_start: str,
    test_end: str,
    test_out: str,
) -> None:
    da = load_tos(test_src)
    ds = compute_nino34(da, test_start, test_end)
    save_target(ds, outroot / test_out, test_src)


def build_cmip6_targets(
    cmip6_root: Path,
    outroot: Path,
    cmip6_start: str | None,
    cmip6_end: str | None,
) -> None:
    groups = find_cmip6_groups(cmip6_root)
    if not groups:
        raise FileNotFoundError(f"No CMIP6 monthly directories found under {cmip6_root}")

    for exp, model, member, monthly_dir in groups:
        tos_path = find_tos_file(monthly_dir)
        if tos_path is None:
            print(f"[SKIP] no tos file: {monthly_dir}")
            continue

        try:
            da = load_tos(tos_path)
            ds = compute_nino34(da, cmip6_start, cmip6_end)
            out_name = f"target_{model}_{exp}_{member}.nc"
            save_target(ds, outroot / out_name, tos_path)
        except Exception as e:
            print(f"[FAIL] {monthly_dir}: {type(e).__name__}: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create Niño3.4 target files from SST anomaly data")
    p.add_argument(
        "--validation-src",
        type=Path,
        default=Path("/mnt/d/project/01_ENSO/01_data/02_interim/oras5/tos.195801-197812.anom_1x2.nc"),
        help="Validation SST anomaly file (usually ORAS5 tos anomaly)",
    )
    p.add_argument(
        "--test-src",
        type=Path,
        default=Path("/mnt/d/project/01_ENSO/01_data/02_interim/godas/tos.198001-202512.anom_1x2.nc"),
        help="Test SST anomaly file (usually GODAS tos anomaly)",
    )
    p.add_argument(
        "--cmip6-root",
        type=Path,
        default=Path("/mnt/d/project/01_ENSO/01_data/02_interim/cmip6"),
        help="CMIP6 interim root directory containing <experiment>/<model>/<member>/monthly/",
    )
    p.add_argument(
        "--outroot",
        type=Path,
        default=Path("/mnt/d/project/01_ENSO/01_data/04_target"),
        help="Output directory for target files",
    )
    p.add_argument("--validation-start", default="1958-01", help="Start month for validation target")
    p.add_argument("--validation-end", default="1978-12", help="End month for validation target")
    p.add_argument("--test-start", default="1980-01", help="Start month for test target")
    p.add_argument("--test-end", default="2025-12", help="End month for test target")
    p.add_argument(
        "--cmip6-start",
        default=None,
        help="Optional common start month for CMIP6 targets (default: first available month in each file)",
    )
    p.add_argument(
        "--cmip6-end",
        default=None,
        help="Optional common end month for CMIP6 targets (default: last available month in each file)",
    )
    p.add_argument(
        "--validation-out",
        default="validation_target.nc",
        help="Output filename for validation target",
    )
    p.add_argument(
        "--test-out",
        default="test_target.nc",
        help="Output filename for test target",
    )
    p.add_argument(
        "--make-test-target",
        action="store_true",
        help="Also create test_target.nc from GODAS tos anomaly",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    build_validation_target(
        validation_src=args.validation_src,
        outroot=args.outroot,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        validation_out=args.validation_out,
    )

    build_cmip6_targets(
        cmip6_root=args.cmip6_root,
        outroot=args.outroot,
        cmip6_start=args.cmip6_start,
        cmip6_end=args.cmip6_end,
    )

    build_test_target(
        test_src=args.test_src,
        outroot=args.outroot,
        test_start=args.test_start,
        test_end=args.test_end,
        test_out=args.test_out,
    )


if __name__ == "__main__":
    main()
