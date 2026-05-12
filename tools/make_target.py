from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from preprocess_config import (
    INTERIM_ROOT,
    INPUT_ROOT,
    LAT_CANDIDATES,
    LON_CANDIDATES,
    TARGET_ROOT,
    TEST_END,
    TEST_START,
    TIME_CANDIDATES,
    VALIDATION_END,
    VALIDATION_START,
    VAR_ORDER,
    default_test_target_candidates,
    default_validation_target_candidates,
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
    with xr.open_dataset(path, decode_times=True) as ds_in:
        ds = rename_dims_coords(ds_in)
        ds = sort_lon_if_needed(ds)
        if "time" in ds.coords or "time" in ds.dims:
            ds = normalize_monthly_time(ds)
        main = detect_main_var(ds, expected="tos")
        da = ds[main]
        if main != "tos":
            da = da.rename("tos")
        missing_dims = [d for d in ["time", "lat", "lon"] if d not in da.dims]
        if missing_dims:
            raise ValueError(f"{path} missing dims {missing_dims}; got {da.dims}")
        da = da.transpose("time", "lat", "lon").load()
    if da.sizes.get("time", 0) == 0:
        raise ValueError(f"No timesteps found in {path}")
    return da


def load_nino34_series(path: Path) -> xr.Dataset:
    with xr.open_dataset(path, decode_times=True) as ds_in:
        ds = rename_dims_coords(ds_in)
        if "time" not in ds.coords and "time" not in ds.dims:
            raise ValueError(f"{path} has no time coordinate")
        ds = normalize_monthly_time(ds)
        main = detect_main_var(ds, expected="nino34")
        da = ds[main]
        if "time" not in da.dims:
            raise ValueError(f"{path} main variable has no time dimension: {da.dims}")
        da = da.squeeze(drop=True)
        if da.ndim != 1 or da.dims != ("time",):
            raise ValueError(f"{path} expected 1D time series for direct nino34 source; got dims={da.dims}")
        if main != "nino34":
            da = da.rename("nino34")
        ds_out = da.to_dataset(name="nino34").load()

    if ds_out.sizes.get("time", 0) == 0:
        raise ValueError(f"No timesteps found in {path}")

    ds_out["nino34"].attrs.update(
        {
            "long_name": str(ds_out["nino34"].attrs.get("long_name", "Niño3.4 index")) or "Niño3.4 index",
            "source_variable": main,
            "note": "Direct Niño3.4 time series source; copied/renamed without spatial recomputation",
        }
    )
    return ds_out


def slice_nino34_dataset(ds: xr.Dataset, time_start: str | None = None, time_end: str | None = None) -> xr.Dataset:
    da = ds["nino34"]
    if time_start is None or time_end is None:
        inferred_start, inferred_end = infer_time_range(da)
        if time_start is None:
            time_start = inferred_start
        if time_end is None:
            time_end = inferred_end
    assert_requested_time_covered(da, time_start, time_end, label="nino34 source")
    da = da.sel(time=slice(time_start, time_end))
    if da.sizes.get("time", 0) == 0:
        raise ValueError(f"No timesteps after time slice: {time_start} to {time_end}")
    out = da.to_dataset(name="nino34")
    t0 = pd.Timestamp(out["time"].values[0]).strftime("%Y%m")
    t1 = pd.Timestamp(out["time"].values[-1]).strftime("%Y%m")
    out.attrs["time_range"] = f"{t0}-{t1}"
    return out


def load_time_bounds_from_file(path: Path, expected_var: str | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    with xr.open_dataset(path, decode_times=True) as ds_in:
        ds = rename_dims_coords(ds_in)
        if "time" not in ds.coords and "time" not in ds.dims:
            raise ValueError(f"{path} has no time coordinate")
        ds = normalize_monthly_time(ds)
        if expected_var is not None:
            _ = detect_main_var(ds, expected=expected_var)
        vals = ds["time"].values
        if len(vals) == 0:
            raise ValueError(f"No timesteps found in {path}")
        first = pd.Timestamp(vals[0])
        last = pd.Timestamp(vals[-1])
    return first, last


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


def assert_requested_time_covered(da: xr.DataArray, time_start: str | None, time_end: str | None, label: str) -> None:
    first = pd.Timestamp(da["time"].values[0])
    last = pd.Timestamp(da["time"].values[-1])
    if time_start is not None and first > pd.Timestamp(time_start):
        raise ValueError(f"{label} starts at {first:%Y-%m}, later than requested {time_start}")
    if time_end is not None and last < pd.Timestamp(time_end):
        raise ValueError(f"{label} ends at {last:%Y-%m}, earlier than requested {time_end}")


def compute_nino34(da: xr.DataArray, time_start: str | None = None, time_end: str | None = None) -> xr.Dataset:
    if time_start is None or time_end is None:
        inferred_start, inferred_end = infer_time_range(da)
        if time_start is None:
            time_start = inferred_start
        if time_end is None:
            time_end = inferred_end
    assert_requested_time_covered(da, time_start, time_end, label="tos source")
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
    t0 = pd.Timestamp(out["time"].values[0]).strftime("%Y%m")
    t1 = pd.Timestamp(out["time"].values[-1]).strftime("%Y%m")
    out.attrs["time_range"] = f"{t0}-{t1}"
    return out


def save_target(ds: xr.Dataset, out_path: Path, source_path: Path, *, period_source: str | None = None) -> None:
    out = ds.copy()
    out.attrs["source_file"] = str(source_path)
    if period_source is not None:
        out.attrs["period_source"] = period_source
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {"nino34": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    out.to_netcdf(out_path, format="NETCDF4", encoding=encoding)
    extra = f"; period_source={period_source}" if period_source else ""
    print(f"[OK] {out_path}  ({out.sizes['time']} steps; {out.attrs['time_range']}{extra})")


def infer_var_from_name(path: Path) -> str | None:
    name = path.name
    m = re.match(r"^([A-Za-z0-9]+)\.", name)
    if m:
        return m.group(1)
    m = re.match(r"^([A-Za-z0-9]+)_", name)
    if m:
        return m.group(1)
    return None


def find_cmip6_groups(inroot: Path, experiment_name: str | None = None) -> list[tuple[str, str, str, Path]]:
    groups: list[tuple[str, str, str, Path]] = []

    found_canonical = False
    for exp_dir in sorted([p for p in inroot.iterdir() if p.is_dir()]):
        for model_dir in sorted([p for p in exp_dir.iterdir() if p.is_dir()]):
            for member_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
                monthly = member_dir / "monthly"
                if monthly.is_dir():
                    found_canonical = True
                    groups.append((exp_dir.name, model_dir.name, member_dir.name, monthly))
    if found_canonical:
        return groups

    exp_name = experiment_name or inroot.name
    for model_dir in sorted([p for p in inroot.iterdir() if p.is_dir()]):
        for member_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
            monthly = member_dir / "monthly"
            if monthly.is_dir():
                groups.append((exp_name, model_dir.name, member_dir.name, monthly))
    return groups


def find_var_file(monthly_dir: Path, var: str, *, allow_raw_fallback: bool) -> Path | None:
    candidates: list[Path] = []
    for path in sorted(monthly_dir.glob("*.anom_1x2.nc")):
        if infer_var_from_name(path) == var:
            candidates.append(path)
    if allow_raw_fallback:
        for path in sorted(monthly_dir.glob("*.nc")):
            if path in candidates:
                continue
            if infer_var_from_name(path) == var:
                candidates.append(path)
    chosen = select_preferred_path(candidates)
    return chosen


def find_tos_file(monthly_dir: Path, *, allow_raw_fallback: bool) -> Path | None:
    chosen = find_var_file(monthly_dir, "tos", allow_raw_fallback=allow_raw_fallback)
    if chosen is not None and allow_raw_fallback and not is_anomaly_file(chosen):
        print(f"[warn] using non-anomaly CMIP6 tos fallback: {chosen}")
    return chosen


def resolve_default_source(provided: Path | None, candidates: Sequence[Path], label: str) -> Path:
    if provided is not None:
        return provided.expanduser().resolve()
    for path in candidates:
        if path.exists():
            return path
    joined = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Default {label} source not found; tried: {joined}")


def build_validation_target(
    validation_src: Path,
    outroot: Path,
    validation_start: str,
    validation_end: str,
    validation_out: str,
    *,
    validation_src_kind: str = "tos",
) -> None:
    if validation_src_kind == "nino34":
        ds_in = load_nino34_series(validation_src)
        ds = slice_nino34_dataset(ds_in, validation_start, validation_end)
    else:
        da = load_tos(validation_src)
        ds = compute_nino34(da, validation_start, validation_end)
    save_target(ds, outroot / validation_out, validation_src, period_source=f"explicit:{validation_start}~{validation_end}")


def build_test_target(
    test_src: Path,
    outroot: Path,
    test_start: str,
    test_end: str,
    test_out: str,
    *,
    test_src_kind: str = "tos",
) -> None:
    if test_src_kind == "nino34":
        ds_in = load_nino34_series(test_src)
        ds = slice_nino34_dataset(ds_in, test_start, test_end)
    else:
        da = load_tos(test_src)
        ds = compute_nino34(da, test_start, test_end)
    save_target(ds, outroot / test_out, test_src, period_source=f"explicit:{test_start}~{test_end}")


def infer_time_range_from_input_file(input_path: Path) -> tuple[str, str]:
    first, last = load_time_bounds_from_file(input_path, expected_var=None)
    return first.strftime("%Y-%m"), last.strftime("%Y-%m")


def infer_common_time_range_from_monthly_dir(
    monthly_dir: Path,
    predictor_vars: Sequence[str],
    *,
    allow_raw_fallback: bool,
) -> tuple[str, str, list[str]]:
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    missing: list[str] = []
    for var in predictor_vars:
        path = find_var_file(monthly_dir, var, allow_raw_fallback=allow_raw_fallback)
        if path is None:
            missing.append(var)
            continue
        first, last = load_time_bounds_from_file(path, expected_var=var)
        starts.append(first)
        ends.append(last)
    if not starts or not ends:
        raise FileNotFoundError(f"No predictor files found under {monthly_dir}")
    common_start = max(starts)
    common_end = min(ends)
    if common_start > common_end:
        raise ValueError(
            f"No overlapping predictor period in {monthly_dir}: "
            f"common_start={common_start:%Y-%m}, common_end={common_end:%Y-%m}"
        )
    return common_start.strftime("%Y-%m"), common_end.strftime("%Y-%m"), missing


def choose_cmip6_time_range(
    *,
    exp: str,
    model: str,
    member: str,
    monthly_dir: Path,
    cmip6_start: str | None,
    cmip6_end: str | None,
    period_mode: str,
    input_root: Path | None,
    predictor_vars: Sequence[str],
    allow_raw_fallback: bool,
) -> tuple[str | None, str | None, str]:
    if cmip6_start is not None or cmip6_end is not None:
        return cmip6_start, cmip6_end, f"explicit:{cmip6_start or 'auto'}~{cmip6_end or 'auto'}"

    input_path = None
    if input_root is not None:
        candidate = input_root / f"input_{model}_{exp}_{member}.nc"
        if candidate.exists():
            input_path = candidate

    if period_mode in {"input", "auto"} and input_path is not None:
        start, end = infer_time_range_from_input_file(input_path)
        return start, end, f"input:{input_path}"

    if period_mode in {"common", "auto"}:
        try:
            start, end, missing = infer_common_time_range_from_monthly_dir(
                monthly_dir,
                predictor_vars,
                allow_raw_fallback=allow_raw_fallback,
            )
            source = "common_predictor_overlap"
            if missing:
                source += f";missing={','.join(missing)}"
            return start, end, source
        except Exception:
            if period_mode == "common":
                raise

    return None, None, "tos_full_period"


def build_cmip6_targets(
    cmip6_root: Path,
    outroot: Path,
    cmip6_start: str | None,
    cmip6_end: str | None,
    *,
    allow_raw_fallback: bool,
    experiment_name: str | None,
    include_experiments: set[str] | None,
    exclude_models: set[str],
    input_root: Path | None,
    period_mode: str,
    predictor_vars: Sequence[str],
) -> None:
    groups = find_cmip6_groups(cmip6_root, experiment_name=experiment_name)
    if not groups:
        raise FileNotFoundError(
            f"No CMIP6 monthly directories found under {cmip6_root}. "
            "Expected either <root>/<exp>/<model>/<member>/monthly or <root>/<model>/<member>/monthly."
        )
    for exp, model, member, monthly_dir in groups:
        if include_experiments is not None and exp not in include_experiments:
            continue
        if model in exclude_models:
            print(f"[SKIP] excluded model: {exp}/{model}/{member}")
            continue
        tos_path = find_tos_file(monthly_dir, allow_raw_fallback=allow_raw_fallback)
        if tos_path is None:
            print(f"[SKIP] no compatible tos file: {monthly_dir}")
            continue
        try:
            time_start, time_end, period_source = choose_cmip6_time_range(
                exp=exp,
                model=model,
                member=member,
                monthly_dir=monthly_dir,
                cmip6_start=cmip6_start,
                cmip6_end=cmip6_end,
                period_mode=period_mode,
                input_root=input_root,
                predictor_vars=predictor_vars,
                allow_raw_fallback=allow_raw_fallback,
            )
            da = load_tos(tos_path)
            ds = compute_nino34(da, time_start, time_end)
            out_name = f"target_{model}_{exp}_{member}.nc"
            save_target(ds, outroot / out_name, tos_path, period_source=period_source)
        except Exception as e:
            print(f"[FAIL] {monthly_dir}: {type(e).__name__}: {e}")


def parse_csv_set(text: str | None) -> set[str] | None:
    if text is None:
        return None
    parts = [x.strip() for x in text.split(",") if x.strip()]
    return set(parts)


def parse_csv_list(text: str | None) -> list[str] | None:
    if text is None:
        return None
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create Niño3.4 target files from SST anomaly data")
    p.add_argument(
        "--validation-src",
        type=Path,
        default=None,
        help="Validation SST anomaly file. Default is the canonical SODA tos anomaly.",
    )
    p.add_argument(
        "--test-src",
        type=Path,
        default=None,
        help="Test SST anomaly file. Default is the canonical GODAS tos anomaly.",
    )
    p.add_argument(
        "--validation-src-kind",
        choices=["tos", "nino34"],
        default="tos",
        help="Interpret --validation-src as gridded tos anomaly or direct nino34 time series.",
    )
    p.add_argument(
        "--test-src-kind",
        choices=["tos", "nino34"],
        default="tos",
        help="Interpret --test-src as gridded tos anomaly or direct nino34 time series.",
    )
    p.add_argument(
        "--cmip6-root",
        type=Path,
        default=INTERIM_ROOT / "cmip6",
        help=(
            "CMIP6 interim root. Supports both <root>/<exp>/<model>/<member>/monthly and "
            "single-experiment layout <root>/<model>/<member>/monthly."
        ),
    )
    p.add_argument(
        "--cmip6-exp-name",
        default=None,
        help=(
            "Experiment name to use when --cmip6-root points directly to a single experiment directory "
            "such as .../cmip6/historical_ssp370. Default: use the basename of --cmip6-root."
        ),
    )
    p.add_argument(
        "--only-experiments",
        default=None,
        help="Comma-separated experiment names to include, e.g. historical_ssp370",
    )
    p.add_argument(
        "--exclude-models",
        default="MRI-ESM2-0",
        help="Comma-separated model names to exclude. Default: MRI-ESM2-0",
    )
    p.add_argument("--outroot", type=Path, default=TARGET_ROOT, help="Output directory for target files")
    p.add_argument("--input-root", type=Path, default=INPUT_ROOT, help="Directory containing input_<model>_<exp>_<member>.nc files")
    p.add_argument(
        "--period-mode",
        choices=["auto", "input", "common", "tos"],
        default="auto",
        help=(
            "How to choose each CMIP6 target time range when --cmip6-start/--cmip6-end are not given. "
            "auto: use corresponding input file period if present, else predictor common overlap, else tos full period. "
            "input: require corresponding input file period. "
            "common: infer overlap across predictor variables in monthly dir. "
            "tos: use the tos file full period."
        ),
    )
    p.add_argument(
        "--predictor-vars",
        default=",".join(VAR_ORDER),
        help="Comma-separated predictor variable order used to infer common overlap. Default: preprocess_config.VAR_ORDER",
    )
    p.add_argument("--validation-start", default=VALIDATION_START, help="Start month for validation target")
    p.add_argument("--validation-end", default=VALIDATION_END, help="End month for validation target")
    p.add_argument("--test-start", default=TEST_START, help="Start month for test target")
    p.add_argument("--test-end", default=TEST_END, help="End month for test target")
    p.add_argument("--cmip6-start", default=None, help="Optional common start month for CMIP6 targets")
    p.add_argument("--cmip6-end", default=None, help="Optional common end month for CMIP6 targets")
    p.add_argument("--validation-out", default="validation_target.nc", help="Output filename for validation target")
    p.add_argument("--test-out", default="test_target.nc", help="Output filename for test target")
    p.add_argument(
        "--allow-raw-fallback",
        action="store_true",
        help="Allow fallback to non-anomaly CMIP6 files when anomaly files are missing.",
    )
    p.add_argument("--skip-validation-target", action="store_true", help="Skip creation of validation_target.nc")
    p.add_argument("--skip-cmip6-targets", action="store_true", help="Skip creation of CMIP6 target files")
    p.add_argument("--skip-test-target", action="store_true", help="Skip creation of test_target.nc")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    include_experiments = parse_csv_set(args.only_experiments)
    exclude_models = parse_csv_set(args.exclude_models) or set()
    predictor_vars = parse_csv_list(args.predictor_vars) or list(VAR_ORDER)

    if not args.skip_validation_target:
        validation_src = resolve_default_source(args.validation_src, default_validation_target_candidates(), label="validation")
        build_validation_target(
            validation_src=validation_src,
            outroot=args.outroot,
            validation_start=args.validation_start,
            validation_end=args.validation_end,
            validation_out=args.validation_out,
            validation_src_kind=args.validation_src_kind,
        )
    if not args.skip_cmip6_targets:
        build_cmip6_targets(
            cmip6_root=args.cmip6_root,
            outroot=args.outroot,
            cmip6_start=args.cmip6_start,
            cmip6_end=args.cmip6_end,
            allow_raw_fallback=args.allow_raw_fallback,
            experiment_name=args.cmip6_exp_name,
            include_experiments=include_experiments,
            exclude_models=exclude_models,
            input_root=args.input_root,
            period_mode=args.period_mode,
            predictor_vars=predictor_vars,
        )
    if not args.skip_test_target:
        test_src = resolve_default_source(args.test_src, default_test_target_candidates(), label="test")
        build_test_target(
            test_src=test_src,
            outroot=args.outroot,
            test_start=args.test_start,
            test_end=args.test_end,
            test_out=args.test_out,
            test_src_kind=args.test_src_kind,
        )


if __name__ == "__main__":
    main()
