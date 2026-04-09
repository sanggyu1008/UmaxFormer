#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from preprocess_config import GODAS_RAW_FILE_CANDIDATES, resolve_existing_path

DEPTH_DIM_CANDIDATES = ["level", "lev", "zlev", "depth", "olevel", "z", "deptht", "st_ocean"]
TIME_DIM_CANDIDATES = ["time", "t"]


def _default_project_root() -> Path:
    for cand in (
        Path("/home/sanggyu1008/project/01_ENSO"),
        Path("/mnt/d/project/01_ENSO"),
    ):
        if cand.exists():
            return cand
    return Path.cwd()


PROJECT_ROOT = _default_project_root()
DEFAULT_GODAS_BASE = PROJECT_ROOT / "01_data/01_raw/godas"
DEFAULT_SODA_BASE = PROJECT_ROOT / "01_data/01_raw/soda"


def append_history(ds: xr.Dataset, message: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = f"{now}: {message}"
    old = ds.attrs.get("history")
    ds.attrs["history"] = f"{entry}\n{old}" if old else entry


def find_dim(da: xr.DataArray, candidates: list[str], kind: str) -> str:
    for name in candidates:
        if name in da.dims:
            return name

    for dim in da.dims:
        if dim in da.coords:
            units = str(da[dim].attrs.get("units", "")).lower()
            if kind == "depth" and (units in ("m", "meter", "meters") or "meter" in units or "metre" in units):
                return dim
            if kind == "time" and ("since" in units or "days" in units or "hours" in units):
                return dim

    raise ValueError(f"Cannot find {kind} dimension. dims={da.dims}")


def carry_aux_vars(ds: xr.Dataset, out: xr.Dataset, main_var: str) -> xr.Dataset:
    out_dims = set(out[main_var].dims)
    for name, var in ds.variables.items():
        if name in out.variables or name == main_var or name in ds.coords:
            continue
        if set(var.dims).issubset(out_dims):
            out[name] = var
    return out


def maybe_chunk(da: xr.DataArray, dim: str, size: int) -> xr.DataArray:
    if int(size) <= 0:
        return da
    try:
        import dask  # noqa: F401
    except Exception:
        return da
    return da.chunk({dim: int(size)})


def infer_bounds_from_midpoints(z: xr.DataArray) -> xr.DataArray:
    """Infer cell bounds from depth midpoints. The top edge is forced to 0 m."""
    zvals = z.values.astype("float64")
    if zvals.ndim != 1:
        raise ValueError("Depth coordinate is not 1D; cannot infer bounds robustly.")
    if zvals.size == 0:
        raise ValueError("Empty depth coordinate.")

    edges = np.empty(zvals.size + 1, dtype="float64")
    if zvals.size == 1:
        edges[0] = 0.0
        edges[1] = float(zvals[0])
    else:
        edges[1:-1] = 0.5 * (zvals[1:] + zvals[:-1])
        edges[-1] = zvals[-1] + (zvals[-1] - edges[-2])
        edges[0] = 0.0

    bounds = xr.DataArray(
        np.vstack([edges[:-1], edges[1:]]).T,
        dims=(z.dims[0], "bnds"),
        coords={z.dims[0]: z, "bnds": [0, 1]},
        name=f"{z.dims[0]}_bnds_midpoint",
    )
    bounds.attrs["comment"] = "Inferred bounds from depth midpoints; top edge forced to 0 m."
    return bounds


def overlap_thickness(bounds: xr.DataArray, zmin: float, zmax: float) -> xr.DataArray:
    lo = bounds.isel(bnds=0)
    hi = bounds.isel(bnds=1)
    lower = xr.where(lo < hi, lo, hi)
    upper = xr.where(lo < hi, hi, lo)

    ov = xr.apply_ufunc(np.minimum, upper, zmax) - xr.apply_ufunc(np.maximum, lower, zmin)
    ov = ov.clip(min=0.0)
    ov.name = "dz_overlap"
    ov.attrs["units"] = "m"
    return ov


def compute_ohc(
    src_path: Path,
    out_path: Path,
    input_var: str,
    output_var: str,
    source_name: str,
    zmax: float,
    rho0: float,
    cp0: float,
    to_degC: bool,
    time_chunk: int,
    mask_shallow: bool,
    shallow_eps: float,
    carry_aux: bool,
    add_history: bool,
) -> None:
    if int(time_chunk) < 1:
        raise ValueError(f"time_chunk must be >= 1, got {time_chunk}")

    with xr.open_dataset(src_path, decode_times=False) as ds:
        if input_var not in ds.variables:
            raise ValueError(f"Variable '{input_var}' not found in {src_path}")

        th = ds[input_var]
        depth_dim = find_dim(th, DEPTH_DIM_CANDIDATES, kind="depth")
        time_dim = find_dim(th, TIME_DIM_CANDIDATES, kind="time")

        th = th.sortby(depth_dim)
        th = maybe_chunk(th, time_dim, time_chunk)

        z = th[depth_dim]
        bounds = infer_bounds_from_midpoints(z)
        dz = overlap_thickness(bounds, 0.0, float(zmax)).astype("float64")

        units = str(th.attrs.get("units", "")).strip().lower()
        th_use = th
        temp_scale = units or "unknown"
        if to_degC and units in ("k", "kelvin"):
            th_use = th - 273.15
            temp_scale = "degC"

        if mask_shallow:
            wet_3d = th_use.isel({time_dim: 0}).notnull()
            coverage = dz.where(wet_3d).sum(dim=depth_dim, skipna=True)
            deep_enough = coverage >= (float(zmax) - float(shallow_eps))
        else:
            deep_enough = None

        ocean_mask_time = th_use.notnull().any(dim=depth_dim)
        th_filled = th_use.fillna(0.0)

        tint = (th_filled * dz).sum(dim=depth_dim, skipna=False)
        ohc = tint * (float(rho0) * float(cp0))
        ohc = ohc.where(ocean_mask_time)
        if mask_shallow:
            ohc = ohc.where(deep_enough)

        ohc = ohc.astype("float32")
        ohc.name = output_var
        out = ohc.to_dataset(name=output_var)
        out.attrs = dict(ds.attrs)

        if carry_aux:
            out = carry_aux_vars(ds, out, output_var)

        for key in ("_FillValue", "missing_value", "valid_min", "valid_max", "valid_range"):
            out[output_var].attrs.pop(key, None)

        out.attrs["variable_id"] = output_var
        if "notes" in out.attrs:
            out.attrs["notes"] = str(out.attrs["notes"]).replace(input_var, output_var)

        out[output_var].attrs.update(
            {
                "long_name": f"Ocean heat content integrated from 0 to {float(zmax)} m",
                "units": "J m-2",
                "source_variable": input_var,
                "source_file": src_path.name,
                "source_dataset": source_name,
                "rho0": float(rho0),
                "cp0": float(cp0),
                "zmin": 0.0,
                "zmax": float(zmax),
                "dz_method": "midpoint_inferred_bounds_top0m",
                "temperature_scale_for_integration": temp_scale,
                "shallow_masked": int(mask_shallow),
                "comment": (
                    f"Computed from {source_name} {input_var} by vertical integration using dz from "
                    "midpoint-inferred bounds; overlap thickness with [0, zmax]."
                ),
            }
        )

        if add_history:
            msg = (
                f"Created {output_var} from {src_path.name} using {input_var} integrated over 0-{zmax:g} m "
                f"(rho0={rho0:g}, cp0={cp0:g}, temp_scale={temp_scale}, mask_shallow={int(mask_shallow)})"
            )
            append_history(out, msg)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fill = np.float32(1.0e20)
        encoding = {
            output_var: {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "dtype": "float32",
                "_FillValue": fill,
            }
        }
        out.to_netcdf(out_path, encoding=encoding)


def replace_prefix(name: str, old: str, new: str) -> str:
    return new + name[len(old):] if name.startswith(old) else name.replace(old, new, 1)


def add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--zmax", default=300.0, type=float)
    p.add_argument("--rho0", default=1025.0, type=float)
    p.add_argument("--cp0", default=3992.0, type=float)
    p.add_argument("--to-degC", action="store_true")
    p.add_argument("--time-chunk", default=12, type=int)
    p.add_argument("--mask-shallow", action="store_true")
    p.add_argument("--shallow-eps", default=1e-3, type=float)
    p.add_argument("--overwrite", action="store_true")


def run_cmip6(args: argparse.Namespace) -> None:
    infiles = sorted(args.inroot.glob(args.pattern))
    if not infiles:
        raise SystemExit(f"No input files found under {args.inroot} with pattern: {args.pattern}")

    for src_path in infiles:
        rel = src_path.relative_to(args.inroot)
        outname = replace_prefix(src_path.name, args.input_prefix, args.output_prefix)
        out_path = args.outroot / rel.parent / outname

        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path}")
            continue

        print(f"[IN ] {src_path}")
        print(f"[OUT] {out_path}")
        compute_ohc(
            src_path=src_path,
            out_path=out_path,
            input_var=args.input_var,
            output_var=args.output_var,
            source_name="CMIP6",
            zmax=args.zmax,
            rho0=args.rho0,
            cp0=args.cp0,
            to_degC=args.to_degC,
            time_chunk=args.time_chunk,
            mask_shallow=args.mask_shallow,
            shallow_eps=args.shallow_eps,
            carry_aux=False,
            add_history=False,
        )


def run_godas(args: argparse.Namespace) -> None:
    base_dir = args.base_dir.expanduser().resolve()
    if not base_dir.exists():
        raise SystemExit(f"Base directory does not exist: {base_dir}")

    src_path = base_dir / args.src_file
    out_path = base_dir / args.out_file

    if not src_path.exists():
        fallback = resolve_existing_path(base_dir, GODAS_RAW_FILE_CANDIDATES.get(args.input_var, ()))
        if fallback is not None:
            print(f"[info] using fallback GODAS file: {fallback.name}")
            src_path = fallback
    if not src_path.exists():
        raise SystemExit(f"Missing input file: {src_path}")
    if out_path.exists() and not args.overwrite:
        print(f"[skip] {out_path}")
        return

    print(f"[IN ] {src_path}")
    print(f"[OUT] {out_path}")
    compute_ohc(
        src_path=src_path,
        out_path=out_path,
        input_var=args.input_var,
        output_var=args.output_var,
        source_name="GODAS",
        zmax=args.zmax,
        rho0=args.rho0,
        cp0=args.cp0,
        to_degC=args.to_degC,
        time_chunk=args.time_chunk,
        mask_shallow=args.mask_shallow,
        shallow_eps=args.shallow_eps,
        carry_aux=True,
        add_history=True,
    )


def run_soda(args: argparse.Namespace) -> None:
    base_dir = args.base_dir.expanduser().resolve()
    if not base_dir.exists():
        raise SystemExit(f"Base directory does not exist: {base_dir}")

    src_path = base_dir / args.src_file
    out_path = base_dir / args.out_file

    if not src_path.exists():
        raise SystemExit(f"Missing input file: {src_path}")
    if out_path.exists() and not args.overwrite:
        print(f"[skip] {out_path}")
        return

    print(f"[IN ] {src_path}")
    print(f"[OUT] {out_path}")
    compute_ohc(
        src_path=src_path,
        out_path=out_path,
        input_var=args.input_var,
        output_var=args.output_var,
        source_name="SODA2.2.4",
        zmax=args.zmax,
        rho0=args.rho0,
        cp0=args.cp0,
        to_degC=args.to_degC,
        time_chunk=args.time_chunk,
        mask_shallow=args.mask_shallow,
        shallow_eps=args.shallow_eps,
        carry_aux=True,
        add_history=True,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create ohc300 from CMIP6 thetao or reanalysis temperature files (GODAS/SODA)."
    )
    sub = p.add_subparsers(dest="source", required=True)

    p_cmip6 = sub.add_parser("cmip6", help="Batch mode for CMIP6 thetao files")
    p_cmip6.add_argument("--inroot", required=True, type=Path)
    p_cmip6.add_argument("--outroot", required=True, type=Path)
    p_cmip6.add_argument("--pattern", default="*/*/monthly/thetao_*.nc")
    p_cmip6.add_argument("--input-var", default="thetao")
    p_cmip6.add_argument("--output-var", default="ohc300")
    p_cmip6.add_argument("--input-prefix", default="thetao_")
    p_cmip6.add_argument("--output-prefix", default="ohc300_")
    add_shared_args(p_cmip6)
    p_cmip6.set_defaults(func=run_cmip6)

    p_godas = sub.add_parser("godas", help="Single-file mode for GODAS pottmp")
    p_godas.add_argument("--base-dir", type=Path, default=DEFAULT_GODAS_BASE)
    p_godas.add_argument("--src-file", default="pottmp.198001-202512.nc")
    p_godas.add_argument("--out-file", default="ohc300.198001-202512.nc")
    p_godas.add_argument("--input-var", default="pottmp")
    p_godas.add_argument("--output-var", default="ohc300")
    add_shared_args(p_godas)
    p_godas.set_defaults(func=run_godas)

    p_soda = sub.add_parser("soda", help="Single-file mode for SODA temperature")
    p_soda.add_argument("--base-dir", type=Path, default=DEFAULT_SODA_BASE)
    p_soda.add_argument("--src-file", default="SODA_temp.nc")
    p_soda.add_argument("--out-file", default="SODA_ohc300.nc")
    p_soda.add_argument("--input-var", default="temp")
    p_soda.add_argument("--output-var", default="ohc300")
    add_shared_args(p_soda)
    p_soda.set_defaults(func=run_soda)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
