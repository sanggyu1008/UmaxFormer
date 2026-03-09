#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import xarray as xr

DEPTH_DIM_CANDIDATES = ["lev", "zlev", "depth", "olevel", "z", "deptht", "st_ocean"]
TIME_DIM_CANDIDATES = ["time", "t"]

def find_dim(da: xr.DataArray, candidates: list[str], kind: str) -> str:
    for name in candidates:
        if name in da.dims:
            return name
    # fallback: coords units 기반 탐색
    for dim in da.dims:
        if dim in da.coords:
            units = str(da[dim].attrs.get("units", "")).lower()
            if kind == "depth" and (units in ("m", "meter", "meters") or "meter" in units):
                return dim
            if kind == "time" and ("since" in units or "days" in units or "hours" in units):
                return dim
    raise ValueError(f"Cannot find {kind} dimension. dims={da.dims}")

def infer_bounds_from_midpoints(z: xr.DataArray) -> xr.DataArray:
    """Always infer bounds from midpoints (A: ignore provided bounds).
    Top edge is forced to 0 m for physical consistency.
    """
    zvals = z.values.astype("float64")
    if zvals.ndim != 1:
        raise ValueError("Depth coordinate is not 1D; cannot infer bounds robustly.")

    # 정렬(혹시라도 depth가 역순/무질서인 경우)
    # xarray sort는 바깥에서 수행 권장. 여기서는 z 자체가 정렬되어 있다고 가정.
    edges = np.empty(zvals.size + 1, dtype="float64")
    edges[1:-1] = 0.5 * (zvals[1:] + zvals[:-1])

    # bottom edge extrapolation
    edges[-1] = zvals[-1] + (zvals[-1] - edges[-2])

    # top edge: 0 m 강제
    edges[0] = 0.0

    b = xr.DataArray(
        np.vstack([edges[:-1], edges[1:]]).T,
        dims=(z.dims[0], "bnds"),
        coords={z.dims[0]: z, "bnds": [0, 1]},
        name=f"{z.dims[0]}_bnds_midpoint",
    )
    b.attrs["comment"] = "Inferred bounds from depth midpoints; top edge forced to 0 m."
    return b

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

def compute_ohc300(
    infile: Path,
    outfile: Path,
    zmax: float,
    rho0: float,
    cp0: float,
    to_degC: bool,
    time_chunk: int,
    mask_shallow: bool,
    shallow_eps: float,
):
    ds = xr.open_dataset(infile, decode_times=False)

    if "thetao" not in ds.variables:
        raise ValueError(f"'thetao' not found in {infile}")

    th = ds["thetao"]
    depth_dim = find_dim(th, DEPTH_DIM_CANDIDATES, kind="depth")
    time_dim  = find_dim(th, TIME_DIM_CANDIDATES,  kind="time")

    th = th.sortby(depth_dim)
    th = th.chunk({time_dim: int(time_chunk)})

    z = ds[depth_dim]
    bnds = infer_bounds_from_midpoints(z)          # (A)
    dz = overlap_thickness(bnds, 0.0, zmax)        # (depth_dim)

    # Kelvin -> degC 옵션 (anomaly 계획이더라도 수치 안정성 측면에서 보통 이게 낫다)
    units = str(th.attrs.get("units", "")).lower()
    th_use = th
    if to_degC and units in ("k", "kelvin"):
        th_use = th - 273.15

    # 얕은 바다 마스크(D): 0–300을 전부 채우는지 확인
    if mask_shallow:
        # ocean mask는 시간에 따라 바뀌지 않는다는 가정 하에 첫 시점 사용(빠르고 충분히 안전한 경우가 대부분)
        wet_3d = th_use.isel({time_dim: 0}).notnull()          # (depth, y, x)
        coverage = (dz.where(wet_3d) ).sum(dim=depth_dim)      # (y, x)
        deep_enough = coverage >= (zmax - shallow_eps)         # (y, x)
    else:
        deep_enough = None

    # 적분: NaN은 0으로 채우고, 결과는 마스크로 되돌림
    ocean_mask_time = th_use.notnull().any(depth_dim)  # (time, y, x)
    th_filled = th_use.where(th_use.notnull(), 0.0)

    tint = (th_filled * dz).sum(dim=depth_dim)          # degC*m
    ohc = tint * (rho0 * cp0)                           # J/m^2

    # land/missing 복원
    ohc = ohc.where(ocean_mask_time)

    # (D) shallow(<300m)인 격자점 전체 시간 결측 처리
    if mask_shallow:
        ohc = ohc.where(deep_enough)

    out = ohc.to_dataset(name="ohc300")
    out.attrs.update(ds.attrs)

    out["ohc300"].attrs.update({
        "long_name": f"Ocean heat content integrated from 0 to {zmax} m",
        "units": "J m-2",
        "rho0": float(rho0),
        "cp0": float(cp0),
        "zmin": 0.0,
        "zmax": float(zmax),
        "dz_method": "midpoint_inferred_bounds_top0m",
        "shallow_masked": int(mask_shallow),
        "comment": "Computed from thetao by vertical integration using dz from midpoint-inferred bounds; overlap thickness with [0,zmax]."
    })

    outfile.parent.mkdir(parents=True, exist_ok=True)

    enc = {"ohc300": {"zlib": True, "complevel": 4, "shuffle": True, "dtype": "float32"}}
    out.to_netcdf(outfile, encoding=enc)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inroot", required=True, type=Path)
    p.add_argument("--outroot", required=True, type=Path)
    p.add_argument("--zmax", default=300.0, type=float)
    p.add_argument("--rho0", default=1025.0, type=float)
    p.add_argument("--cp0", default=3992.0, type=float)
    p.add_argument("--to-degC", action="store_true")
    p.add_argument("--time-chunk", default=12, type=int)  # (G)
    p.add_argument("--mask-shallow", action="store_true") # (D)
    p.add_argument("--shallow-eps", default=1e-3, type=float)
    args = p.parse_args()

    infiles = sorted(args.inroot.glob("*/*/monthly/thetao_*.nc"))
    if not infiles:
        raise SystemExit(f"No thetao files found under {args.inroot}")

    for f in infiles:
        rel = f.relative_to(args.inroot)
        outname = f.name.replace("thetao_", "ohc300_")
        outpath = args.outroot / rel.parent / outname

        print(f"[IN ] {f}")
        print(f"[OUT] {outpath}")
        compute_ohc300(
            infile=f,
            outfile=outpath,
            zmax=args.zmax,
            rho0=args.rho0,
            cp0=args.cp0,
            to_degC=args.to_degC,
            time_chunk=args.time_chunk,
            mask_shallow=args.mask_shallow,
            shallow_eps=args.shallow_eps,
        )

if __name__ == "__main__":
    main()
