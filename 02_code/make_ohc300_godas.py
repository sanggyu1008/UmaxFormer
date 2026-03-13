#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

# ============================================================
# Fixed GODAS setup requested by user
# - input/output directory: /mnt/d/project/01_ENSO/01_data/01_raw/godas
# - pottmp.1980-2025.nc -> ohc300.1980-2025.nc
# ============================================================

BASE_DIR = Path('/mnt/d/project/01_ENSO/01_data/01_raw/godas')
SRC_FILE = 'pottmp.198001-202512.nc'
OUT_FILE = 'ohc300.198001-202512.nc'
SRC_VAR = 'pottmp'
OUT_VAR = 'ohc300'

DEPTH_DIM_CANDIDATES = ['level', 'lev', 'depth', 'zlev', 'olevel', 'z', 'deptht', 'st_ocean']
TIME_DIM_CANDIDATES = ['time', 't']


def append_history(ds: xr.Dataset, message: str) -> None:
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    entry = f'{now}: {message}'
    old = ds.attrs.get('history')
    ds.attrs['history'] = f'{entry}\n{old}' if old else entry


def find_dim(da: xr.DataArray, candidates: list[str], kind: str) -> str:
    for name in candidates:
        if name in da.dims:
            return name
    for dim in da.dims:
        if dim in da.coords:
            units = str(da[dim].attrs.get('units', '')).lower()
            if kind == 'depth' and (units in ('m', 'meter', 'meters') or 'meter' in units):
                return dim
            if kind == 'time' and ('since' in units or 'days' in units or 'hours' in units):
                return dim
    raise ValueError(f'Cannot find {kind} dimension. dims={da.dims}')


def carry_aux_vars(ds: xr.Dataset, out: xr.Dataset, main_var: str) -> xr.Dataset:
    out_dims = set(out[main_var].dims)
    for name, var in ds.variables.items():
        if name in out.variables or name == main_var or name in ds.coords:
            continue
        if set(var.dims).issubset(out_dims):
            out[name] = var
    return out


def maybe_chunk(da: xr.DataArray, dim: str, size: int) -> xr.DataArray:
    if size <= 0:
        return da
    try:
        import dask  # noqa: F401
    except Exception:
        return da
    return da.chunk({dim: int(size)})


def infer_bounds_from_midpoints(z: xr.DataArray) -> xr.DataArray:
    zvals = z.values.astype('float64')
    if zvals.ndim != 1:
        raise ValueError('Depth coordinate is not 1D; cannot infer bounds robustly.')
    if zvals.size == 0:
        raise ValueError('Empty depth coordinate.')

    edges = np.empty(zvals.size + 1, dtype='float64')
    if zvals.size == 1:
        edges[0] = 0.0
        edges[1] = float(zvals[0])
    else:
        edges[1:-1] = 0.5 * (zvals[1:] + zvals[:-1])
        edges[-1] = zvals[-1] + (zvals[-1] - edges[-2])
        edges[0] = 0.0

    bounds = xr.DataArray(
        np.vstack([edges[:-1], edges[1:]]).T,
        dims=(z.dims[0], 'bnds'),
        coords={z.dims[0]: z, 'bnds': [0, 1]},
        name=f'{z.dims[0]}_bnds_midpoint',
    )
    bounds.attrs['comment'] = 'Inferred bounds from depth midpoints; top edge forced to 0 m.'
    return bounds


def overlap_thickness(bounds: xr.DataArray, zmin: float, zmax: float) -> xr.DataArray:
    lo = bounds.isel(bnds=0)
    hi = bounds.isel(bnds=1)
    lower = xr.where(lo < hi, lo, hi)
    upper = xr.where(lo < hi, hi, lo)

    ov = xr.apply_ufunc(np.minimum, upper, zmax) - xr.apply_ufunc(np.maximum, lower, zmin)
    ov = ov.clip(min=0.0)
    ov.name = 'dz_overlap'
    ov.attrs['units'] = 'm'
    return ov


def compute_ohc300(
    src_path: Path,
    out_path: Path,
    zmax: float,
    rho0: float,
    cp0: float,
    to_degC: bool,
    time_chunk: int,
    mask_shallow: bool,
    shallow_eps: float,
) -> None:
    ds = xr.open_dataset(src_path, decode_times=False)
    if SRC_VAR not in ds.variables:
        raise ValueError(f"Variable '{SRC_VAR}' not found in {src_path}")

    th = ds[SRC_VAR]
    depth_dim = find_dim(th, DEPTH_DIM_CANDIDATES, kind='depth')
    time_dim = find_dim(th, TIME_DIM_CANDIDATES, kind='time')

    th = th.sortby(depth_dim)
    z = th[depth_dim]
    th = maybe_chunk(th, time_dim, time_chunk)

    bounds = infer_bounds_from_midpoints(z)
    dz = overlap_thickness(bounds, 0.0, zmax)

    units = str(th.attrs.get('units', '')).strip().lower()
    th_use = th
    temp_scale = units or 'unknown'
    if to_degC and units in ('k', 'kelvin'):
        th_use = th - 273.15
        temp_scale = 'degC'

    if mask_shallow:
        wet_3d = th_use.isel({time_dim: 0}).notnull()
        coverage = dz.where(wet_3d).sum(dim=depth_dim)
        deep_enough = coverage >= (zmax - shallow_eps)
    else:
        deep_enough = None

    ocean_mask_time = th_use.notnull().any(depth_dim)
    th_filled = th_use.where(th_use.notnull(), 0.0)

    tint = (th_filled * dz).sum(dim=depth_dim)
    ohc = (tint * (rho0 * cp0)).where(ocean_mask_time)
    if mask_shallow:
        ohc = ohc.where(deep_enough)

    ohc = ohc.astype('float32')
    ohc = ohc.rename(OUT_VAR)
    out = ohc.to_dataset(name=OUT_VAR)
    out.attrs.update(ds.attrs)
    out = carry_aux_vars(ds, out, OUT_VAR)

    for key in ('_FillValue', 'missing_value', 'valid_min', 'valid_max', 'valid_range'):
        out[OUT_VAR].attrs.pop(key, None)

    out[OUT_VAR].attrs.update({
        'long_name': f'Ocean heat content integrated from 0 to {zmax:g} m',
        'units': 'J m-2',
        'source_variable': SRC_VAR,
        'source_file': src_path.name,
        'rho0': float(rho0),
        'cp0': float(cp0),
        'zmin': 0.0,
        'zmax': float(zmax),
        'dz_method': 'midpoint_inferred_bounds_top0m',
        'temperature_scale_for_integration': temp_scale,
        'shallow_masked': int(mask_shallow),
        'comment': (
            'Computed from GODAS pottmp by vertical integration using dz from midpoint-inferred bounds; '
            'overlap thickness with [0, zmax].'
        ),
    })

    msg = (
        f'Created {OUT_VAR} from {src_path.name} using {SRC_VAR} integrated over 0-{zmax:g} m '
        f'(rho0={rho0:g}, cp0={cp0:g}, temp_scale={temp_scale}, mask_shallow={int(mask_shallow)})'
    )
    append_history(out, msg)

    fill = np.float32(1.0e20)
    encoding = {
        OUT_VAR: {
            'zlib': True,
            'complevel': 4,
            'shuffle': True,
            'dtype': 'float32',
            '_FillValue': fill,
        }
    }
    out.to_netcdf(out_path, encoding=encoding)


def main() -> None:
    p = argparse.ArgumentParser(
        description='Create GODAS ohc300 in-place from pottmp.1980-2025.nc.'
    )
    p.add_argument(
        '--base-dir', type=Path, default=BASE_DIR,
        help=f'Directory containing GODAS merged files (default: {BASE_DIR})'
    )
    p.add_argument('--zmax', default=300.0, type=float)
    p.add_argument('--rho0', default=1025.0, type=float)
    p.add_argument('--cp0', default=3992.0, type=float)
    p.add_argument('--to-degC', action='store_true', help='Convert pottmp from Kelvin to degree Celsius before integration.')
    p.add_argument('--time-chunk', default=12, type=int)
    p.add_argument('--mask-shallow', action='store_true')
    p.add_argument('--shallow-eps', default=1e-3, type=float)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    base_dir = args.base_dir.expanduser().resolve()
    if not base_dir.exists():
        raise SystemExit(f'Base directory does not exist: {base_dir}')

    src_path = base_dir / SRC_FILE
    out_path = base_dir / OUT_FILE

    if not src_path.exists():
        raise SystemExit(f'Missing input file: {src_path}')
    if out_path.exists() and not args.overwrite:
        print(f'[skip] {out_path}')
        return

    print(f'[IN ] {src_path}')
    print(f'[OUT] {out_path}')

    compute_ohc300(
        src_path=src_path,
        out_path=out_path,
        zmax=args.zmax,
        rho0=args.rho0,
        cp0=args.cp0,
        to_degC=args.to_degC,
        time_chunk=args.time_chunk,
        mask_shallow=args.mask_shallow,
        shallow_eps=args.shallow_eps,
    )


if __name__ == '__main__':
    main()
