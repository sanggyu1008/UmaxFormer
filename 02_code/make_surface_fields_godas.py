#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr

# ============================================================
# Fixed GODAS setup requested by user
# - input/output directory: /mnt/d/project/01_ENSO/01_data/01_raw/godas
# - pottmp.1980-2025.nc -> tos.1980-2025.nc
# - salt.1980-2025.nc   -> sos.1980-2025.nc (x1000, kg/kg -> g/kg)
# - ucur.1980-2025.nc   -> uos.1980-2025.nc
# - vcur.1980-2025.nc   -> vos.1980-2025.nc
# ============================================================

BASE_DIR = Path('/mnt/d/project/01_ENSO/01_data/01_raw/godas')
DEPTH_DIM_CANDIDATES = ['level', 'lev', 'depth', 'zlev', 'olevel', 'z', 'deptht', 'st_ocean']
DROP_ATTRS_AFTER_DERIVATION = {
    'actual_range', 'valid_range', 'valid_min', 'valid_max', 'scale_factor', 'add_offset'
}
KGKG_UNITS = {
    'kg/kg', 'kg kg-1', 'kg kg^-1', 'kgkg-1', 'kgkg^-1', '1', 'fraction', 'mass fraction'
}

JOBMAP = {
    'pottmp': {
        'src_name': 'pottmp',
        'dst_name': 'tos',
        'src_file': 'pottmp.1980-2025.nc',
        'dst_file': 'tos.1980-2025.nc',
        'long_name': 'Surface potential temperature',
    },
    'salt': {
        'src_name': 'salt',
        'dst_name': 'sos',
        'src_file': 'salt.1980-2025.nc',
        'dst_file': 'sos.1980-2025.nc',
        'long_name': 'Surface salinity',
    },
    'ucur': {
        'src_name': 'ucur',
        'dst_name': 'uos',
        'src_file': 'ucur.1980-2025.nc',
        'dst_file': 'uos.1980-2025.nc',
        'long_name': 'Surface eastward current',
    },
    'vcur': {
        'src_name': 'vcur',
        'dst_name': 'vos',
        'src_file': 'vcur.1980-2025.nc',
        'dst_file': 'vos.1980-2025.nc',
        'long_name': 'Surface northward current',
    },
}


def normalize_units(units: str) -> str:
    return ' '.join(str(units).strip().lower().replace('**', '^').split())


def append_history(ds: xr.Dataset, message: str) -> None:
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    entry = f'{now}: {message}'
    old = ds.attrs.get('history')
    ds.attrs['history'] = f'{entry}\n{old}' if old else entry


def find_depth_dim(da: xr.DataArray) -> str:
    for name in DEPTH_DIM_CANDIDATES:
        if name in da.dims:
            return name
    for dim in da.dims:
        if dim in da.coords:
            units = str(da[dim].attrs.get('units', '')).lower()
            if units in ('m', 'meter', 'meters') or 'meter' in units:
                return dim
    raise ValueError(f'Cannot find depth dimension in {da.dims}')


def carry_aux_vars(ds: xr.Dataset, out: xr.Dataset, main_var: str) -> xr.Dataset:
    out_dims = set(out[main_var].dims)
    for name, var in ds.variables.items():
        if name in out.variables or name == main_var or name in ds.coords:
            continue
        if set(var.dims).issubset(out_dims):
            out[name] = var
    return out


def build_surface_field(src_path: Path, src_var: str, dst_var: str, long_name: str, convert_salt_to_gkg: bool) -> xr.Dataset:
    ds = xr.open_dataset(src_path, decode_times=False)
    if src_var not in ds.variables:
        raise ValueError(f"Variable '{src_var}' not found in {src_path}")

    da = ds[src_var]
    depth_dim = find_depth_dim(da)
    da = da.sortby(depth_dim)

    try:
        source_level_m = float(ds[depth_dim].isel({depth_dim: 0}).item())
    except Exception:
        source_level_m = None

    surf = da.isel({depth_dim: 0}, drop=True)
    notes = []

    if src_var == 'salt' and convert_salt_to_gkg:
        units = normalize_units(surf.attrs.get('units', ''))
        if units in KGKG_UNITS:
            surf = surf * 1000.0
            surf.attrs['units'] = 'g/kg'
            surf.attrs['salinity_unit_conversion'] = 'Converted from kg/kg to g/kg by multiplying by 1000.'
            notes.append('salinity kg/kg->g/kg')
        else:
            surf.attrs['salinity_unit_warning'] = (
                f"Source units '{surf.attrs.get('units', '')}' were not recognized as kg/kg. "
                'Values were copied without rescaling.'
            )
    
    surf = surf.rename(dst_var)
    attrs = dict(surf.attrs)
    for key in DROP_ATTRS_AFTER_DERIVATION:
        attrs.pop(key, None)
    attrs.update({
        'long_name': long_name,
        'source_variable': src_var,
        'selection': 'first_vertical_level',
        'comment': 'Derived from GODAS by selecting the first vertical level and renaming the variable.',
    })
    if source_level_m is not None:
        attrs['source_level_m'] = source_level_m
    if notes:
        attrs['postprocess'] = '; '.join(notes)
    surf.attrs = attrs

    out = surf.to_dataset(name=dst_var)
    out.attrs.update(ds.attrs)
    out = carry_aux_vars(ds, out, dst_var)

    msg = f'Created {dst_var} from {src_path.name} by selecting first vertical level'
    if notes:
        msg += f" and applying {', '.join(notes)}"
    append_history(out, msg)
    return out


def save_dataset(ds: xr.Dataset, out_path: Path) -> None:
    encoding = {
        list(ds.data_vars)[0]: {
            'zlib': True,
            'complevel': 4,
            'shuffle': True,
            'dtype': 'float32',
        }
    }
    ds.to_netcdf(out_path, encoding=encoding)


def main() -> None:
    p = argparse.ArgumentParser(
        description='Create GODAS tos/uos/vos/sos from merged 1980-2025 files in-place.'
    )
    p.add_argument(
        '--base-dir', type=Path, default=BASE_DIR,
        help=f'Directory containing GODAS merged files (default: {BASE_DIR})'
    )
    p.add_argument(
        '--vars', nargs='+', choices=list(JOBMAP.keys()),
        default=list(JOBMAP.keys()),
        help='Subset to process: pottmp salt ucur vcur'
    )
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing output files.')
    p.add_argument(
        '--keep-salt-kgkg', action='store_true',
        help='Do not convert salt from kg/kg to g/kg. Default is to convert and save sos in g/kg.'
    )
    args = p.parse_args()

    base_dir = args.base_dir.expanduser().resolve()
    if not base_dir.exists():
        raise SystemExit(f'Base directory does not exist: {base_dir}')

    for key in args.vars:
        job = JOBMAP[key]
        src_path = base_dir / job['src_file']
        out_path = base_dir / job['dst_file']

        if not src_path.exists():
            raise SystemExit(f'Missing input file: {src_path}')
        if out_path.exists() and not args.overwrite:
            print(f'[skip] {out_path}')
            continue

        print(f'[IN ] {src_path}')
        print(f'[OUT] {out_path}')

        out = build_surface_field(
            src_path=src_path,
            src_var=job['src_name'],
            dst_var=job['dst_name'],
            long_name=job['long_name'],
            convert_salt_to_gkg=not args.keep_salt_kgkg,
        )
        save_dataset(out, out_path)


if __name__ == '__main__':
    main()
