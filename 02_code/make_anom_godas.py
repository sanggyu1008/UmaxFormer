#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

# ============================================================
# GODAS monthly anomaly + 1x2 remap
# - input directory:  /mnt/d/project/01_ENSO/01_data/01_raw/godas
# - output directory: /mnt/d/project/01_ENSO/01_data/02_interim/godas
# - variables: mlotst, tos, vos, ohc300, sos, uos
# - operation: linear detrend + monthly climatology removal + remapbil(1x2)
#
# IMPORTANT FIX:
#   custom-generated GODAS surface files (tos/sos/uos/vos) may carry NaN-based
#   missing values, which can cause CDO detrend to produce all-NaN outputs.
#   This script rewrites the selected input variable with an explicit finite
#   _FillValue BEFORE running detrend.
#
# IMPORTANT FIX 2:
#   remapbil is now applied using the same 1x2 target grid file as ERA5.
#   Therefore, files named *.anom_1x2.nc are actually remapped 1x2 outputs.
# ============================================================

IN_DIR = Path('/mnt/d/project/01_ENSO/01_data/01_raw/godas')
OUT_DIR = Path('/mnt/d/project/01_ENSO/01_data/02_interim/godas')
GRIDFILE = Path('/mnt/d/project/01_ENSO/01_data/01_raw/grid_1x2_60S60N_120x180.txt')
TARGET_VARS = ['mlotst', 'tos', 'vos', 'ohc300', 'sos', 'uos']
PERIOD = '198001-202512'
FILL_VALUE = np.float32(1.0e20)
ZIPLVL = 'zip_4'


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def append_history(ds: xr.Dataset, message: str) -> None:
    entry = f'{utc_now()}: {message}'
    old = ds.attrs.get('history')
    ds.attrs['history'] = f'{entry}\n{old}' if old else entry


def run(cmd: list[str]) -> None:
    print('[cmd]', ' '.join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def ensure_cdo() -> None:
    if shutil.which('cdo') is None:
        raise SystemExit('cdo not found in PATH')


def ensure_gridfile(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f'Grid file does not exist: {path}')


def input_path(base_dir: Path, var: str) -> Path:
    return base_dir / f'{var}.{PERIOD}.nc'


def output_path(out_dir: Path, var: str) -> Path:
    return out_dir / f'{var}.{PERIOD}.anom_1x2.nc'


def rewrite_with_finite_fill(src: Path, dst: Path, var: str, add_anom_attrs: bool) -> None:
    with xr.open_dataset(src, decode_times=False) as ds:
        if var not in ds.variables:
            raise ValueError(f"Variable '{var}' not found in {src}")

        out = ds.copy()

        # NaN-based missing metadata 제거
        for key in ('_FillValue', 'missing_value', 'valid_min', 'valid_max', 'valid_range'):
            out[var].attrs.pop(key, None)

        if add_anom_attrs:
            out[var].attrs.update({
                'anomaly_definition': 'linear_detrend_then_remove_calendar_month_climatology',
                'climatology_period': PERIOD,
                'target_grid': '1x2_60S60N_120x180',
                'regridding': 'cdo remapbil',
            })
            append_history(
                out,
                f'Created anomaly file for {var}: detrend + ymonmean/ymonsub + remapbil({GRIDFILE})'
            )
        else:
            append_history(
                out,
                f'Rewrote input for {var} with explicit finite _FillValue before CDO processing'
            )

        dst.parent.mkdir(parents=True, exist_ok=True)
        encoding = {
            var: {
                'zlib': True,
                'complevel': 4,
                'shuffle': True,
                'dtype': 'float32',
                '_FillValue': FILL_VALUE,
            }
        }
        out.to_netcdf(dst, encoding=encoding)


def process_one(base_dir: Path, out_dir: Path, var: str, overwrite: bool, keep_intermediate: bool) -> None:
    infile = input_path(base_dir, var)
    outfile = output_path(out_dir, var)

    if not infile.exists():
        print(f'[missing] {infile}')
        return

    if outfile.exists() and not overwrite:
        print(f'[skip] {outfile}')
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f'godas_{var}_', dir=str(out_dir)) as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        sel = tmpdir / f'{var}.sel.nc'
        clean = tmpdir / f'{var}.clean.nc'
        detr = tmpdir / f'{var}.detr.nc'
        clim = tmpdir / f'{var}.clim.nc'
        anom = tmpdir / f'{var}.anom.tmp.nc'
        remap = tmpdir / f'{var}.anom_1x2.tmp.nc'
        final = tmpdir / f'{var}.anom_1x2.setmiss.nc'

        print(f'[IN ] {infile}')
        print(f'[OUT] {outfile}')

        # 0) target variable만 선택
        run(['cdo', '-L', '-O', '-f', 'nc4c', '-z', ZIPLVL, f'selname,{var}', str(infile), str(sel)])

        # 0.5) CDO detrend 전에 explicit finite _FillValue로 다시 저장
        rewrite_with_finite_fill(sel, clean, var, add_anom_attrs=False)

        # 1) linear detrend
        run(['cdo', '-L', '-O', '-f', 'nc4c', '-z', ZIPLVL, 'detrend', str(clean), str(detr)])

        # 2) monthly climatology
        run(['cdo', '-L', '-O', '-f', 'nc4c', '-z', ZIPLVL, 'ymonmean', str(detr), str(clim)])

        # 3) anomaly
        run(['cdo', '-L', '-O', '-f', 'nc4c', '-z', ZIPLVL, 'ymonsub', str(detr), str(clim), str(anom)])

        # 4) 1x2 remap (ERA5와 동일한 target grid)
        run(['cdo', '-L', '-O', '-f', 'nc4c', '-z', ZIPLVL, f'remapbil,{GRIDFILE}', str(anom), str(remap)])

        # 5) missing value를 finite 값으로 명시
        run(['cdo', '-L', '-O', '-f', 'nc4c', '-z', ZIPLVL, f'setmissval,{FILL_VALUE}', str(remap), str(final)])

        # 6) final write with finite _FillValue + attrs
        rewrite_with_finite_fill(final, outfile, var, add_anom_attrs=True)

        if keep_intermediate:
            keep_dir = out_dir / '_intermediate' / var
            keep_dir.mkdir(parents=True, exist_ok=True)
            for f in (sel, clean, detr, clim, anom, remap, final):
                shutil.copy2(f, keep_dir / f.name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Create monthly GODAS anomaly files (linear detrend + monthly climatology removal + 1x2 remap).'
    )
    p.add_argument('--base-dir', type=Path, default=IN_DIR)
    p.add_argument('--out-dir', type=Path, default=OUT_DIR)
    p.add_argument('--gridfile', type=Path, default=GRIDFILE)
    p.add_argument(
        '--vars', nargs='+', default=TARGET_VARS,
        help='Variables to process. Default: mlotst tos vos ohc300 sos uos'
    )
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--keep-intermediate', action='store_true')
    return p.parse_args()


def main() -> None:
    global GRIDFILE

    args = parse_args()
    ensure_cdo()

    base_dir = args.base_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    GRIDFILE = args.gridfile.expanduser().resolve()

    if not base_dir.exists():
        raise SystemExit(f'Base directory does not exist: {base_dir}')
    ensure_gridfile(GRIDFILE)

    vars_to_process = []
    for v in args.vars:
        if v not in TARGET_VARS:
            raise SystemExit(f'Unsupported variable: {v} ; supported = {TARGET_VARS}')
        vars_to_process.append(v)

    for var in vars_to_process:
        process_one(
            base_dir=base_dir,
            out_dir=out_dir,
            var=var,
            overwrite=args.overwrite,
            keep_intermediate=args.keep_intermediate,
        )


if __name__ == '__main__':
    main()