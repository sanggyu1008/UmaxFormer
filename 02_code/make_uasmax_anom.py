#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

# ============================================================
# Unified uasmax anomaly builder for ERA5 and CMIP6
#
# Processing order:
#   daily uas -> monthly max -> detrend -> ymonmean -> ymonsub -> remapbil(1x2)
#
# Notes
# - Keeps the original monthly-mean-based 'uas' products untouched.
# - Writes a separate variable/file named 'uasmax'.
# - ERA5: processes a single daily file.
# - CMIP6: discovers daily files under <MODEL>/<MEMBER>/daily.
# ============================================================

DEFAULT_GRIDFILE = Path('/mnt/d/project/01_ENSO/01_data/01_raw/grid_1x2_60S60N_120x180.txt')

ERA5_INFILE = Path('/mnt/d/project/01_ENSO/01_data/01_raw/era5/uas.day.195801-202512.nc')
ERA5_OUTFILE = Path('/mnt/d/project/01_ENSO/01_data/02_interim/era5/uasmax.mon.195801-202512.anom_1x2.nc')

CMIP6_IN_ROOT = Path('/mnt/d/project/01_ENSO/01_data/01_raw/cmip6/ssp370')
CMIP6_OUT_ROOT = Path('/mnt/d/project/01_ENSO/01_data/02_interim/cmip6/ssp370')

OUTVAR = 'uasmax'
CANONICAL_INVAR = 'uas'
FILL_VALUE = np.float32(1.0e20)
ZIPLVL = 'zip_4'
TIME_CANDIDATES = ('time', 'valid_time', 'time_counter', 't')


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def run(cmd: list[str]) -> None:
    print('[cmd]', ' '.join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def ensure_cdo() -> None:
    if shutil.which('cdo') is None:
        raise SystemExit('cdo not found in PATH')


def ensure_python_deps() -> None:
    for name in ('numpy', 'xarray'):
        __import__(name)


def ensure_exists(path: Path, kind: str = 'path') -> None:
    if not path.exists():
        raise SystemExit(f'{kind} does not exist: {path}')


def append_history(ds: xr.Dataset, message: str) -> None:
    entry = f'{utc_now()}: {message}'
    old = ds.attrs.get('history')
    ds.attrs['history'] = f'{entry}\n{old}' if old else entry


def detect_main_var(ds: xr.Dataset, expected: str) -> str:
    if expected in ds.data_vars:
        return expected

    candidates: list[tuple[str, int]] = []
    for name, da in ds.data_vars.items():
        if any(tn in da.dims for tn in TIME_CANDIDATES):
            size = int(np.prod([da.sizes[d] for d in da.dims], dtype=np.int64))
            candidates.append((name, size))

    if not candidates:
        if not ds.data_vars:
            raise ValueError('No data variables found')
        return list(ds.data_vars)[0]

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def cdo_show_names(infile: Path) -> list[str]:
    result = subprocess.run(
        ['cdo', '-s', 'showname', str(infile)],
        check=True,
        capture_output=True,
        text=True,
    )
    return [name for name in result.stdout.split() if name.strip()]


def resolve_var(infile: Path, candidates: list[str]) -> str:
    names = cdo_show_names(infile)
    for cand in candidates:
        if cand in names:
            return cand
    if names:
        return names[0]
    raise ValueError(f'No variable names found in {infile}')


def rewrite_final_with_attrs(
    src: Path,
    dst: Path,
    outvar: str,
    source_var: str,
    history_message: str,
    extra_attrs: dict[str, str] | None = None,
) -> None:
    with xr.open_dataset(src, decode_times=False) as ds:
        main_var = detect_main_var(ds, outvar)
        out = ds.load()

        if main_var != outvar:
            out = out.rename({main_var: outvar})

        for key in ('_FillValue', 'missing_value', 'valid_min', 'valid_max', 'valid_range'):
            out[outvar].attrs.pop(key, None)

        out[outvar].attrs.update({
            'long_name': 'Monthly maximum of daily eastward near-surface wind anomaly',
            'standard_name': 'eastward_wind',
            'anomaly_definition': 'monthly_max_from_daily_then_linear_detrend_then_remove_calendar_month_climatology',
            'source_variable': source_var,
            'source_frequency': 'daily',
            'temporal_aggregation': 'calendar_month_max',
            'target_grid': '1x2_60S60N_120x180',
            'regridding': 'cdo remapbil',
        })
        if extra_attrs:
            out[outvar].attrs.update(extra_attrs)

        append_history(out, history_message)

        dst.parent.mkdir(parents=True, exist_ok=True)
        encoding = {
            outvar: {
                'zlib': True,
                'complevel': 4,
                'shuffle': True,
                'dtype': 'float32',
                '_FillValue': FILL_VALUE,
            }
        }
        out.to_netcdf(dst, encoding=encoding)


def run_uasmax_pipeline(
    infile: Path,
    outfile: Path,
    gridfile: Path,
    *,
    invar_candidates: list[str],
    overwrite: bool,
    keep_intermediate: bool,
    tmp_prefix: str,
    intermediate_prefix: str,
    history_message: str,
    extra_attrs: dict[str, str] | None = None,
) -> None:
    if outfile.exists() and not overwrite:
        print(f'[skip] {outfile}')
        return

    ensure_exists(infile, 'input file')
    ensure_exists(gridfile, 'grid file')
    outfile.parent.mkdir(parents=True, exist_ok=True)

    invar = resolve_var(infile, invar_candidates)

    with tempfile.TemporaryDirectory(prefix=tmp_prefix, dir=str(outfile.parent)) as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        sel = tmpdir / 'uas.sel.nc'
        ren = tmpdir / 'uas.ren.nc'
        clean = tmpdir / 'uas.clean.nc'
        monmax = tmpdir / 'uas.monmax.nc'
        detr = tmpdir / 'uas.detr.nc'
        clim = tmpdir / 'uas.clim.nc'
        anom = tmpdir / 'uas.anom.nc'
        remap = tmpdir / 'uas.anom_1x2.tmp.nc'
        final = tmpdir / 'uas.anom_1x2.setmiss.nc'

        print(f'[IN ] {infile}')
        print(f'[OUT] {outfile}')
        print(f'[VAR] {invar} -> {OUTVAR}')

        run(['cdo', '-L', '-O', '-f', 'nc4c', '-z', ZIPLVL, f'selname,{invar}', str(infile), str(sel)])

        work_in = sel
        if invar != CANONICAL_INVAR:
            run(['cdo', '-L', '-O', '-f', 'nc4c', '-z', ZIPLVL, f'chname,{invar},{CANONICAL_INVAR}', str(sel), str(ren)])
            work_in = ren

        run(['cdo', '-L', '-O', '-f', 'nc4c', '-z', ZIPLVL, f'setmissval,{FILL_VALUE}', str(work_in), str(clean)])
        run(['cdo', '-L', '-O', '-f', 'nc4c', '-z', ZIPLVL, 'monmax', str(clean), str(monmax)])
        run(['cdo', '-L', '-O', '-f', 'nc4c', '-z', ZIPLVL, 'detrend', str(monmax), str(detr)])
        run(['cdo', '-L', '-O', '-f', 'nc4c', '-z', ZIPLVL, 'ymonmean', str(detr), str(clim)])
        run(['cdo', '-L', '-O', '-f', 'nc4c', '-z', ZIPLVL, 'ymonsub', str(detr), str(clim), str(anom)])
        run(['cdo', '-L', '-O', '-f', 'nc4c', '-z', ZIPLVL, f'remapbil,{gridfile}', str(anom), str(remap)])
        run(['cdo', '-L', '-O', '-f', 'nc4c', '-z', ZIPLVL, f'setmissval,{FILL_VALUE}', str(remap), str(final)])

        merged_attrs = {'source_file': infile.name}
        if extra_attrs:
            merged_attrs.update(extra_attrs)

        rewrite_final_with_attrs(
            final,
            outfile,
            OUTVAR,
            source_var=invar,
            history_message=history_message.format(source_var=invar),
            extra_attrs=merged_attrs,
        )

        if keep_intermediate:
            keep_dir = outfile.parent / '_intermediate_uasmax'
            keep_dir.mkdir(parents=True, exist_ok=True)
            for f in (sel, ren, clean, monmax, detr, clim, anom, remap, final):
                if f.exists():
                    target_name = f'{intermediate_prefix}_{f.name}' if intermediate_prefix else f.name
                    shutil.copy2(f, keep_dir / target_name)


def process_era5(args: argparse.Namespace) -> None:
    infile = args.infile.expanduser().resolve()
    outfile = args.outfile.expanduser().resolve()
    gridfile = args.gridfile.expanduser().resolve()

    run_uasmax_pipeline(
        infile=infile,
        outfile=outfile,
        gridfile=gridfile,
        invar_candidates=['uas', 'u10'],
        overwrite=args.overwrite,
        keep_intermediate=args.keep_intermediate,
        tmp_prefix='era5_uasmax_',
        intermediate_prefix='era5',
        history_message=(
            'Created ERA5 uasmax anomaly from daily {source_var}: '
            'monmax + detrend + ymonmean/ymonsub + remapbil(1x2)'
        ),
        extra_attrs={
            'dataset_source': 'ERA5',
        },
    )


def parse_cmip6_daily_filename(path: Path) -> tuple[str, str, str, str]:
    m = re.match(
        r'^uas_day_(?P<model>.+?)_(?P<exp>[^_]+)_(?P<member>r\d+i\d+p\d+f\d+)_(?P<grid>[^_]+)_time\.nc$',
        path.name,
    )
    if not m:
        raise ValueError(f'Unexpected CMIP6 daily filename: {path.name}')
    return m.group('model'), m.group('exp'), m.group('member'), m.group('grid')


def cmip6_output_path(out_root: Path, model: str, exp: str, member: str, grid: str) -> Path:
    monthly_dir = out_root / model / member / 'monthly'
    return monthly_dir / f'{OUTVAR}_Amon_{model}_{exp}_{member}_{grid}_time.anom_1x2.nc'


def discover_cmip6_inputs(base_root: Path) -> list[Path]:
    files: list[Path] = []
    for model_dir in sorted(p for p in base_root.iterdir() if p.is_dir()):
        for member_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            daily_dir = member_dir / 'daily'
            if not daily_dir.is_dir():
                continue
            files.extend(sorted(daily_dir.glob('uas_day_*_time.nc')))
    return files


def process_one_cmip6(
    infile: Path,
    out_root: Path,
    gridfile: Path,
    *,
    overwrite: bool,
    keep_intermediate: bool,
) -> None:
    model, exp, member, grid = parse_cmip6_daily_filename(infile)
    outfile = cmip6_output_path(out_root, model, exp, member, grid)

    run_uasmax_pipeline(
        infile=infile,
        outfile=outfile,
        gridfile=gridfile,
        invar_candidates=['uas'],
        overwrite=overwrite,
        keep_intermediate=keep_intermediate,
        tmp_prefix=f'cmip6_uasmax_{model}_{member}_',
        intermediate_prefix=f'{model}_{member}',
        history_message=(
            'Created CMIP6 uasmax anomaly from daily {source_var}: '
            'monmax + detrend + ymonmean/ymonsub + remapbil(1x2)'
        ),
        extra_attrs={
            'dataset_source': 'CMIP6',
            'cmip6_model': model,
            'cmip6_experiment': exp,
            'cmip6_member': member,
            'source_grid_label': grid,
        },
    )


def process_cmip6(args: argparse.Namespace) -> None:
    in_root = args.in_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    gridfile = args.gridfile.expanduser().resolve()

    ensure_exists(in_root, 'input root')

    files = discover_cmip6_inputs(in_root)
    if args.model:
        model_set = set(args.model)
        files = [p for p in files if p.parents[2].name in model_set]
    if args.member:
        member_set = set(args.member)
        files = [p for p in files if p.parents[1].name in member_set]

    if not files:
        raise SystemExit(f'No daily uas files found under {in_root}')

    for infile in files:
        process_one_cmip6(
            infile=infile,
            out_root=out_root,
            gridfile=gridfile,
            overwrite=args.overwrite,
            keep_intermediate=args.keep_intermediate,
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            'Create uasmax monthly-max anomaly files from daily uas for ERA5 or CMIP6 '
            '(monmax + detrend + ymonmean/ymonsub + 1x2 remap).'
        )
    )
    sub = p.add_subparsers(dest='source', required=True)

    p_era5 = sub.add_parser('era5', help='Process ERA5 daily uas file')
    p_era5.add_argument('--infile', type=Path, default=ERA5_INFILE)
    p_era5.add_argument('--outfile', type=Path, default=ERA5_OUTFILE)
    p_era5.add_argument('--gridfile', type=Path, default=DEFAULT_GRIDFILE)
    p_era5.add_argument('--overwrite', action='store_true')
    p_era5.add_argument('--keep-intermediate', action='store_true')
    p_era5.set_defaults(func=process_era5)

    p_cmip6 = sub.add_parser('cmip6', help='Process CMIP6 daily uas files')
    p_cmip6.add_argument('--in-root', type=Path, default=CMIP6_IN_ROOT)
    p_cmip6.add_argument('--out-root', type=Path, default=CMIP6_OUT_ROOT)
    p_cmip6.add_argument('--gridfile', type=Path, default=DEFAULT_GRIDFILE)
    p_cmip6.add_argument('--overwrite', action='store_true')
    p_cmip6.add_argument('--keep-intermediate', action='store_true')
    p_cmip6.add_argument('--model', nargs='*', default=None, help='Optional model names to process')
    p_cmip6.add_argument('--member', nargs='*', default=None, help='Optional member names to process')
    p_cmip6.set_defaults(func=process_cmip6)

    return p


def main() -> None:
    ensure_cdo()
    ensure_python_deps()

    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
