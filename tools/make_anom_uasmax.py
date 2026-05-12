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
# Unified uasmax anomaly builder for ERA5, 20CRv2 and CMIP6
#
# Processing order:
#   daily uas -> monthly max -> quadratic detrend -> ymonmean -> ymonsub -> remapbil(1x2)
#
# Notes
# - Keeps the original monthly-mean-based 'uas' products untouched.
# - Writes a separate variable/file named 'uasmax'.
# - ERA5 / 20CRv2: process a single daily file.
# - CMIP6: discovers daily files under <MODEL>/<MEMBER>/daily.
# ============================================================

DEFAULT_GRIDFILE = Path('/mnt/d/project/UmaxFormer/data/raw/grid_1x2_60S60N_120x180.txt')

ERA5_INFILE = Path('/mnt/d/project/UmaxFormer/data/raw/era5/uas.day.195801-202512.nc')
ERA5_OUTFILE = Path('/mnt/d/project/UmaxFormer/data/interim/era5/uasmax.mon.195801-202512.anom_1x2.nc')

CR20V2_INFILE = Path('/mnt/d/project/UmaxFormer/data/raw/20crv2/uwnd.10m.1871-2012.nc')
CR20V2_OUTFILE = Path('/mnt/d/project/UmaxFormer/data/interim/20crv2/uasmax.uwnd.10m.1871-2012.anom_1x2.nc')

CMIP6_IN_ROOT = Path('/mnt/d/project/UmaxFormer/data/raw/cmip6/ssp370')
CMIP6_OUT_ROOT = Path('/mnt/d/project/UmaxFormer/data/interim/cmip6/ssp370')

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




def quadratic_detrend_file(
    src: Path,
    dst: Path,
    var: str,
    fill_value: np.float32,
    *,
    spatial_chunk_size: int = 4,
) -> None:
    if spatial_chunk_size < 1:
        raise ValueError(f"spatial_chunk_size must be >= 1, got {spatial_chunk_size}")

    with xr.open_dataset(src, decode_times=False) as ds:
        if var not in ds.variables:
            raise ValueError(f"Variable '{var}' not found in {src}")

        da = ds[var]
        if 'time' not in da.dims:
            raise ValueError(f"Variable '{var}' in {src} has no time dimension: {da.dims}")

        dims = da.dims
        time_axis = dims.index('time')
        spatial_dims = [d for d in dims if d != 'time']
        ntime = da.sizes['time']
        x = np.linspace(-1.0, 1.0, ntime, dtype=np.float64)
        x2 = x * x

        out_attrs = dict(da.attrs)
        for key in ('_FillValue', 'missing_value', 'valid_min', 'valid_max', 'valid_range'):
            out_attrs.pop(key, None)

        def _detrend_matrix(y2d: np.ndarray) -> np.ndarray:
            mask = np.isfinite(y2d)
            y_filled = np.where(mask, y2d, 0.0)

            s0 = mask.sum(axis=0).astype(np.float64)
            s1 = np.sum(mask * x[:, None], axis=0)
            s2 = np.sum(mask * x2[:, None], axis=0)
            s3 = np.sum(mask * (x[:, None] * x2[:, None]), axis=0)
            s4 = np.sum(mask * (x2[:, None] * x2[:, None]), axis=0)
            y0 = np.sum(y_filled, axis=0)
            y1 = np.sum(y_filled * x[:, None], axis=0)
            y2 = np.sum(y_filled * x2[:, None], axis=0)

            ncol = y2d.shape[1]
            coeff = np.full((ncol, 3), np.nan, dtype=np.float64)
            valid = s0 >= 3
            if np.any(valid):
                idx = np.where(valid)[0]
                a = np.empty((idx.size, 3, 3), dtype=np.float64)
                b = np.empty((idx.size, 3), dtype=np.float64)

                a[:, 0, 0] = s0[valid]
                a[:, 0, 1] = s1[valid]
                a[:, 0, 2] = s2[valid]
                a[:, 1, 0] = s1[valid]
                a[:, 1, 1] = s2[valid]
                a[:, 1, 2] = s3[valid]
                a[:, 2, 0] = s2[valid]
                a[:, 2, 1] = s3[valid]
                a[:, 2, 2] = s4[valid]

                b[:, 0] = y0[valid]
                b[:, 1] = y1[valid]
                b[:, 2] = y2[valid]

                try:
                    coeff[idx] = np.linalg.solve(a, b[..., None])[..., 0]
                except np.linalg.LinAlgError:
                    for j, col in enumerate(idx):
                        try:
                            coeff[col] = np.linalg.solve(a[j], b[j])
                        except np.linalg.LinAlgError:
                            coeff[col] = np.linalg.lstsq(a[j], b[j], rcond=None)[0]

            fit = (
                coeff[:, 0][None, :]
                + x[:, None] * coeff[:, 1][None, :]
                + x2[:, None] * coeff[:, 2][None, :]
            )
            detr = y2d - fit
            detr[~mask] = np.nan
            return detr.astype(np.float32)

        dst.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=f'{var}_quadratic_', suffix='.mmap', dir=str(dst.parent), delete=False) as tmp:
            mmap_path = Path(tmp.name)

        try:
            arr_out = np.memmap(mmap_path, mode='w+', dtype='float32', shape=da.shape)

            if not spatial_dims:
                y = da.astype('float64').values.reshape(ntime, 1)
                arr_out[...] = _detrend_matrix(y).reshape(da.shape)
            else:
                lead_dim = spatial_dims[0]
                lead_axis = dims.index(lead_dim)
                nlead = da.sizes[lead_dim]
                print(f'[detrend] {var}: quadratic chunked detrend along {lead_dim} with chunk={spatial_chunk_size}', flush=True)
                for start in range(0, nlead, spatial_chunk_size):
                    end = min(start + spatial_chunk_size, nlead)
                    print(f'[detrend] {var}: {lead_dim}[{start}:{end}] / {nlead}', flush=True)
                    block = da.isel({lead_dim: slice(start, end)}).astype('float64').values
                    block_tf = np.moveaxis(block, time_axis, 0)
                    rest_shape = block_tf.shape[1:]
                    detr_tf = _detrend_matrix(block_tf.reshape(ntime, -1)).reshape((ntime, *rest_shape))
                    detr_block = np.moveaxis(detr_tf, 0, time_axis)
                    sl = [slice(None)] * arr_out.ndim
                    sl[lead_axis] = slice(start, end)
                    arr_out[tuple(sl)] = detr_block

            arr_out.flush()
            out = ds.copy(deep=False)
            out[var] = xr.DataArray(arr_out, dims=da.dims, coords=da.coords, attrs=out_attrs)
            append_history(out, f'Applied quadratic detrend (degree=2 polynomial) to {var} before anomaly calculation in spatial chunks of {spatial_chunk_size}.')
            encoding = {
                var: {
                    'zlib': True,
                    'complevel': 4,
                    'shuffle': True,
                    'dtype': 'float32',
                    '_FillValue': fill_value,
                }
            }
            out.to_netcdf(dst, encoding=encoding)
        finally:
            try:
                mmap_path.unlink(missing_ok=True)
            except Exception:
                pass

def rewrite_final_with_attrs(
    src: Path,
    dst: Path,
    outvar: str,
    source_var: str,
    history_message: str,
    extra_attrs: dict[str, str] | None = None,
    detrend_spatial_chunk: int = 4,
) -> None:
    with xr.open_dataset(src, decode_times=False) as ds:
        main_var = detect_main_var(ds, outvar)
        out = ds.copy(deep=False)

        if main_var != outvar:
            out = out.rename({main_var: outvar})

        for key in ('_FillValue', 'missing_value', 'valid_min', 'valid_max', 'valid_range'):
            out[outvar].attrs.pop(key, None)

        out[outvar].attrs.update({
            'long_name': 'Monthly maximum of daily eastward near-surface wind anomaly',
            'standard_name': 'eastward_wind',
            'anomaly_definition': 'monthly_max_from_daily_then_quadratic_detrend_then_remove_calendar_month_climatology',
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
    detrend_spatial_chunk: int = 4,
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
        quadratic_detrend_file(
            monmax,
            detr,
            CANONICAL_INVAR,
            FILL_VALUE,
            spatial_chunk_size=detrend_spatial_chunk,
        )
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
            'monmax + quadratic detrend + ymonmean/ymonsub + remapbil(1x2)'
        ),
        extra_attrs={
            'dataset_source': 'ERA5',
        },
        detrend_spatial_chunk=args.detrend_spatial_chunk,
    )


def process_20crv2(args: argparse.Namespace) -> None:
    infile = args.infile.expanduser().resolve()
    outfile = args.outfile.expanduser().resolve()
    gridfile = args.gridfile.expanduser().resolve()

    run_uasmax_pipeline(
        infile=infile,
        outfile=outfile,
        gridfile=gridfile,
        invar_candidates=['uas', 'uwnd', 'u10'],
        overwrite=args.overwrite,
        keep_intermediate=args.keep_intermediate,
        tmp_prefix='20crv2_uasmax_',
        intermediate_prefix='20crv2',
        history_message=(
            'Created 20CRv2 uasmax anomaly from daily {source_var}: '
            'monmax + quadratic detrend + ymonmean/ymonsub + remapbil(1x2)'
        ),
        extra_attrs={
            'dataset_source': '20CRv2',
        },
        detrend_spatial_chunk=args.detrend_spatial_chunk,
    )


def parse_cmip6_daily_filename(path: Path) -> tuple[str, str, str, str]:
    daily_dir = path.parent
    member_dir = daily_dir.parent
    model_dir = member_dir.parent

    if daily_dir.name != 'daily':
        raise ValueError(f'Expected parent directory named daily, got: {daily_dir}')
    model = model_dir.name
    member = member_dir.name

    prefix = f'uas_day_{model}_'
    if not path.name.startswith(prefix):
        raise ValueError(
            f'CMIP6 daily filename does not match model directory name: '
            f'file={path.name}, model_dir={model}'
        )

    rest = path.name[len(prefix):]
    m = re.match(
        r'^(?P<exp>.+)_(?P<member>r\d+i\d+p\d+f\d+)_(?P<grid>[^_]+)_time\.nc$',
        rest,
    )
    if not m:
        raise ValueError(f'Unexpected CMIP6 daily filename tail: {path.name}')

    member_in_name = m.group('member')
    grid = m.group('grid')
    exp = m.group('exp')

    if member_in_name != member:
        raise ValueError(
            f'CMIP6 daily filename member mismatch: file={path.name}, '
            f'member_dir={member}, member_in_name={member_in_name}'
        )

    return model, exp, member, grid


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
    detrend_spatial_chunk: int,
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
            'monmax + quadratic detrend + ymonmean/ymonsub + remapbil(1x2)'
        ),
        extra_attrs={
            'dataset_source': 'CMIP6',
            'cmip6_model': model,
            'cmip6_experiment': exp,
            'cmip6_member': member,
            'source_grid_label': grid,
        },
        detrend_spatial_chunk=detrend_spatial_chunk,
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
            detrend_spatial_chunk=args.detrend_spatial_chunk,
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            'Create uasmax monthly-max anomaly files from daily uas for ERA5, 20CRv2 or CMIP6 '
            '(monmax + quadratic detrend + ymonmean/ymonsub + 1x2 remap).'
        )
    )
    sub = p.add_subparsers(dest='source', required=True)

    p_era5 = sub.add_parser('era5', help='Process ERA5 daily uas file')
    p_era5.add_argument('--infile', type=Path, default=ERA5_INFILE)
    p_era5.add_argument('--outfile', type=Path, default=ERA5_OUTFILE)
    p_era5.add_argument('--gridfile', type=Path, default=DEFAULT_GRIDFILE)
    p_era5.add_argument('--overwrite', action='store_true')
    p_era5.add_argument('--keep-intermediate', action='store_true')
    p_era5.add_argument('--detrend-spatial-chunk', type=int, default=4, help='Number of indices to process at once along the first non-time dimension during quadratic detrend.')
    p_era5.set_defaults(func=process_era5)

    p_20cr = sub.add_parser('20crv2', help='Process 20CRv2 daily uas file')
    p_20cr.add_argument('--infile', type=Path, default=CR20V2_INFILE)
    p_20cr.add_argument('--outfile', type=Path, default=CR20V2_OUTFILE)
    p_20cr.add_argument('--gridfile', type=Path, default=DEFAULT_GRIDFILE)
    p_20cr.add_argument('--overwrite', action='store_true')
    p_20cr.add_argument('--keep-intermediate', action='store_true')
    p_20cr.add_argument('--detrend-spatial-chunk', type=int, default=4, help='Number of indices to process at once along the first non-time dimension during quadratic detrend.')
    p_20cr.set_defaults(func=process_20crv2)

    p_cmip6 = sub.add_parser('cmip6', help='Process CMIP6 daily uas files')
    p_cmip6.add_argument('--in-root', type=Path, default=CMIP6_IN_ROOT)
    p_cmip6.add_argument('--out-root', type=Path, default=CMIP6_OUT_ROOT)
    p_cmip6.add_argument('--gridfile', type=Path, default=DEFAULT_GRIDFILE)
    p_cmip6.add_argument('--overwrite', action='store_true')
    p_cmip6.add_argument('--keep-intermediate', action='store_true')
    p_cmip6.add_argument('--detrend-spatial-chunk', type=int, default=4, help='Number of indices to process at once along the first non-time dimension during quadratic detrend.')
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
