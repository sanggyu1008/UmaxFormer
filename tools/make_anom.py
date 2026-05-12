#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr

from preprocess_config import ORAS5_RANGE_TAG


COMMON_GRIDFILE = Path("/mnt/d/project/UmaxFormer/data/raw/grid_1x2_60S60N_120x180.txt")
COMMON_MISSVAL = np.float32(1.0e20)


@dataclass(frozen=True)
class SourceDefaults:
    in_dir: Path
    out_dir: Path
    target_vars: tuple[str, ...]
    use_finite_fill_before_detrend: bool
    apply_setmissval_after_remap: bool


SOURCE_DEFAULTS: dict[str, SourceDefaults] = {
    "cmip6": SourceDefaults(
        in_dir=Path("/mnt/d/project/UmaxFormer/data/raw/cmip6/ssp370"),
        out_dir=Path("/mnt/d/project/UmaxFormer/data/interim/cmip6/ssp370"),
        target_vars=("mlotst", "ohc300", "psl", "sos", "tos", "uas", "uos", "vas", "vos"),
        use_finite_fill_before_detrend=False,
        apply_setmissval_after_remap=False,
    ),
    "era5": SourceDefaults(
        in_dir=Path("/mnt/d/project/UmaxFormer/data/raw/era5"),
        out_dir=Path("/mnt/d/project/UmaxFormer/data/interim/era5"),
        target_vars=("psl", "uas", "vas"),
        use_finite_fill_before_detrend=True,
        apply_setmissval_after_remap=True,
    ),
    "oras5": SourceDefaults(
        in_dir=Path("/mnt/d/project/UmaxFormer/data/raw/oras5"),
        out_dir=Path("/mnt/d/project/UmaxFormer/data/interim/oras5"),
        target_vars=("mlotst", "ohc300", "sos", "tos", "uos", "vos"),
        use_finite_fill_before_detrend=False,
        apply_setmissval_after_remap=True,
    ),
    "godas": SourceDefaults(
        in_dir=Path("/mnt/d/project/UmaxFormer/data/raw/godas"),
        out_dir=Path("/mnt/d/project/UmaxFormer/data/interim/godas"),
        target_vars=("mlotst", "tos", "vos", "ohc300", "sos", "uos"),
        use_finite_fill_before_detrend=True,
        apply_setmissval_after_remap=True,
    ),
    "soda": SourceDefaults(
        in_dir=Path("/mnt/d/project/UmaxFormer/data/raw/soda"),
        out_dir=Path("/mnt/d/project/UmaxFormer/data/interim/soda"),
        target_vars=("mlotst", "ohc300", "sos", "tos", "uos", "vos"),
        use_finite_fill_before_detrend=True,
        apply_setmissval_after_remap=True,
    ),
    "20crv2": SourceDefaults(
        in_dir=Path("/mnt/d/project/UmaxFormer/data/raw/20crv2"),
        out_dir=Path("/mnt/d/project/UmaxFormer/data/interim/20crv2"),
        target_vars=("psl", "uas", "vas"),
        use_finite_fill_before_detrend=True,
        apply_setmissval_after_remap=True,
    ),
}


SOURCE_VAR_CANDIDATES: dict[str, dict[str, tuple[str, ...]]] = {
    "cmip6": {
        "mlotst": ("mlotst",),
        "ohc300": ("ohc300",),
        "psl": ("psl",),
        "sos": ("sos",),
        "tos": ("tos",),
        "uas": ("uas",),
        "uos": ("uos",),
        "vas": ("vas",),
        "vos": ("vos",),
    },
    "era5": {
        "psl": ("psl", "msl"),
        "uas": ("uas", "u10"),
        "vas": ("vas", "v10"),
    },
    "oras5": {
        "mlotst": ("mlotst", "somxl010"),
        "ohc300": ("ohc300", "sohtc300"),
        "sos": ("sos",),
        "tos": ("tos",),
        "uos": ("uos", "vozocrtx"),
        "vos": ("vos", "vomecrty"),
    },
    "godas": {
        "mlotst": ("mlotst",),
        "tos": ("tos",),
        "vos": ("vos",),
        "ohc300": ("ohc300",),
        "sos": ("sos",),
        "uos": ("uos",),
    },
    "soda": {
        "mlotst": ("mlotst",),
        "ohc300": ("ohc300",),
        "sos": ("sos",),
        "tos": ("tos",),
        "uos": ("uos", "uo"),
        "vos": ("vos", "vo"),
    },
    "20crv2": {
        "psl": ("psl", "prmsl", "msl"),
        "uas": ("uas", "uwnd", "u10"),
        "vas": ("vas", "vwnd", "v10"),
    },
}


ERA5_FILES = {
    "psl": "psl.mon.195801-202512.nc",
    "uas": "uas.mon.195801-202512.nc",
    "vas": "vas.mon.195801-202512.nc",
}

def build_oras5_file_map(tag: str) -> dict[str, str]:
    return {
        "mlotst": f"mlotst.{tag}.nc",
        "ohc300": f"ohc300.{tag}.nc",
        "sos": f"sos.{tag}.nc",
        "tos": f"tos.{tag}.nc",
        "uos": f"uos.{tag}.nc",
        "vos": f"vos.{tag}.nc",
    }


ORAS5_FILES = build_oras5_file_map(ORAS5_RANGE_TAG)

SODA_FILES = {
    "mlotst": "SODA_mlotst.nc",
    "ohc300": "SODA_ohc300.nc",
    "sos": "SODA_sos.nc",
    "tos": "SODA_tos.nc",
    "uos": "SODA_uo.nc",
    "vos": "SODA_vo.nc",
}

CR20V2_FILES = {
    "psl": "prmsl.mon.mean.nc",
    "uas": "uwnd.10m.mon.mean.nc",
    "vas": "vwnd.10m.mon.mean.nc",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str]) -> None:
    log("[cmd] " + " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def ensure_requirements(gridfile: Path) -> None:
    if shutil.which("cdo") is None:
        raise SystemExit("[error] cdo not found in PATH")
    if not gridfile.exists():
        raise SystemExit(f"[error] grid file not found: {gridfile}")


def append_history(ds: xr.Dataset, message: str) -> None:
    entry = f"{utc_now()}: {message}"
    old = ds.attrs.get("history")
    ds.attrs["history"] = f"{entry}\n{old}" if old else entry


LON_COORD_CANDIDATES = ("lon", "longitude", "nav_lon")
LAT_COORD_CANDIDATES = ("lat", "latitude", "nav_lat")
OCEAN_VARS = {"mlotst", "ohc300", "sos", "tos", "uos", "vos"}


def find_existing_name(ds: xr.Dataset, candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if name in ds.coords or name in ds.variables:
            return name
    return None


def find_x_dim_for_longitude(var: xr.DataArray, lon: xr.DataArray) -> str | None:
    if lon.ndim == 1:
        return lon.dims[0]

    cand_dims = [d for d in lon.dims if d in var.dims]
    if not cand_dims:
        return None
    if len(cand_dims) == 1:
        return cand_dims[0]

    lon_vals = np.asarray(lon.values, dtype=np.float64)
    lon_vals = np.where(np.isfinite(lon_vals) & (lon_vals < 0.0), lon_vals + 360.0, lon_vals)

    best_dim = None
    best_score = -1.0
    for dim in cand_dims:
        axis = lon.get_axis_num(dim)
        dlon = np.diff(lon_vals, axis=axis)
        wrap_count = np.sum(np.isfinite(dlon) & ((dlon < -180.0) | (dlon > 180.0)))
        size_bonus = lon.sizes[dim] * 1e-6
        score = float(wrap_count) + size_bonus
        if score > best_score:
            best_score = score
            best_dim = dim

    if best_dim is not None:
        return best_dim
    return max(cand_dims, key=lambda d: lon.sizes[d])


def unwrap_longitude_for_remap(lon: xr.DataArray, x_dim: str) -> tuple[xr.DataArray, bool]:
    lon_vals = np.asarray(lon.values, dtype=np.float64)
    lon_vals = np.where(np.isfinite(lon_vals) & (lon_vals < 0.0), lon_vals + 360.0, lon_vals)

    axis = lon.get_axis_num(x_dim)
    dlon = np.diff(lon_vals, axis=axis)
    has_internal_wrap = bool(np.nanmin(dlon) < -180.0)
    if not has_internal_wrap:
        return lon, False

    lon_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(lon_vals), axis=axis))
    out = xr.DataArray(lon_unwrapped, dims=lon.dims, coords=lon.coords, attrs=dict(lon.attrs), name=lon.name)
    out.attrs.setdefault("standard_name", "longitude")
    out.attrs.setdefault("units", "degrees_east")
    return out, True


def repair_cyclic_longitude(ds: xr.Dataset, var_name: str) -> tuple[xr.Dataset, bool]:
    if var_name not in ds.variables:
        return ds, False

    lon_name = find_existing_name(ds, LON_COORD_CANDIDATES)
    lat_name = find_existing_name(ds, LAT_COORD_CANDIDATES)
    if lon_name is None or lat_name is None:
        return ds, False

    lon = ds[lon_name]
    x_dim = find_x_dim_for_longitude(ds[var_name], lon)
    if x_dim is None:
        return ds, False

    lon_fixed, changed = unwrap_longitude_for_remap(lon, x_dim)
    if not changed:
        return ds, False

    out = ds.copy()
    out = out.assign_coords({lon_name: lon_fixed})

    lat = out[lat_name]
    lat.attrs = dict(lat.attrs)
    lat.attrs.setdefault("standard_name", "latitude")
    lat.attrs.setdefault("units", "degrees_north")
    out = out.assign_coords({lat_name: lat})

    append_history(out, f"Unwrapped source longitudes along {x_dim} before remap to avoid an internal seam.")
    return out, True


SPATIAL_BOUNDS_NAMES = {
    "lon_bnds", "lat_bnds",
    "longitude_bnds", "latitude_bnds",
    "bounds_lon", "bounds_lat",
    "vertices_longitude", "vertices_latitude",
}


def strip_spatial_bounds_for_remap(ds: xr.Dataset, var_name: str) -> tuple[xr.Dataset, bool]:
    out = ds.copy()

    lon_name = find_existing_name(out, LON_COORD_CANDIDATES)
    lat_name = find_existing_name(out, LAT_COORD_CANDIDATES)

    drop = set()

    for cname in (lon_name, lat_name):
        if cname and cname in out.variables:
            attrs = dict(out[cname].attrs)
            bname = attrs.pop("bounds", None)
            out[cname].attrs = attrs
            if bname and bname in out.variables and bname != "time_bnds":
                drop.add(bname)

    for name in list(out.variables):
        lname = name.lower()
        if name in {var_name, lon_name, lat_name, "time", "time_bnds"}:
            continue
        if lname in SPATIAL_BOUNDS_NAMES or "vertex" in lname:
            drop.add(name)

    if drop:
        out = out.drop_vars(sorted(drop), errors="ignore")
        append_history(out, "Dropped spatial bounds/vertex metadata before remap.")
        return out, True

    return out, False


def rewrite_with_finite_fill(
    src: Path,
    dst: Path,
    var: str,
    fill_value: np.float32,
    *,
    add_anom_attrs: bool,
    source: str,
    gridfile: Path,
) -> None:
    with xr.open_dataset(src, decode_times=False) as ds:
        if var not in ds.variables:
            raise ValueError(f"Variable '{var}' not found in {src}")

        out = ds.copy()

        for key in ("_FillValue", "missing_value", "valid_min", "valid_max", "valid_range"):
            out[var].attrs.pop(key, None)

        if add_anom_attrs:
            out[var].attrs.update({
                "anomaly_definition": "quadratic_detrend_then_remove_calendar_month_climatology",
                "target_grid": gridfile.name,
                "regridding": "cdo remapbil",
                "source": source,
            })
            append_history(
                out,
                f"Created anomaly file for {var}: quadratic detrend + ymonmean/ymonsub + remapbil({gridfile})",
            )
        else:
            append_history(
                out,
                f"Rewrote input for {var} with explicit finite _FillValue before CDO processing",
            )

        dst.parent.mkdir(parents=True, exist_ok=True)
        encoding = {
            var: {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "dtype": "float32",
                "_FillValue": fill_value,
            }
        }
        out.to_netcdf(dst, encoding=encoding)


def write_single_var_dataset(ds: xr.Dataset, dst: Path, var: str, fill_value: np.float32) -> None:
    out = ds.copy()
    for key in ("_FillValue", "missing_value", "valid_min", "valid_max", "valid_range"):
        if var in out.variables:
            out[var].attrs.pop(key, None)

    dst.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        var: {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "dtype": "float32",
            "_FillValue": fill_value,
        }
    }
    out.to_netcdf(dst, encoding=encoding)


def quadratic_detrend_file(
    src: Path,
    dst: Path,
    var: str,
    fill_value: np.float32,
    *,
    spatial_chunk_size: int = 16,
) -> None:
    if spatial_chunk_size < 1:
        raise ValueError(f"spatial_chunk_size must be >= 1, got {spatial_chunk_size}")

    time_candidates = ("time", "time_counter", "valid_time", "t")

    with xr.open_dataset(src, decode_times=False) as ds:
        if var not in ds.variables:
            raise ValueError(f"Variable '{var}' not found in {src}")

        da = ds[var]

        time_name = next((name for name in time_candidates if name in da.dims), None)
        if time_name is None:
            raise ValueError(
                f"Variable '{var}' in {src} has no recognized time dimension: {da.dims}"
            )

        if time_name != "time":
            ds = ds.rename({time_name: "time"})
            da = ds[var]

        dims = da.dims
        time_axis = dims.index("time")
        spatial_dims = [d for d in dims if d != "time"]
        ntime = da.sizes["time"]
        x = np.linspace(-1.0, 1.0, ntime, dtype=np.float64)
        x2 = x * x

        out_attrs = dict(da.attrs)
        for key in ("_FillValue", "missing_value", "valid_min", "valid_max", "valid_range"):
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
        with tempfile.NamedTemporaryFile(prefix=f"{var}_quadratic_", suffix=".mmap", dir=str(dst.parent), delete=False) as tmp:
            mmap_path = Path(tmp.name)

        try:
            arr_out = np.memmap(mmap_path, mode="w+", dtype="float32", shape=da.shape)

            if not spatial_dims:
                y = da.astype("float64").values.reshape(ntime, 1)
                arr_out[...] = _detrend_matrix(y).reshape(da.shape)
            else:
                lead_dim = spatial_dims[0]
                lead_axis = dims.index(lead_dim)
                nlead = da.sizes[lead_dim]
                print(f"[detrend] {var}: quadratic chunked detrend along {lead_dim} with chunk={spatial_chunk_size}", flush=True)
                for start in range(0, nlead, spatial_chunk_size):
                    end = min(start + spatial_chunk_size, nlead)
                    print(f"[detrend] {var}: {lead_dim}[{start}:{end}] / {nlead}", flush=True)
                    block = da.isel({lead_dim: slice(start, end)}).astype("float64").values
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
            append_history(
                out,
                f"Applied quadratic detrend (degree=2 polynomial) to {var} before anomaly calculation in spatial chunks of {spatial_chunk_size}."
            )
            encoding = {
                var: {
                    "zlib": True,
                    "complevel": 4,
                    "shuffle": True,
                    "dtype": "float32",
                    "_FillValue": fill_value,
                }
            }
            out.to_netcdf(dst, encoding=encoding)
        finally:
            try:
                mmap_path.unlink(missing_ok=True)
            except Exception:
                pass

def prepare_ocean_remap_input_file(
    src: Path,
    dst: Path,
    var: str,
    fill_value: np.float32,
) -> tuple[bool, bool]:
    with xr.open_dataset(src, decode_times=False) as ds:
        out, lon_changed = repair_cyclic_longitude(ds, var)
        out, meta_changed = strip_spatial_bounds_for_remap(out, var)
        write_single_var_dataset(out, dst, var, fill_value)
        return lon_changed, meta_changed


def cdo_show_names(infile: Path) -> list[str]:
    result = subprocess.run(
        ["cdo", "-s", "showname", str(infile)],
        check=True,
        capture_output=True,
        text=True,
    )
    return [name for name in result.stdout.replace("\n", " ").split() if name]


def resolve_var(infile: Path, candidates: Iterable[str]) -> str:
    names = cdo_show_names(infile)
    for cand in candidates:
        if cand in names:
            return cand
    if not names:
        raise ValueError(f"No variable names found in {infile}")
    return names[0]


@dataclass(frozen=True)
class Job:
    source: str
    var: str
    infile: Path
    outfile: Path
    outvar: str
    invar_candidates: tuple[str, ...]
    keep_intermediate_dir: Path | None


def jobs_for_cmip6(
    in_dir: Path,
    out_dir: Path,
    selected_vars: set[str],
    keep_intermediate: bool,
) -> list[Job]:
    jobs: list[Job] = []
    files = sorted(in_dir.rglob("monthly/*.nc"))

    for infile in files:
        rel = infile.relative_to(in_dir)
        base = infile.stem
        outvar = base.split("_", 1)[0]
        if outvar not in selected_vars:
            continue
        outfile = (out_dir / rel).with_name(f"{base}.anom_1x2.nc")
        keep_dir = None
        if keep_intermediate:
            keep_dir = outfile.parent / "_intermediate" / base
        jobs.append(
            Job(
                source="cmip6",
                var=outvar,
                infile=infile,
                outfile=outfile,
                outvar=outvar,
                invar_candidates=SOURCE_VAR_CANDIDATES["cmip6"][outvar],
                keep_intermediate_dir=keep_dir,
            )
        )
    return jobs


def build_fixed_source_file_map(
    source: str,
    in_dir: Path,
    selected_vars: set[str],
    *,
    oras5_tag: str,
) -> dict[str, str]:
    if source == "era5":
        return ERA5_FILES
    if source == "oras5":
        return build_oras5_file_map(oras5_tag)
    if source == "godas":
        return discover_godas_files(in_dir, selected_vars)
    if source == "soda":
        return SODA_FILES
    if source == "20crv2":
        return CR20V2_FILES
    raise ValueError(f"Unsupported fixed source: {source}")


def build_fixed_source_outfile(source: str, out_dir: Path, outvar: str, infile: Path) -> Path:
    if source in {"soda", "20crv2"}:
        return out_dir / f"{outvar}.{infile.stem}.anom_1x2.nc"
    if source == "godas":
        return out_dir / f"{infile.stem}.anom_1x2.nc"
    return out_dir / f"{infile.stem}.anom_1x2.nc"


def jobs_for_fixed_source(
    source: str,
    in_dir: Path,
    out_dir: Path,
    selected_vars: set[str],
    keep_intermediate: bool,
    *,
    oras5_tag: str,
) -> list[Job]:
    jobs: list[Job] = []
    file_map = build_fixed_source_file_map(source, in_dir, selected_vars, oras5_tag=oras5_tag)

    for outvar, filename in file_map.items():
        if outvar not in selected_vars:
            continue

        infile = in_dir / filename if source != "godas" else Path(filename)
        outfile = build_fixed_source_outfile(source, out_dir, outvar, infile)

        keep_dir = None
        if keep_intermediate:
            keep_dir = out_dir / "_intermediate" / outvar

        jobs.append(
            Job(
                source=source,
                var=outvar,
                infile=infile,
                outfile=outfile,
                outvar=outvar,
                invar_candidates=SOURCE_VAR_CANDIDATES[source][outvar],
                keep_intermediate_dir=keep_dir,
            )
        )
    return jobs


def discover_godas_files(in_dir: Path, selected_vars: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for var in selected_vars:
        matches = sorted(in_dir.glob(f"{var}.*.nc"))
        if not matches:
            continue
        found[var] = str(matches[-1])
    return found


def build_jobs(
    source: str,
    in_dir: Path,
    out_dir: Path,
    selected_vars: set[str],
    keep_intermediate: bool,
    *,
    oras5_tag: str,
) -> list[Job]:
    if source == "cmip6":
        return jobs_for_cmip6(in_dir, out_dir, selected_vars, keep_intermediate)
    return jobs_for_fixed_source(
        source,
        in_dir,
        out_dir,
        selected_vars,
        keep_intermediate,
        oras5_tag=oras5_tag,
    )


def maybe_keep_intermediate(keep_dir: Path | None, *files: Path) -> None:
    if keep_dir is None:
        return
    keep_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        if path.exists():
            shutil.copy2(path, keep_dir / path.name)


def process_job(
    job: Job,
    *,
    gridfile: Path,
    zip_level: int,
    overwrite: bool,
    fill_value: np.float32,
    use_finite_fill_before_detrend: bool,
    apply_setmissval_after_remap: bool,
    annotate_final_with_fill_attrs: bool,
    detrend_spatial_chunk: int,
) -> None:
    if not job.infile.exists():
        log(f"[missing] {job.infile}")
        return

    if job.outfile.exists() and not overwrite:
        log(f"[skip] {job.outfile}")
        return

    job.outfile.parent.mkdir(parents=True, exist_ok=True)
    invar = resolve_var(job.infile, job.invar_candidates)

    with tempfile.TemporaryDirectory(prefix=f"{job.source}_{job.outvar}_", dir=str(job.outfile.parent)) as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        stem = job.outfile.stem.replace(".anom_1x2", "")

        sel = tmpdir / f"{stem}.sel.nc"
        ren = tmpdir / f"{stem}.ren.nc"
        clean = tmpdir / f"{stem}.clean.nc"
        detr = tmpdir / f"{stem}.detr.nc"
        clim = tmpdir / f"{stem}.clim.nc"
        anom = tmpdir / f"{stem}.anom.nc"
        prep = tmpdir / f"{stem}.prep.nc"
        remap = tmpdir / f"{stem}.anom_1x2.tmp.nc"
        setmiss = tmpdir / f"{stem}.anom_1x2.setmiss.nc"

        log(f"[proc] {job.infile} (input var: {invar} -> output var: {job.outvar})")

        run(["cdo", "-L", "-O", "-f", "nc4c", "-z", f"zip_{zip_level}", f"selname,{invar}", str(job.infile), str(sel)])

        if invar != job.outvar:
            run(["cdo", "-L", "-O", "-f", "nc4c", "-z", f"zip_{zip_level}", f"chname,{invar},{job.outvar}", str(sel), str(ren)])
        else:
            shutil.copy2(sel, ren)

        detr_input = ren
        if use_finite_fill_before_detrend:
            rewrite_with_finite_fill(
                ren,
                clean,
                job.outvar,
                fill_value,
                add_anom_attrs=False,
                source=job.source,
                gridfile=gridfile,
            )
            detr_input = clean

        quadratic_detrend_file(
            detr_input,
            detr,
            job.outvar,
            fill_value,
            spatial_chunk_size=detrend_spatial_chunk,
        )
        run(["cdo", "-L", "-O", "-f", "nc4c", "-z", f"zip_{zip_level}", "ymonmean", str(detr), str(clim)])
        run(["cdo", "-L", "-O", "-f", "nc4c", "-z", f"zip_{zip_level}", "ymonsub", str(detr), str(clim), str(anom)])

        remap_input = anom
        if job.outvar in OCEAN_VARS:
            lon_changed, meta_changed = prepare_ocean_remap_input_file(
                anom,
                prep,
                job.outvar,
                fill_value,
            )
            if lon_changed:
                log(f"[fix ] cyclic seam repaired before remap: {job.outvar}")
            if meta_changed:
                log(f"[fix ] dropped spatial bounds/vertex metadata before remap: {job.outvar}")
            remap_input = prep

        run(["cdo", "-L", "-O", "-f", "nc4c", "-z", f"zip_{zip_level}", f"remapbil,{gridfile}", str(remap_input), str(remap)])

        final_input = remap
        if apply_setmissval_after_remap:
            run(["cdo", "-L", "-O", "-f", "nc4c", "-z", f"zip_{zip_level}", f"setmissval,{fill_value}", str(remap), str(setmiss)])
            final_input = setmiss

        maybe_keep_intermediate(
            job.keep_intermediate_dir,
            sel,
            ren,
            clean,
            detr,
            clim,
            anom,
            prep,
            remap,
            setmiss,
        )

        if annotate_final_with_fill_attrs:
            rewrite_with_finite_fill(
                final_input,
                job.outfile,
                job.outvar,
                fill_value,
                add_anom_attrs=True,
                source=job.source,
                gridfile=gridfile,
            )
        else:
            shutil.move(final_input, job.outfile)

        log(f"[done] {job.outfile}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create monthly anomaly files for CMIP6 / ERA5 / ORAS5 / GODAS / SODA / 20CRv2 "
            "using quadratic detrend + monthly climatology removal + 1x2 remap."
        )
    )
    parser.add_argument("--source", required=True, choices=sorted(SOURCE_DEFAULTS))
    parser.add_argument("--in-dir", type=Path, help="Input directory. Default depends on --source.")
    parser.add_argument("--out-dir", type=Path, help="Output directory. Default depends on --source.")
    parser.add_argument("--gridfile", type=Path, default=COMMON_GRIDFILE)
    parser.add_argument("--vars", nargs="+", help="Subset of variables to process.")
    parser.add_argument("--zip-level", type=int, default=4, choices=range(1, 10))
    parser.add_argument("--fill-value", type=np.float32, default=COMMON_MISSVAL)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--detrend-spatial-chunk", type=int, default=4, help="Number of indices to process at once along the first non-time dimension during quadratic detrend.")
    parser.add_argument(
        "--oras5-tag",
        default=ORAS5_RANGE_TAG,
        help="Input period tag for ORAS5 raw filenames, e.g. 195801-198812.",
    )
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> tuple[str, Path, Path, Path, set[str], int, np.float32, bool, bool, int, str]:
    defaults = SOURCE_DEFAULTS[args.source]
    in_dir = args.in_dir.expanduser().resolve() if args.in_dir else defaults.in_dir
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else defaults.out_dir
    gridfile = args.gridfile.expanduser().resolve()

    supported = set(defaults.target_vars)
    if args.vars:
        selected = set(args.vars)
        unsupported = sorted(selected - supported)
        if unsupported:
            raise SystemExit(
                f"[error] unsupported vars for source={args.source}: {unsupported}; supported={sorted(supported)}"
            )
    else:
        selected = supported

    return (
        args.source,
        in_dir,
        out_dir,
        gridfile,
        selected,
        zip_level := args.zip_level,
        fill_value := args.fill_value,
        overwrite := args.overwrite,
        keep_intermediate := args.keep_intermediate,
        detrend_spatial_chunk := args.detrend_spatial_chunk,
        oras5_tag := args.oras5_tag,
    )


def main() -> None:
    args = parse_args()
    (
        source,
        in_dir,
        out_dir,
        gridfile,
        selected_vars,
        zip_level,
        fill_value,
        overwrite,
        keep_intermediate,
        detrend_spatial_chunk,
        oras5_tag,
    ) = normalize_args(args)

    if not in_dir.exists():
        raise SystemExit(f"[error] input dir not found: {in_dir}")

    ensure_requirements(gridfile)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(source, in_dir, out_dir, selected_vars, keep_intermediate, oras5_tag=oras5_tag)
    if not jobs:
        raise SystemExit(f"[error] no input files found for source={source} in {in_dir}")

    defaults = SOURCE_DEFAULTS[source]
    annotate_final_with_fill_attrs = defaults.use_finite_fill_before_detrend

    for job in jobs:
        process_job(
            job,
            gridfile=gridfile,
            zip_level=zip_level,
            overwrite=overwrite,
            fill_value=fill_value,
            use_finite_fill_before_detrend=defaults.use_finite_fill_before_detrend,
            apply_setmissval_after_remap=defaults.apply_setmissval_after_remap,
            annotate_final_with_fill_attrs=annotate_final_with_fill_attrs,
            detrend_spatial_chunk=detrend_spatial_chunk,
        )

    log(f"[all done] source={source} finished.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
