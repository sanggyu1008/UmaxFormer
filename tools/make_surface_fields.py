#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import xarray as xr

from preprocess_config import GODAS_RAW_FILE_CANDIDATES, GODAS_SURFACE_FILES, ORAS5_RANGE_TAG, require_existing_path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CMIP6_ROOT = (SCRIPT_DIR / "../data/raw/cmip6/ssp370").resolve()
DEFAULT_GODAS_ROOT = Path("/mnt/d/project/UmaxFormer/data/raw/godas")
DEFAULT_ORAS5_ROOT = Path("/mnt/d/project/UmaxFormer/data/raw/oras5")

DEPTH_DIM_CANDIDATES = ["level", "lev", "depth", "zlev", "olevel", "z", "deptht", "st_ocean"]
DROP_ATTRS_AFTER_DERIVATION = {
    "actual_range", "valid_range", "valid_min", "valid_max", "scale_factor", "add_offset"
}
KGKG_UNITS = {
    "kg/kg", "kg kg-1", "kg kg^-1", "kgkg-1", "kgkg^-1", "mass fraction"
}

CMIP6_JOBS = {
    "thetao": {"dst": "tos", "pattern": "**/monthly/thetao_Omon_*_time.nc"},
    "uo": {"dst": "uos", "pattern": "**/monthly/uo_Omon_*_time.nc"},
    "vo": {"dst": "vos", "pattern": "**/monthly/vo_Omon_*_time.nc"},
}

GODAS_JOBS = {
    "pottmp": {
        "src_name": "pottmp",
        "dst_name": "tos",
        "src_files": GODAS_RAW_FILE_CANDIDATES["pottmp"],
        "dst_file": GODAS_SURFACE_FILES["tos"],
        "long_name": "Surface potential temperature",
    },
    "salt": {
        "src_name": "salt",
        "dst_name": "sos",
        "src_files": GODAS_RAW_FILE_CANDIDATES["salt"],
        "dst_file": GODAS_SURFACE_FILES["sos"],
        "long_name": "Surface salinity",
    },
    "ucur": {
        "src_name": "ucur",
        "dst_name": "uos",
        "src_files": GODAS_RAW_FILE_CANDIDATES["ucur"],
        "dst_file": GODAS_SURFACE_FILES["uos"],
        "long_name": "Surface eastward current",
    },
    "vcur": {
        "src_name": "vcur",
        "dst_name": "vos",
        "src_files": GODAS_RAW_FILE_CANDIDATES["vcur"],
        "dst_file": GODAS_SURFACE_FILES["vos"],
        "long_name": "Surface northward current",
    },
}

ORAS5_JOBS = {
    "vozocrtx": {"subdir": "zonal_velocity", "dst": "uos"},
    "vomecrty": {"subdir": "meridional_velocity", "dst": "vos"},
}


def log(msg: str) -> None:
    print(msg, flush=True)



def ensure_command(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"[error] required command not found in PATH: {name}")



def run(cmd: list[str]) -> None:
    log("[run] " + " ".join(cmd))
    subprocess.run(cmd, check=True)



def replace_file(tmp_path: Path, out_path: Path) -> None:
    tmp_path.replace(out_path)



def normalize_units(units: str) -> str:
    return " ".join(str(units).strip().lower().replace("**", "^").split())



def append_history(ds: xr.Dataset, message: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = f"{now}: {message}"
    old = ds.attrs.get("history")
    ds.attrs["history"] = f"{entry}\n{old}" if old else entry



def find_depth_dim(da: xr.DataArray) -> str:
    for name in DEPTH_DIM_CANDIDATES:
        if name in da.dims:
            return name
    for dim in da.dims:
        if dim in da.coords:
            units = str(da[dim].attrs.get("units", "")).lower()
            if units in ("m", "meter", "meters") or "meter" in units:
                return dim
    raise ValueError(f"Cannot find depth dimension in {da.dims}")



def carry_aux_vars(ds: xr.Dataset, out: xr.Dataset, main_var: str) -> xr.Dataset:
    out_dims = set(out[main_var].dims)
    for name, var in ds.variables.items():
        if name in out.variables or name == main_var or name in ds.coords:
            continue
        if set(var.dims).issubset(out_dims):
            out[name] = var
    return out



def build_godas_surface_field(
    src_path: Path,
    src_var: str,
    dst_var: str,
    long_name: str,
    convert_salt_to_gkg: bool,
) -> xr.Dataset:
    with xr.open_dataset(src_path, decode_times=False) as ds:
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
        notes: list[str] = []

        if src_var == "salt" and convert_salt_to_gkg:
            units = normalize_units(surf.attrs.get("units", ""))
            if units in KGKG_UNITS:
                surf = surf * 1000.0
                surf.attrs["units"] = "g/kg"
                surf.attrs["salinity_unit_conversion"] = (
                    "Converted from kg/kg to g/kg by multiplying by 1000."
                )
                notes.append("salinity kg/kg->g/kg")
            else:
                surf.attrs["salinity_unit_warning"] = (
                    f"Source units '{surf.attrs.get('units', '')}' were not recognized as kg/kg. "
                    "Values were copied without rescaling."
                )

        surf = surf.rename(dst_var)
        attrs = dict(surf.attrs)
        for key in DROP_ATTRS_AFTER_DERIVATION:
            attrs.pop(key, None)
        attrs.update(
            {
                "long_name": long_name,
                "source_variable": src_var,
                "selection": "first_vertical_level",
                "comment": "Derived from GODAS by selecting the first vertical level and renaming the variable.",
            }
        )
        if source_level_m is not None:
            attrs["source_level_m"] = source_level_m
        if notes:
            attrs["postprocess"] = "; ".join(notes)
        surf.attrs = attrs

        out = surf.to_dataset(name=dst_var)
        out.attrs.update(ds.attrs)
        out = carry_aux_vars(ds, out, dst_var)

    msg = f"Created {dst_var} from {src_path.name} by selecting first vertical level"
    if notes:
        msg += f" and applying {', '.join(notes)}"
    append_history(out, msg)
    return out



def save_dataset(ds: xr.Dataset, out_path: Path) -> None:
    tmp_path = out_path.with_name(out_path.name + f".tmp.{os.getpid()}")
    encoding = {
        list(ds.data_vars)[0]: {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "dtype": "float32",
        }
    }
    ds.to_netcdf(tmp_path, encoding=encoding)
    replace_file(tmp_path, out_path)


def smoke_check_output(out_path: Path, expected_var: str) -> None:
    with xr.open_dataset(out_path, decode_times=False) as ds:
        if expected_var not in ds.data_vars:
            raise ValueError(f"{out_path} missing expected variable '{expected_var}'")
        dims = set(ds[expected_var].dims)
        if {"time", "lat", "lon"}.issubset(dims):
            return
        if {"time_counter", "y", "x"}.issubset(dims):
            return
        raise ValueError(f"{out_path} has unexpected dims for {expected_var}: {ds[expected_var].dims}")



def run_cmip6(root: Path, vars_to_process: Iterable[str], zip_level: int, reduce_dim: bool, overwrite: bool) -> None:
    ensure_command("cdo")
    root = root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"[error] CMIP6 root does not exist: {root}")

    log(f"[info] CMIP6 ROOT       = {root}")
    log(f"[info] CMIP6 ZIPLVL     = {zip_level}")
    log(f"[info] CMIP6 REDUCE_DIM = {1 if reduce_dim else 0}")

    cdo_base = ["cdo", "-O", "-L", "-f", "nc4c", "-z", f"zip_{zip_level}"]
    if reduce_dim:
        cdo_base.append("--reduce_dim")

    any_found = False
    for src in vars_to_process:
        job = CMIP6_JOBS[src]
        dst = job["dst"]
        files = sorted(root.glob(job["pattern"]))
        if not files:
            log(f"[warn] no files matched: {job['pattern']}")
            continue
        any_found = True

        for in_path in files:
            out_name = in_path.name.replace(f"{src}_", f"{dst}_", 1)
            out_path = in_path.with_name(out_name)
            tmp_path = out_path.with_name(out_path.name + f".tmp.{os.getpid()}")

            if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
                log(f"[skip] {out_path}")
                continue

            cmd = cdo_base + [f"-sellevidx,1", f"-chname,{src},{dst}", str(in_path), str(tmp_path)]
            try:
                log(f"[make] {out_path}")
                run(cmd)
                replace_file(tmp_path, out_path)
            except subprocess.CalledProcessError:
                tmp_path.unlink(missing_ok=True)
                raise SystemExit(f"[error] failed: {in_path} -> {out_path}")

    if not any_found:
        log("[warn] no CMIP6 input files were found.")
    else:
        log("[done] CMIP6 surface fields generated into each monthly/ directory.")



def run_godas(root: Path, vars_to_process: Iterable[str], overwrite: bool, keep_salt_kgkg: bool) -> None:
    root = root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"[error] GODAS root does not exist: {root}")

    for key in vars_to_process:
        job = GODAS_JOBS[key]
        try:
            src_path = require_existing_path(root, job["src_files"], f"GODAS input for {key}")
        except FileNotFoundError as exc:
            raise SystemExit(f"[error] {exc}") from exc
        out_path = root / job["dst_file"]

        if out_path.exists() and not overwrite:
            log(f"[skip] {out_path}")
            continue

        log(f"[IN ] {src_path}")
        log(f"[OUT] {out_path}")
        out = build_godas_surface_field(
            src_path=src_path,
            src_var=job["src_name"],
            dst_var=job["dst_name"],
            long_name=job["long_name"],
            convert_salt_to_gkg=not keep_salt_kgkg,
        )
        save_dataset(out, out_path)
        smoke_check_output(out_path, job["dst_name"])

    log("[done] GODAS surface fields generated.")



def run_oras5(
    root: Path,
    vars_to_process: Iterable[str],
    overwrite: bool,
    tmp_dir: Path | None,
    time_dim: str,
    tag: str,
    check: bool,
) -> None:
    ensure_command("cdo")
    ensure_command("ncrcat")
    ensure_command("ncks")

    root = root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"[error] ORAS5 root does not exist: {root}")
    tmp_root = (tmp_dir.expanduser().resolve() if tmp_dir is not None else root / "_tmp_surface_oras5")
    tmp_root.mkdir(parents=True, exist_ok=True)

    for stem in vars_to_process:
        job = ORAS5_JOBS[stem]
        outvar = job["dst"]
        indir = root / job["subdir"]
        pattern = f"{stem}_control_monthly_highres_3D_*.nc"
        files = sorted(indir.glob(pattern))
        out_path = root / f"{outvar}.{tag}.nc"

        if out_path.exists() and not overwrite:
            log(f"[skip] {out_path}")
            continue
        if not files:
            raise SystemExit(f"[error] no input files found for {stem} in {indir}")

        with tempfile.TemporaryDirectory(prefix=f"oras5_{outvar}_", dir=str(tmp_root)) as work_str:
            work = Path(work_str)
            seldig = work / "selz"
            seldig.mkdir(parents=True, exist_ok=True)

            log(f"=== building {outvar} from {stem} ===")

            for in_path in files:
                sel_path = seldig / in_path.name
                run(["cdo", "-L", "-O", "sellevidx,1", str(in_path), str(sel_path)])
                run(["ncks", "-O", "--mk_rec_dmn", time_dim, str(sel_path), str(sel_path)])

            merged3d = work / f"{stem}_surface_3d_{tag}.nc"
            merged2d = work / f"{stem}_surface_2d_{tag}.nc"
            tmp_out = out_path.with_name(out_path.name + f".tmp.{os.getpid()}")

            run(["ncrcat", "-O", *[str(p) for p in sorted(seldig.glob("*.nc"))], str(merged3d)])
            run(["cdo", "-L", "-O", "--reduce_dim", "copy", str(merged3d), str(merged2d)])
            run(["cdo", "-L", "-O", f"chname,{stem},{outvar}", str(merged2d), str(tmp_out)])
            replace_file(tmp_out, out_path)

        smoke_check_output(out_path, outvar)

        if check:
            run(["cdo", "showname", str(out_path)])
            run(["cdo", "ntime", str(out_path)])
            run(["ncdump", "-h", str(out_path)])

    log("[done] ORAS5 surface fields generated.")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified surface-field builder for CMIP6, GODAS, and ORAS5."
    )
    p.add_argument(
        "--source",
        choices=["cmip6", "godas", "oras5", "all"],
        default="all",
        help="Which source pipeline to run.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")

    p.add_argument("--cmip6-root", type=Path, default=DEFAULT_CMIP6_ROOT)
    p.add_argument(
        "--cmip6-vars",
        nargs="+",
        choices=list(CMIP6_JOBS.keys()),
        default=list(CMIP6_JOBS.keys()),
        help="Subset of CMIP6 variables to process: thetao uo vo",
    )
    p.add_argument("--zip-level", type=int, default=4, help="CMIP6 output compression level.")
    p.add_argument(
        "--no-reduce-dim",
        action="store_true",
        help="For CMIP6, keep singleton vertical dimension instead of using --reduce_dim.",
    )

    p.add_argument("--godas-root", type=Path, default=DEFAULT_GODAS_ROOT)
    p.add_argument(
        "--godas-vars",
        nargs="+",
        choices=list(GODAS_JOBS.keys()),
        default=list(GODAS_JOBS.keys()),
        help="Subset of GODAS variables to process: pottmp salt ucur vcur",
    )
    p.add_argument(
        "--keep-salt-kgkg",
        action="store_true",
        help="For GODAS salt, do not convert kg/kg to g/kg.",
    )

    p.add_argument("--oras5-root", type=Path, default=DEFAULT_ORAS5_ROOT)
    p.add_argument(
        "--oras5-vars",
        nargs="+",
        choices=list(ORAS5_JOBS.keys()),
        default=list(ORAS5_JOBS.keys()),
        help="Subset of ORAS5 variables to process: vozocrtx vomecrty",
    )
    p.add_argument(
        "--oras5-tmp",
        type=Path,
        default=None,
        help="Temporary working directory for ORAS5. Default: <oras5-root>/_tmp_surface_oras5",
    )
    p.add_argument(
        "--oras5-time-dim",
        default="time_counter",
        help="Record dimension name to set with ncks for ORAS5.",
    )
    p.add_argument(
        "--oras5-tag",
        default=ORAS5_RANGE_TAG,
        help="Output period tag for ORAS5 output filenames.",
    )
    p.add_argument(
        "--oras5-check",
        action="store_true",
        help="Run cdo showname/ntime and ncdump -h on ORAS5 outputs.",
    )
    return p.parse_args()



def main() -> None:
    args = parse_args()

    if args.source in {"cmip6", "all"}:
        run_cmip6(
            root=args.cmip6_root,
            vars_to_process=args.cmip6_vars,
            zip_level=args.zip_level,
            reduce_dim=not args.no_reduce_dim,
            overwrite=args.overwrite,
        )
        print()

    if args.source in {"godas", "all"}:
        run_godas(
            root=args.godas_root,
            vars_to_process=args.godas_vars,
            overwrite=args.overwrite,
            keep_salt_kgkg=args.keep_salt_kgkg,
        )
        print()

    if args.source in {"oras5", "all"}:
        run_oras5(
            root=args.oras5_root,
            vars_to_process=args.oras5_vars,
            overwrite=args.overwrite,
            tmp_dir=args.oras5_tmp,
            time_dim=args.oras5_time_dim,
            tag=args.oras5_tag,
            check=args.oras5_check,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
