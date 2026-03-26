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


COMMON_GRIDFILE = Path("/mnt/d/project/01_ENSO/01_data/01_raw/grid_1x2_60S60N_120x180.txt")
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
        in_dir=Path("/mnt/d/project/01_ENSO/01_data/01_raw/cmip6/ssp370"),
        out_dir=Path("/mnt/d/project/01_ENSO/01_data/02_interim/cmip6/ssp370"),
        target_vars=("mlotst", "ohc300", "psl", "sos", "tos", "uas", "uos", "vas", "vos"),
        use_finite_fill_before_detrend=False,
        apply_setmissval_after_remap=False,
    ),
    "era5": SourceDefaults(
        in_dir=Path("/mnt/d/project/01_ENSO/01_data/01_raw/era5"),
        out_dir=Path("/mnt/d/project/01_ENSO/01_data/02_interim/era5"),
        target_vars=("psl", "uas", "vas"),
        use_finite_fill_before_detrend=True,
        apply_setmissval_after_remap=True,
    ),
    "oras5": SourceDefaults(
        in_dir=Path("/mnt/d/project/01_ENSO/01_data/01_raw/oras5"),
        out_dir=Path("/mnt/d/project/01_ENSO/01_data/02_interim/oras5"),
        target_vars=("mlotst", "ohc300", "sos", "tos", "uos", "vos"),
        use_finite_fill_before_detrend=False,
        apply_setmissval_after_remap=True,
    ),
    "godas": SourceDefaults(
        in_dir=Path("/mnt/d/project/01_ENSO/01_data/01_raw/godas"),
        out_dir=Path("/mnt/d/project/01_ENSO/01_data/02_interim/godas"),
        target_vars=("mlotst", "tos", "vos", "ohc300", "sos", "uos"),
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
}


ERA5_FILES = {
    "psl": "psl.mon.195801-202512.nc",
    "uas": "uas.mon.195801-202512.nc",
    "vas": "vas.mon.195801-202512.nc",
}

ORAS5_FILES = {
    "mlotst": "mlotst.195801-197812.nc",
    "ohc300": "ohc300.195801-197812.nc",
    "sos": "sos.195801-197812.nc",
    "tos": "tos.195801-197812.nc",
    "uos": "uos.195801-197812.nc",
    "vos": "vos.195801-197812.nc",
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
                "anomaly_definition": "linear_detrend_then_remove_calendar_month_climatology",
                "target_grid": gridfile.name,
                "regridding": "cdo remapbil",
                "source": source,
            })
            append_history(
                out,
                f"Created anomaly file for {var}: detrend + ymonmean/ymonsub + remapbil({gridfile})"
            )
        else:
            append_history(
                out,
                f"Rewrote input for {var} with explicit finite _FillValue before CDO processing"
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


def jobs_for_fixed_source(
    source: str,
    in_dir: Path,
    out_dir: Path,
    selected_vars: set[str],
    keep_intermediate: bool,
) -> list[Job]:
    jobs: list[Job] = []

    if source == "era5":
        file_map = ERA5_FILES
    elif source == "oras5":
        file_map = ORAS5_FILES
    elif source == "godas":
        file_map = discover_godas_files(in_dir, selected_vars)
    else:
        raise ValueError(f"Unsupported fixed source: {source}")

    for outvar, filename in file_map.items():
        if outvar not in selected_vars:
            continue
        infile = in_dir / filename if source != "godas" else Path(filename)
        if source == "godas":
            infile = Path(filename)
        if source == "godas":
            stem = infile.stem
            outfile = out_dir / f"{stem}.anom_1x2.nc"
        else:
            outfile = out_dir / f"{Path(filename).stem}.anom_1x2.nc"
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
) -> list[Job]:
    if source == "cmip6":
        return jobs_for_cmip6(in_dir, out_dir, selected_vars, keep_intermediate)
    return jobs_for_fixed_source(source, in_dir, out_dir, selected_vars, keep_intermediate)


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

        run(["cdo", "-L", "-O", "-f", "nc4c", "-z", f"zip_{zip_level}", "detrend", str(detr_input), str(detr)])
        run(["cdo", "-L", "-O", "-f", "nc4c", "-z", f"zip_{zip_level}", "ymonmean", str(detr), str(clim)])
        run(["cdo", "-L", "-O", "-f", "nc4c", "-z", f"zip_{zip_level}", "ymonsub", str(detr), str(clim), str(anom)])
        run(["cdo", "-L", "-O", "-f", "nc4c", "-z", f"zip_{zip_level}", f"remapbil,{gridfile}", str(anom), str(remap)])

        final_input = remap
        if apply_setmissval_after_remap:
            run(["cdo", "-L", "-O", "-f", "nc4c", "-z", f"zip_{zip_level}", f"setmissval,{fill_value}", str(remap), str(setmiss)])
            final_input = setmiss

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

        maybe_keep_intermediate(
            job.keep_intermediate_dir,
            sel,
            ren,
            clean,
            detr,
            clim,
            anom,
            remap,
            setmiss,
        )

        log(f"[done] {job.outfile}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create monthly anomaly files for CMIP6 / ERA5 / ORAS5 / GODAS "
            "using linear detrend + monthly climatology removal + 1x2 remap."
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
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> tuple[str, Path, Path, Path, set[str], int, np.float32, bool, bool]:
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
        args.zip_level,
        args.fill_value,
        args.overwrite,
        args.keep_intermediate,
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
    ) = normalize_args(args)

    if not in_dir.exists():
        raise SystemExit(f"[error] input dir not found: {in_dir}")

    ensure_requirements(gridfile)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(source, in_dir, out_dir, selected_vars, keep_intermediate)
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
        )

    log(f"[all done] source={source} finished.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
