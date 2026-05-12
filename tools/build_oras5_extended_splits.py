#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


OLD_TAG = "195801-197812"
EXT_TAG = "197901-198812"
MERGED_TAG = "195801-198812"
TEST_TAG = "199001-202512"

ORAS5_VARS = ("mlotst", "ohc300", "sos", "tos", "uos", "vos")
ORAS5_2D_JOBS = {
    "mixed_layer_depth_0_03": "mlotst",
    "ocean_heat_content_for_the_upper_300m": "ohc300",
    "sea_surface_salinity": "sos",
    "sea_surface_temperature": "tos",
}
ORAS5_3D_VARS = ("vozocrtx", "vomecrty")


def log(message: str) -> None:
    print(message, flush=True)


def run(cmd: list[str], *, dry_run: bool = False) -> None:
    log("[run] " + " ".join(str(x) for x in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def command_output(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"[error] required command not found in PATH: {name}")


def period_month_tokens(start_year: int, end_year: int) -> tuple[str, ...]:
    return tuple(f"{year}{month:02d}" for year in range(start_year, end_year + 1) for month in range(1, 13))


def monthly_files_for_period(directory: Path, start_year: int, end_year: int) -> list[Path]:
    tokens = period_month_tokens(start_year, end_year)
    files = sorted(
        path
        for path in directory.glob("*.nc")
        if path.is_file() and any(token in path.name for token in tokens)
    )
    expected = (end_year - start_year + 1) * 12
    if len(files) < expected:
        raise SystemExit(
            f"[error] expected at least {expected} monthly files in {directory}, found {len(files)}"
        )
    return files


def cdo_show_first_var(path: Path) -> str:
    names = command_output(["cdo", "-s", "showname", str(path)]).split()
    if not names:
        raise SystemExit(f"[error] cdo showname returned no variables for {path}")
    return names[0]


def replace_file(tmp_path: Path, out_path: Path) -> None:
    tmp_path.replace(out_path)


def build_oras5_2d_raw(raw_root: Path, *, overwrite: bool, dry_run: bool) -> None:
    for download_var, outvar in ORAS5_2D_JOBS.items():
        source_dir = raw_root / download_var
        out_path = raw_root / f"{outvar}.{EXT_TAG}.nc"
        if dry_run:
            log(f"[dry-run] would build {out_path} from monthly files in {source_dir}")
            continue
        if out_path.exists() and not overwrite:
            log(f"[skip] {out_path}")
            continue
        if not source_dir.is_dir():
            raise SystemExit(f"[error] missing ORAS5 monthly directory: {source_dir}")

        files = monthly_files_for_period(source_dir, 1979, 1988)
        tmp_merged = out_path.with_name(out_path.name + ".tmp.merge")
        tmp_named = out_path.with_name(out_path.name + ".tmp.named")

        run(["cdo", "-O", "-L", "-f", "nc4c", "-z", "zip_4", "mergetime", *map(str, files), str(tmp_merged)], dry_run=dry_run)
        if dry_run:
            continue

        invar = cdo_show_first_var(tmp_merged)
        if invar == outvar:
            replace_file(tmp_merged, out_path)
            tmp_named.unlink(missing_ok=True)
        else:
            run(["cdo", "-O", "-L", f"chname,{invar},{outvar}", str(tmp_merged), str(tmp_named)])
            tmp_merged.unlink(missing_ok=True)
            replace_file(tmp_named, out_path)
        log(f"[done] {out_path}")


def build_oras5_3d_surface(
    project_root: Path,
    raw_root: Path,
    python_exe: str,
    *,
    overwrite: bool,
    dry_run: bool,
) -> None:
    cmd = [
        python_exe,
        str(project_root / "tools" / "make_surface_fields.py"),
        "--source",
        "oras5",
        "--oras5-root",
        str(raw_root),
        "--oras5-vars",
        *ORAS5_3D_VARS,
        "--oras5-tag",
        EXT_TAG,
    ]
    if overwrite:
        cmd.append("--overwrite")
    run(cmd, dry_run=dry_run)


def merge_oras5_raw(raw_root: Path, *, overwrite: bool, dry_run: bool) -> None:
    for var in ORAS5_VARS:
        old_path = raw_root / f"{var}.{OLD_TAG}.nc"
        ext_path = raw_root / f"{var}.{EXT_TAG}.nc"
        out_path = raw_root / f"{var}.{MERGED_TAG}.nc"
        if out_path.exists() and not overwrite:
            log(f"[skip] {out_path}")
            continue
        if not dry_run and not old_path.exists():
            raise SystemExit(f"[error] missing existing ORAS5 raw file: {old_path}")
        if not dry_run and not ext_path.exists():
            raise SystemExit(f"[error] missing incremental ORAS5 raw file: {ext_path}")
        tmp_path = out_path.with_name(out_path.name + ".tmp")
        run(
            ["cdo", "-O", "-L", "-f", "nc4c", "-z", "zip_4", "mergetime", str(old_path), str(ext_path), str(tmp_path)],
            dry_run=dry_run,
        )
        if not dry_run:
            replace_file(tmp_path, out_path)
        log(f"[done] {out_path}")


def build_anomaly(
    project_root: Path,
    raw_root: Path,
    interim_root: Path,
    python_exe: str,
    *,
    overwrite: bool,
    dry_run: bool,
) -> None:
    cmd = [
        python_exe,
        str(project_root / "tools" / "make_anom.py"),
        "--source",
        "oras5",
        "--in-dir",
        str(raw_root),
        "--out-dir",
        str(interim_root / "oras5"),
        "--vars",
        *ORAS5_VARS,
        "--oras5-tag",
        MERGED_TAG,
    ]
    if overwrite:
        cmd.append("--overwrite")
    run(cmd, dry_run=dry_run)


def build_final_splits(project_root: Path, data_root: Path, python_exe: str, *, dry_run: bool) -> None:
    run(
        [
            python_exe,
            str(project_root / "tools" / "make_input.py"),
            "--mode",
            "reanalysis",
            "--inroot",
            str(data_root / "interim"),
            "--outroot",
            str(data_root / "input"),
            "--validation-ocean-subdir",
            "oras5",
            "--validation-atm-subdir",
            "era5",
            "--validation-start",
            "1958-01",
            "--validation-end",
            "1988-12",
            "--validation-out",
            f"validation_dataset_oras5_era5_{MERGED_TAG.replace('-', '_')}.nc",
            "--test-start",
            "1990-01",
            "--test-end",
            "2025-12",
            "--test-out",
            f"test_dataset_godas_era5_{TEST_TAG.replace('-', '_')}.nc",
        ],
        dry_run=dry_run,
    )
    run(
        [
            python_exe,
            str(project_root / "tools" / "make_target.py"),
            "--skip-cmip6-targets",
            "--skip-test-target",
            "--validation-src",
            str(data_root / "interim" / "oras5" / f"tos.{MERGED_TAG}.anom_1x2.nc"),
            "--validation-start",
            "1958-01",
            "--validation-end",
            "1988-12",
            "--validation-out",
            f"validation_target_oras5_era5_{MERGED_TAG.replace('-', '_')}.nc",
        ],
        dry_run=dry_run,
    )
    run(
        [
            python_exe,
            str(project_root / "tools" / "make_target.py"),
            "--skip-validation-target",
            "--skip-cmip6-targets",
            "--test-src",
            str(data_root / "target" / "test_target_nino34_ersstv5_198001_202512.nc"),
            "--test-src-kind",
            "nino34",
            "--test-start",
            "1990-01",
            "--test-end",
            "2025-12",
            "--test-out",
            f"test_target_nino34_ersstv5_{TEST_TAG.replace('-', '_')}.nc",
        ],
        dry_run=dry_run,
    )


def timestamp_label(value) -> str:
    if hasattr(value, "year") and hasattr(value, "month"):
        return f"{int(value.year):04d}-{int(value.month):02d}"
    text = str(value)
    return f"{text[:4]}-{text[5:7]}"


def verify_time_range(path: Path, expected_start: str, expected_end: str, expected_months: int, variables: tuple[str, ...]) -> None:
    import xarray as xr

    if not path.exists():
        raise SystemExit(f"[error] missing expected output: {path}")
    with xr.open_dataset(path, decode_times=True) as ds:
        missing = [var for var in variables if var not in ds.data_vars]
        if missing:
            raise SystemExit(f"[error] {path} missing variables: {missing}")
        time_name = next((name for name in ("time", "time_counter", "valid_time", "t") if name in ds.coords or name in ds.dims), None)
        if time_name is None:
            raise SystemExit(f"[error] {path} has no time dimension")
        months = int(ds.sizes[time_name])
        start = timestamp_label(ds[time_name].values[0])
        end = timestamp_label(ds[time_name].values[-1])
    if (start, end, months) != (expected_start, expected_end, expected_months):
        raise SystemExit(
            f"[error] {path} expected {expected_start}..{expected_end} ({expected_months}), "
            f"got {start}..{end} ({months})"
        )
    log(f"[ok] {path} {start}..{end} ({months} months)")


def verify_outputs(data_root: Path, raw_root: Path) -> None:
    for var in ORAS5_VARS:
        verify_time_range(raw_root / f"{var}.{MERGED_TAG}.nc", "1958-01", "1988-12", 372, (var,))
        verify_time_range(data_root / "interim" / "oras5" / f"{var}.{MERGED_TAG}.anom_1x2.nc", "1958-01", "1988-12", 372, (var,))
    verify_time_range(
        data_root / "input" / f"validation_dataset_oras5_era5_{MERGED_TAG.replace('-', '_')}.nc",
        "1958-01",
        "1988-12",
        372,
        ORAS5_VARS + ("uas", "uasmax", "vas", "psl"),
    )
    verify_time_range(
        data_root / "target" / f"validation_target_oras5_era5_{MERGED_TAG.replace('-', '_')}.nc",
        "1958-01",
        "1988-12",
        372,
        ("nino34",),
    )
    verify_time_range(
        data_root / "input" / f"test_dataset_godas_era5_{TEST_TAG.replace('-', '_')}.nc",
        "1990-01",
        "2025-12",
        432,
        ORAS5_VARS + ("uas", "uasmax", "vas", "psl"),
    )
    verify_time_range(
        data_root / "target" / f"test_target_nino34_ersstv5_{TEST_TAG.replace('-', '_')}.nc",
        "1990-01",
        "2025-12",
        432,
        ("nino34",),
    )


def download_increment(project_root: Path, raw_root: Path, python_exe: str, *, dry_run: bool) -> None:
    run(
        [
            python_exe,
            str(project_root / "tools" / "download_oras5.py"),
            "--kind",
            "all",
            "--start-year",
            "1979",
            "--end-year",
            "1988",
            "--outdir",
            str(raw_root),
        ],
        dry_run=dry_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 1958-1988 ORAS5+ERA5 validation and 1990-2025 GODAS+ERA5 test NetCDF files."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run project scripts.")
    parser.add_argument("--skip-download", action="store_true", help="Use already downloaded 1979-1988 monthly ORAS5 files.")
    parser.add_argument("--skip-verify", action="store_true", help="Do not verify output periods after generation.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated files.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    data_root = project_root / "data"
    raw_root = data_root / "raw" / "oras5"
    interim_root = data_root / "interim"

    if not args.dry_run:
        require_command("cdo")
        require_command("ncrcat")
        require_command("ncks")

    if not args.skip_download:
        download_increment(project_root, raw_root, args.python, dry_run=args.dry_run)
    build_oras5_2d_raw(raw_root, overwrite=args.overwrite, dry_run=args.dry_run)
    build_oras5_3d_surface(project_root, raw_root, args.python, overwrite=args.overwrite, dry_run=args.dry_run)
    merge_oras5_raw(raw_root, overwrite=args.overwrite, dry_run=args.dry_run)
    build_anomaly(project_root, raw_root, interim_root, args.python, overwrite=args.overwrite, dry_run=args.dry_run)
    build_final_splits(project_root, data_root, args.python, dry_run=args.dry_run)
    if not args.dry_run and not args.skip_verify:
        verify_outputs(data_root, raw_root)


if __name__ == "__main__":
    main()
