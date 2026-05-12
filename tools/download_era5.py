#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import calendar
import os
import shutil
import time
import zipfile
from pathlib import Path

import cdsapi

DATASET_MONTHLY = "reanalysis-era5-single-levels-monthly-means"
DATASET_DAILY = "derived-era5-single-levels-daily-statistics"

DEFAULT_MONTHLY_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]

DEFAULT_DAILY_VARIABLES = [
    "10m_u_component_of_wind",
]

DEFAULT_OUTROOT = Path("/mnt/d/project/UmaxFormer/data/raw/era5")
DEFAULT_MONTHS = list(range(1, 13))

MAX_RETRIES = 6
RETRY_WAIT_SEC = 15
REQUEST_GAP_SEC = 2


def detect_file_type(path: Path) -> str:
    """
    ?뚯씪 ?ㅻ뜑瑜?蹂닿퀬 zip / netcdf / unknown ?먮퀎
    """
    with open(path, "rb") as f:
        head = f.read(16)

    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06") or head.startswith(b"PK\x07\x08"):
        return "zip"

    if head.startswith(b"CDF\x01") or head.startswith(b"CDF\x02") or head.startswith(b"CDF\x05"):
        return "netcdf"

    if head.startswith(b"\x89HDF\r\n\x1a\n"):
        return "netcdf"

    return "unknown"


def extract_zip_keep_only_nc(zip_path: Path, dest_dir: Path, base_name: str) -> list[Path]:
    """
    zip???怨?nc留??④릿??
    ?대? nc媛 1媛쒕㈃ base_name?쇰줈 ?대쫫 蹂寃?
    ?щ윭 媛쒕㈃ ?먮옒 ?대쫫 ?좎?.
    """
    extracted_nc: list[Path] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue

            member_name = Path(member.filename).name
            out_path = dest_dir / member_name

            with zf.open(member) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

            if out_path.suffix.lower() == ".nc":
                extracted_nc.append(out_path)
            else:
                out_path.unlink(missing_ok=True)

    zip_path.unlink(missing_ok=True)

    if len(extracted_nc) == 1:
        renamed = dest_dir / f"{base_name}.nc"
        if extracted_nc[0] != renamed:
            extracted_nc[0].replace(renamed)
        extracted_nc = [renamed]

    return extracted_nc


def extract_zip_to_single_target(zip_path: Path, target: Path, work_dir: Path) -> Path:
    """
    zip???꾩떆 ?붾젆?좊━???怨?
    ?대? .nc媛 ?뺥솗??1媛쒖씪 ??target ?대쫫?쇰줈 ??ν븳??
    """
    tmp_extract_dir = work_dir / f".tmp_extract_{target.stem}"
    tmp_extract_dir.mkdir(parents=True, exist_ok=True)

    extracted_nc: list[Path] = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue

                member_name = Path(member.filename).name
                out_path = tmp_extract_dir / member_name

                with zf.open(member) as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

                if out_path.suffix.lower() == ".nc":
                    extracted_nc.append(out_path)
                else:
                    out_path.unlink(missing_ok=True)

        if len(extracted_nc) == 0:
            raise RuntimeError("zip downloaded but no .nc file found inside")

        if len(extracted_nc) > 1:
            names = ", ".join(p.name for p in extracted_nc)
            raise RuntimeError(
                "zip contains multiple .nc files; expected exactly one monthly file.\n"
                f"Found: {names}"
            )

        os.replace(extracted_nc[0], target)
        return target

    finally:
        zip_path.unlink(missing_ok=True)
        shutil.rmtree(tmp_extract_dir, ignore_errors=True)


def move_single_netcdf(src_path: Path, target: Path) -> Path:
    os.replace(src_path, target)
    return target


def move_single_netcdf_to_dir(src_path: Path, dest_dir: Path, base_name: str) -> list[Path]:
    target = dest_dir / f"{base_name}.nc"
    os.replace(src_path, target)
    return [target]


def show_unknown_file_preview(path: Path, n: int = 400) -> str:
    try:
        with open(path, "rb") as f:
            raw = f.read(n)
        return raw.decode("utf-8", errors="replace")
    except Exception as e:
        return f"<cannot read preview: {e}>"


def normalize_months(months: list[int]) -> list[int]:
    out = sorted(set(months))
    for m in out:
        if m < 1 or m > 12:
            raise ValueError(f"invalid month: {m}")
    return out


def build_monthly_base_name(var: str, years: list[int], months: list[int]) -> str:
    start_tag = f"{years[0]}{months[0]:02d}"
    end_tag = f"{years[-1]}{months[-1]:02d}"
    return f"ERA5_monthly_{var}_{start_tag}_{end_tag}"


def target_daily_nc_path(outdir: Path, yyyymm: str, var: str, include_var_name: bool) -> Path:
    if include_var_name:
        return outdir / f"ERA5_daily_{var}_{yyyymm}.nc"
    return outdir / f"ERA5_daily_{yyyymm}.nc"


def build_monthly_request(var: str, years: list[int], months: list[int]) -> dict:
    return {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [var],
        "year": [str(y) for y in years],
        "month": [f"{m:02d}" for m in months],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [60, -180, -60, 180],
    }


def build_daily_request(var: str, year: int, month: int) -> dict:
    last_day = calendar.monthrange(year, month)[1]
    days = [f"{d:02d}" for d in range(1, last_day + 1)]

    return {
        "product_type": "reanalysis",
        "variable": [var],
        "year": f"{year}",
        "month": [f"{month:02d}"],
        "day": days,
        "daily_statistic": "daily_mean",
        "time_zone": "utc+00:00",
        "frequency": "3_hourly",
        "area": [60, -180, -60, 180],
    }


def download_monthly(
    client: cdsapi.Client,
    outdir: Path,
    years: list[int],
    months: list[int],
    variables: list[str],
    max_retries: int,
    retry_wait_sec: int,
    request_gap_sec: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    for var in variables:
        base_name = build_monthly_base_name(var, years, months)
        final_nc = outdir / f"{base_name}.nc"
        tmp_path = outdir / f"{base_name}.part"

        if final_nc.exists() and final_nc.stat().st_size > 0:
            print(f"[SKIP][monthly] {final_nc.name}")
            continue

        request = build_monthly_request(var, years, months)

        success = False
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[GET ][monthly] {var} (attempt {attempt}/{max_retries})")

                if tmp_path.exists():
                    tmp_path.unlink()

                client.retrieve(DATASET_MONTHLY, request).download(str(tmp_path))
                ftype = detect_file_type(tmp_path)

                if ftype == "netcdf":
                    saved = move_single_netcdf_to_dir(tmp_path, outdir, base_name)
                    print("[DONE][monthly] saved netCDF directly")
                    for f in saved:
                        print(f"                - {f}")

                elif ftype == "zip":
                    saved = extract_zip_keep_only_nc(tmp_path, outdir, base_name)
                    if not saved:
                        raise RuntimeError("zip downloaded but no .nc found inside")
                    print(f"[DONE][monthly] extracted {len(saved)} nc file(s)")
                    for f in saved:
                        print(f"                - {f}")

                else:
                    preview = show_unknown_file_preview(tmp_path, n=500)
                    raise RuntimeError(
                        "downloaded file is neither zip nor netCDF.\n"
                        f"Preview:\n{preview}"
                    )

                success = True
                break

            except Exception as e:
                print(f"[ERR ][monthly] {var} attempt {attempt}: {e}")
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)

                if attempt < max_retries:
                    time.sleep(retry_wait_sec)

        if not success:
            print(f"[FAIL][monthly] {var}: exceeded retries")

        time.sleep(request_gap_sec)


def download_daily(
    client: cdsapi.Client,
    outdir: Path,
    years: list[int],
    months: list[int],
    variables: list[str],
    max_retries: int,
    retry_wait_sec: int,
    request_gap_sec: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    include_var_name = len(variables) > 1

    for var in variables:
        for year in years:
            for month in months:
                yyyymm = f"{year}{month:02d}"
                target_nc = target_daily_nc_path(outdir, yyyymm, var, include_var_name)
                tmp_path = outdir / f"{target_nc.stem}.download.part"

                if target_nc.exists() and target_nc.stat().st_size > 0:
                    print(f"[SKIP][daily] {target_nc.name}")
                    continue

                request = build_daily_request(var, year, month)

                success = False
                for attempt in range(1, max_retries + 1):
                    try:
                        print(
                            f"[GET ][daily] {var} {yyyymm} "
                            f"(attempt {attempt}/{max_retries})"
                        )

                        if tmp_path.exists():
                            tmp_path.unlink()

                        client.retrieve(DATASET_DAILY, request).download(str(tmp_path))
                        ftype = detect_file_type(tmp_path)

                        if ftype == "zip":
                            saved = extract_zip_to_single_target(tmp_path, target_nc, outdir)
                            print(f"[DONE][daily] {yyyymm}: extracted zip -> {saved}")

                        elif ftype == "netcdf":
                            saved = move_single_netcdf(tmp_path, target_nc)
                            print(f"[DONE][daily] {yyyymm}: saved netCDF directly -> {saved}")

                        else:
                            preview = show_unknown_file_preview(tmp_path, n=500)
                            raise RuntimeError(
                                "downloaded file is neither zip nor netCDF.\n"
                                f"Preview:\n{preview}"
                            )

                        success = True
                        break

                    except Exception as e:
                        print(f"[ERR ][daily] {var} {yyyymm} attempt {attempt}: {e}")

                        if tmp_path.exists():
                            tmp_path.unlink(missing_ok=True)

                        if target_nc.exists():
                            target_nc.unlink(missing_ok=True)

                        if attempt < max_retries:
                            time.sleep(retry_wait_sec)

                if not success:
                    print(f"[FAIL][daily] {var} {yyyymm}: exceeded retries")

                time.sleep(request_gap_sec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified ERA5 downloader for daily statistics and monthly means"
    )
    parser.add_argument(
        "--freq",
        choices=["daily", "monthly", "all"],
        default="all",
        help="what to download",
    )
    parser.add_argument("--start-year", type=int, default=1958)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--months",
        type=int,
        nargs="*",
        default=DEFAULT_MONTHS,
        help="months to download, e.g. --months 1 2 3",
    )
    parser.add_argument(
        "--outroot",
        type=Path,
        default=DEFAULT_OUTROOT,
        help="root directory; daily/monthly subdirectories are created below this",
    )
    parser.add_argument(
        "--daily-vars",
        nargs="*",
        default=DEFAULT_DAILY_VARIABLES,
        help="variables for daily statistics request",
    )
    parser.add_argument(
        "--monthly-vars",
        nargs="*",
        default=DEFAULT_MONTHLY_VARIABLES,
        help="variables for monthly means request",
    )
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--retry-wait-sec", type=int, default=RETRY_WAIT_SEC)
    parser.add_argument("--request-gap-sec", type=int, default=REQUEST_GAP_SEC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.start_year > args.end_year:
        raise ValueError("start-year must be <= end-year")

    years = list(range(args.start_year, args.end_year + 1))
    months = normalize_months(args.months)

    daily_outdir = args.outroot / "daily"
    monthly_outdir = args.outroot / "monthly"
    daily_outdir.mkdir(parents=True, exist_ok=True)
    monthly_outdir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()

    if args.freq in ("monthly", "all"):
        download_monthly(
            client=client,
            outdir=monthly_outdir,
            years=years,
            months=months,
            variables=args.monthly_vars,
            max_retries=args.max_retries,
            retry_wait_sec=args.retry_wait_sec,
            request_gap_sec=args.request_gap_sec,
        )

    if args.freq in ("daily", "all"):
        download_daily(
            client=client,
            outdir=daily_outdir,
            years=years,
            months=months,
            variables=args.daily_vars,
            max_retries=args.max_retries,
            retry_wait_sec=args.retry_wait_sec,
            request_gap_sec=args.request_gap_sec,
        )


if __name__ == "__main__":
    main()
