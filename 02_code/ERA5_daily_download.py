#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import shutil
import zipfile
from pathlib import Path

import cdsapi

# =========================
# 사용자 설정
# =========================
DATASET = "derived-era5-single-levels-daily-statistics"

VARIABLES = [
    "10m_u_component_of_wind",
]

YEARS = range(1958, 2026)   # 1958 ~ 2025
MONTHS = range(1, 13)
DAYS = [f"{d:02d}" for d in range(1, 32)]

OUTDIR = Path("/mnt/d/project/01_ENSO/01_data/01_raw/era5/daily")
OUTDIR.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 6
RETRY_WAIT_SEC = 15
REQUEST_GAP_SEC = 2

# True  -> OUTDIR/YYYYMM/*.nc
# False -> OUTDIR/*.nc
USE_MONTH_SUBDIR = True


# =========================
# 유틸
# =========================
def detect_file_type(path: Path) -> str:
    """
    파일 헤더를 보고 zip / netcdf3 / netcdf4(hdf5) / unknown 판별
    """
    with open(path, "rb") as f:
        head = f.read(16)

    # zip
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06") or head.startswith(b"PK\x07\x08"):
        return "zip"

    # classic netCDF / 64-bit offset netCDF
    if head.startswith(b"CDF\x01") or head.startswith(b"CDF\x02") or head.startswith(b"CDF\x05"):
        return "netcdf"

    # netCDF4 is HDF5 container
    if head.startswith(b"\x89HDF\r\n\x1a\n"):
        return "netcdf"

    return "unknown"


def extract_zip_keep_only_nc(zip_path: Path, dest_dir: Path):
    """
    zip 파일을 dest_dir에 압축해제하고 .nc만 남긴다.
    zip은 마지막에 삭제.
    """
    extracted_nc = []

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
    return extracted_nc


def move_single_netcdf(src_path: Path, dest_dir: Path, yyyymm: str):
    """
    단일 netCDF 파일이면 목적지로 옮겨 .nc로 저장
    """
    target = dest_dir / f"ERA5_daily_{yyyymm}.nc"
    os.replace(src_path, target)
    return [target]


def expected_nc_exists(dest_dir: Path) -> bool:
    return any(dest_dir.glob("*.nc"))


def show_unknown_file_preview(path: Path, n=300):
    try:
        with open(path, "rb") as f:
            raw = f.read(n)
        try:
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return repr(raw)
    except Exception as e:
        return f"<cannot read file preview: {e}>"


# =========================
# 메인
# =========================
def main():
    client = cdsapi.Client()

    for year in YEARS:
        for month in MONTHS:
            yyyymm = f"{year}{month:02d}"
            month_dir = OUTDIR / yyyymm if USE_MONTH_SUBDIR else OUTDIR
            month_dir.mkdir(parents=True, exist_ok=True)

            if expected_nc_exists(month_dir):
                print(f"[SKIP] {yyyymm}: nc file(s) already exist in {month_dir}")
                continue

            tmp_path = OUTDIR / f"{yyyymm}.download.part"

            request = {
                "product_type": "reanalysis",
                "variable": VARIABLES,
                "year": f"{year}",
                "month": [f"{month:02d}"],
                "day": DAYS,
                "daily_statistic": "daily_mean",
                "time_zone": "utc+00:00",
                "frequency": "3_hourly",
                "area": [60, -180, -60, 180],
            }

            success = False
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    print(f"[GET ] {yyyymm} (attempt {attempt}/{MAX_RETRIES})")
                    if tmp_path.exists():
                        tmp_path.unlink()

                    client.retrieve(DATASET, request).download(str(tmp_path))

                    ftype = detect_file_type(tmp_path)

                    if ftype == "zip":
                        extracted_nc = extract_zip_keep_only_nc(tmp_path, month_dir)
                        if extracted_nc:
                            print(f"[DONE] {yyyymm}: extracted {len(extracted_nc)} nc file(s)")
                            for f in extracted_nc:
                                print(f"       - {f}")
                        else:
                            raise RuntimeError("zip downloaded but no .nc file found inside")

                    elif ftype == "netcdf":
                        saved = move_single_netcdf(tmp_path, month_dir, yyyymm)
                        print(f"[DONE] {yyyymm}: saved netCDF directly")
                        for f in saved:
                            print(f"       - {f}")

                    else:
                        preview = show_unknown_file_preview(tmp_path, n=500)
                        raise RuntimeError(
                            "downloaded file is neither zip nor netCDF.\n"
                            f"Preview:\n{preview}"
                        )

                    success = True
                    break

                except Exception as e:
                    print(f"[ERR ] {yyyymm} attempt {attempt}: {e}")

                    if tmp_path.exists():
                        tmp_path.unlink(missing_ok=True)

                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_WAIT_SEC)

            if not success:
                print(f"[FAIL] {yyyymm}: exceeded retries")

            time.sleep(REQUEST_GAP_SEC)


if __name__ == "__main__":
    main()