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

# 결과를 항상 OUTDIR/ERA5_daily_YYYYMM.nc 형태로 저장
OUTDIR = Path("/mnt/d/project/01_ENSO/01_data/01_raw/era5/daily")
OUTDIR.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 6
RETRY_WAIT_SEC = 15
REQUEST_GAP_SEC = 2


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


def target_nc_path(yyyymm: str) -> Path:
    return OUTDIR / f"ERA5_daily_{yyyymm}.nc"


def expected_nc_exists(yyyymm: str) -> bool:
    """
    해당 월의 기대 출력 파일이 이미 있는지 확인
    """
    return target_nc_path(yyyymm).exists()


def extract_zip_to_single_target(zip_path: Path, target: Path) -> Path:
    """
    zip 파일을 임시 디렉토리에 풀고,
    내부의 .nc가 정확히 1개일 때 target 이름으로 저장한다.
    """
    tmp_extract_dir = OUTDIR / f".tmp_extract_{target.stem}"
    tmp_extract_dir.mkdir(parents=True, exist_ok=True)

    extracted_nc = []
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
    """
    단일 netCDF 파일이면 target 이름으로 저장
    """
    os.replace(src_path, target)
    return target


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
            target_nc = target_nc_path(yyyymm)

            if expected_nc_exists(yyyymm):
                print(f"[SKIP] {yyyymm}: already exists -> {target_nc}")
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
                        saved = extract_zip_to_single_target(tmp_path, target_nc)
                        print(f"[DONE] {yyyymm}: extracted zip -> {saved}")

                    elif ftype == "netcdf":
                        saved = move_single_netcdf(tmp_path, target_nc)
                        print(f"[DONE] {yyyymm}: saved netCDF directly -> {saved}")

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

                    if target_nc.exists():
                        target_nc.unlink(missing_ok=True)

                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_WAIT_SEC)

            if not success:
                print(f"[FAIL] {yyyymm}: exceeded retries")

            time.sleep(REQUEST_GAP_SEC)


if __name__ == "__main__":
    main()