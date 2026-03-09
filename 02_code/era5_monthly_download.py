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
DATASET = "reanalysis-era5-single-levels-monthly-means"

VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]

YEARS = [str(y) for y in range(1958, 2026)]   # 1958 ~ 2025
MONTHS = [f"{m:02d}" for m in range(1, 13)]

OUTDIR = Path("/mnt/d/project/01_ENSO/01_data/01_raw/era5/monthly")
OUTDIR.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 6
RETRY_WAIT_SEC = 15
REQUEST_GAP_SEC = 2


# =========================
# 유틸
# =========================
def detect_file_type(path: Path) -> str:
    """
    파일 헤더를 보고 zip / netcdf / unknown 판별
    """
    with open(path, "rb") as f:
        head = f.read(16)

    # zip
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06") or head.startswith(b"PK\x07\x08"):
        return "zip"

    # classic netCDF / 64-bit offset / CDF5
    if head.startswith(b"CDF\x01") or head.startswith(b"CDF\x02") or head.startswith(b"CDF\x05"):
        return "netcdf"

    # netCDF4(HDF5)
    if head.startswith(b"\x89HDF\r\n\x1a\n"):
        return "netcdf"

    return "unknown"


def extract_zip_keep_only_nc(zip_path: Path, dest_dir: Path, base_name: str):
    """
    zip을 풀고 nc만 남긴다. zip 내부 nc가 1개면 base_name으로 이름 변경.
    여러 개면 원래 이름 유지.
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

    if len(extracted_nc) == 1:
        renamed = dest_dir / f"{base_name}.nc"
        if extracted_nc[0] != renamed:
            extracted_nc[0].replace(renamed)
        extracted_nc = [renamed]

    return extracted_nc


def move_single_netcdf(src_path: Path, dest_dir: Path, base_name: str):
    target = dest_dir / f"{base_name}.nc"
    os.replace(src_path, target)
    return [target]


def show_unknown_file_preview(path: Path, n=400):
    try:
        with open(path, "rb") as f:
            raw = f.read(n)
        return raw.decode("utf-8", errors="replace")
    except Exception as e:
        return f"<cannot read preview: {e}>"


# =========================
# 메인
# =========================
def main():
    client = cdsapi.Client()

    for var in VARIABLES:
        base_name = f"ERA5_monthly_{var}_195801_202512"
        final_nc = OUTDIR / f"{base_name}.nc"
        tmp_path = OUTDIR / f"{base_name}.part"

        if final_nc.exists() and final_nc.stat().st_size > 0:
            print(f"[SKIP] {final_nc.name}")
            continue

        request = {
            "product_type": ["monthly_averaged_reanalysis"],
            "variable": [var],
            "year": YEARS,
            "month": MONTHS,
            "time": ["00:00"],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [60, -180, -60, 180],
        }

        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"[GET ] {var} (attempt {attempt}/{MAX_RETRIES})")

                if tmp_path.exists():
                    tmp_path.unlink()

                client.retrieve(DATASET, request).download(str(tmp_path))

                ftype = detect_file_type(tmp_path)

                if ftype == "netcdf":
                    saved = move_single_netcdf(tmp_path, OUTDIR, base_name)
                    print(f"[DONE] saved netCDF directly")
                    for f in saved:
                        print(f"       - {f}")

                elif ftype == "zip":
                    saved = extract_zip_keep_only_nc(tmp_path, OUTDIR, base_name)
                    if not saved:
                        raise RuntimeError("zip downloaded but no .nc found inside")
                    print(f"[DONE] extracted {len(saved)} nc file(s)")
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
                print(f"[ERR ] {var} attempt {attempt}: {e}")
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)

                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_WAIT_SEC)

        if not success:
            print(f"[FAIL] {var}: exceeded retries")

        time.sleep(REQUEST_GAP_SEC)


if __name__ == "__main__":
    main()