#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
import time
import zipfile
from pathlib import Path

import cdsapi

# =========================
# 사용자 설정
# =========================
DATASET = "reanalysis-oras5"

VARIABLES = [
    "mixed_layer_depth_0_03",
    "ocean_heat_content_for_the_upper_300m",
    "sea_surface_salinity",
    "sea_surface_temperature",
]

YEARS = list(range(1958, 1979))  # 1958~1978
MONTHS = [f"{m:02d}" for m in range(1, 13)]

OUTDIR = Path("/mnt/d/project/01_ENSO/01_data/01_raw/oras5")
OUTDIR.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 6
RETRY_WAIT_SEC = 10
REQUEST_GAP_SEC = 1

EXPECTED_MONTHLY_FILES = 12


# =========================
# 유틸 함수
# =========================
def safe_unlink(path: Path):
    if path.exists():
        path.unlink()


def move_to_bad(path: Path) -> Path:
    bad = path.with_name(f"{path.name}.bad_{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.move(str(path), str(bad))
    return bad


def is_valid_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if not zipfile.is_zipfile(path):
        return False

    try:
        with zipfile.ZipFile(path, "r") as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                print(f"[ERR ] zip CRC 오류: {bad_member}")
                return False
            return True
    except Exception as e:
        print(f"[ERR ] zip 검증 실패: {e}")
        return False


def year_nc_files(work_dir: Path, year: int):
    ystr = str(year)
    return sorted([
        p for p in work_dir.rglob("*.nc")
        if ystr in p.name
    ])


def remove_year_nc_files(work_dir: Path, year: int):
    files = year_nc_files(work_dir, year)
    for p in files:
        try:
            p.unlink()
        except Exception:
            pass


def remove_empty_dirs(base_dir: Path):
    for d in sorted([p for p in base_dir.rglob("*") if p.is_dir()], reverse=True):
        try:
            if not any(d.iterdir()):
                d.rmdir()
        except Exception:
            pass


def flatten_nc_files(work_dir: Path):
    """
    zip 해제 후 하위 폴더에 들어간 nc가 있으면 work_dir 바로 아래로 이동
    """
    nc_files = sorted([p for p in work_dir.rglob("*.nc") if p.is_file()])

    for src in nc_files:
        dst = work_dir / src.name
        if src.resolve() == dst.resolve():
            continue

        if dst.exists():
            # 같은 이름 파일이 이미 있으면 원본 유지, 새 파일은 삭제
            # 정상 상황에서는 거의 발생하지 않음
            try:
                src.unlink()
            except Exception:
                pass
        else:
            shutil.move(str(src), str(dst))

    remove_empty_dirs(work_dir)


def extract_zip_keep_monthly_nc(zip_path: Path, outdir: Path) -> list[Path]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        if not members:
            raise RuntimeError("zip 내부가 비어 있습니다.")
        zf.extractall(outdir)

    flatten_nc_files(outdir)

    extracted_nc = sorted([p for p in outdir.glob("*.nc") if p.is_file()])
    if not extracted_nc:
        raise RuntimeError("압축 해제 후 .nc 파일을 찾지 못했습니다.")

    return extracted_nc


def build_request(var: str, year: int) -> dict:
    return {
        "product_type": ["consolidated"],
        "vertical_resolution": "single_level",
        "variable": [var],
        "year": [str(year)],
        "month": MONTHS,
    }


def retrieve_one(client: cdsapi.Client, var: str, year: int, outroot: Path):
    work_dir = outroot / var
    work_dir.mkdir(parents=True, exist_ok=True)

    zip_path = work_dir / f"{var}_{year}.zip"
    part_path = work_dir / f"{var}_{year}.zip.part"

    existing = year_nc_files(work_dir, year)
    if len(existing) >= EXPECTED_MONTHLY_FILES:
        print(f"[SKIP] {var} {year}: monthly nc already exists ({len(existing)} files)")
        return
    elif 0 < len(existing) < EXPECTED_MONTHLY_FILES:
        print(f"[CLEAN] {var} {year}: partial monthly nc found ({len(existing)} files), remove and redownload")
        remove_year_nc_files(work_dir, year)

    request = build_request(var, year)
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            safe_unlink(part_path)

            print(f"[GET ] {var} {year} (try {attempt}/{MAX_RETRIES})")
            client.retrieve(DATASET, request).download(str(part_path))

            if not is_valid_zip(part_path):
                bad = move_to_bad(part_path)
                raise RuntimeError(f"다운로드 후 zip 검증 실패 -> {bad.name}")

            if zip_path.exists():
                zip_path.unlink()
            part_path.rename(zip_path)

            print(f"[UNZIP] {zip_path.name}")
            extracted = extract_zip_keep_monthly_nc(zip_path, work_dir)

            year_files = year_nc_files(work_dir, year)
            print(f"[INFO] {var} {year}: extracted monthly nc = {len(year_files)} files")
            print(f"[INFO] filenames: {[p.name for p in year_files]}")

            if len(year_files) < EXPECTED_MONTHLY_FILES:
                raise RuntimeError(
                    f"월별 nc 개수 부족: expected={EXPECTED_MONTHLY_FILES}, got={len(year_files)}"
                )

            zip_path.unlink()
            print(f"[DONE] {var} {year} -> monthly nc files kept in {work_dir}")

            time.sleep(REQUEST_GAP_SEC)
            return

        except Exception as e:
            last_err = e
            print(f"[WARN] {var} {year} 실패: {e}")

            if part_path.exists():
                try:
                    bad = move_to_bad(part_path)
                    print(f"[INFO] 손상/미완료 파일 이동: {bad.name}")
                except Exception:
                    pass

            if zip_path.exists() and not is_valid_zip(zip_path):
                try:
                    bad = move_to_bad(zip_path)
                    print(f"[INFO] 손상 zip 이동: {bad.name}")
                except Exception:
                    pass

            # 실패 시 불완전 월별 nc 제거
            remove_year_nc_files(work_dir, year)
            remove_empty_dirs(work_dir)

            if attempt < MAX_RETRIES:
                print(f"[WAIT] {RETRY_WAIT_SEC}s 후 재시도")
                time.sleep(RETRY_WAIT_SEC)
            else:
                print(f"[FAIL] {var} {year}: 최종 실패")

    raise RuntimeError(f"{var} {year} 다운로드 실패: {last_err}")


def main():
    client = cdsapi.Client()
    failed = []

    for var in VARIABLES:
        for year in YEARS:
            try:
                retrieve_one(client, var, year, OUTDIR)
            except Exception as e:
                failed.append((var, year, str(e)))

    print("\n====================")
    print("작업 완료")
    print("====================")
    if failed:
        print("[SUMMARY] 실패 목록")
        for var, year, err in failed:
            print(f" - {var} {year}: {err}")
    else:
        print("[SUMMARY] 모든 다운로드 성공")


if __name__ == "__main__":
    main()