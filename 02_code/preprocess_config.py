from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path("/mnt/d/project/01_ENSO")
RAW_ROOT = PROJECT_ROOT / "01_data/01_raw"
INTERIM_ROOT = PROJECT_ROOT / "01_data/02_interim"
INPUT_ROOT = PROJECT_ROOT / "01_data/03_input"
TARGET_ROOT = PROJECT_ROOT / "01_data/04_target"

LAT_CANDIDATES = ["lat", "latitude", "nav_lat", "y"]
LON_CANDIDATES = ["lon", "longitude", "nav_lon", "x"]
TIME_CANDIDATES = ["time", "time_counter", "t", "valid_time"]

VAR_ORDER = [
    "tos",
    "ohc300",
    "mlotst",
    "sos",
    "psl",
    "uos",
    "vos",
    "uas",
    "uasmax",
    "vas",
]
OCEAN_VARS = frozenset({"tos", "ohc300", "mlotst", "sos", "uos", "vos"})
ATM_VARS = frozenset({"psl", "uas", "uasmax", "vas"})

VALIDATION_START = "1900-01"
VALIDATION_END = "1978-12"
TEST_START = "1980-01"
TEST_END = "2025-12"

GODAS_RANGE_TAG = "198001-202512"
ORAS5_RANGE_TAG = "195801-197812"
ERA5_RANGE_TAG = "195801-202512"

GODAS_RAW_FILE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "pottmp": (f"pottmp.{GODAS_RANGE_TAG}.nc", "pottmp.1980-2025.nc"),
    "salt": (f"salt.{GODAS_RANGE_TAG}.nc", "salt.1980-2025.nc"),
    "ucur": (f"ucur.{GODAS_RANGE_TAG}.nc", "ucur.1980-2025.nc"),
    "vcur": (f"vcur.{GODAS_RANGE_TAG}.nc", "vcur.1980-2025.nc"),
}

GODAS_SURFACE_FILES: dict[str, str] = {
    "tos": f"tos.{GODAS_RANGE_TAG}.nc",
    "sos": f"sos.{GODAS_RANGE_TAG}.nc",
    "uos": f"uos.{GODAS_RANGE_TAG}.nc",
    "vos": f"vos.{GODAS_RANGE_TAG}.nc",
}

ORAS5_SURFACE_FILES: dict[str, str] = {
    "mlotst": f"mlotst.{ORAS5_RANGE_TAG}.nc",
    "ohc300": f"ohc300.{ORAS5_RANGE_TAG}.nc",
    "sos": f"sos.{ORAS5_RANGE_TAG}.nc",
    "tos": f"tos.{ORAS5_RANGE_TAG}.nc",
    "uos": f"uos.{ORAS5_RANGE_TAG}.nc",
    "vos": f"vos.{ORAS5_RANGE_TAG}.nc",
}

ERA5_ANOM_FILES: dict[str, str] = {
    "psl": f"psl.mon.{ERA5_RANGE_TAG}.anom_1x2.nc",
    "uas": f"uas.mon.{ERA5_RANGE_TAG}.anom_1x2.nc",
    "uasmax": f"uasmax.mon.{ERA5_RANGE_TAG}.anom_1x2.nc",
    "vas": f"vas.mon.{ERA5_RANGE_TAG}.anom_1x2.nc",
}

ORAS5_ANOM_FILES: dict[str, str] = {
    var: f"{name}.anom_1x2.nc" for var, name in ORAS5_SURFACE_FILES.items()
}
GODAS_ANOM_FILES: dict[str, str] = {
    var: f"{name}.anom_1x2.nc" for var, name in GODAS_SURFACE_FILES.items()
}
SODA_ANOM_FILES: dict[str, str] = {
    "mlotst": "mlotst.SODA_mlotst.anom_1x2.nc",
    "ohc300": "ohc300.SODA_ohc300.anom_1x2.nc",
    "sos": "sos.SODA_sos.anom_1x2.nc",
    "tos": "tos.SODA_tos.anom_1x2.nc",
    "uos": "uos.SODA_uo.anom_1x2.nc",
    "vos": "vos.SODA_vo.anom_1x2.nc",
}
CR20V2_ANOM_FILES: dict[str, str] = {
    "psl": "psl.prmsl.mon.mean.anom_1x2.nc",
    "uas": "uas.uwnd.10m.mon.mean.anom_1x2.nc",
    "uasmax": "uasmax.uwnd.10m.1871-2012.anom_1x2.nc",
    "vas": "vas.vwnd.10m.mon.mean.anom_1x2.nc",
}

_YYYYMM_SPAN_RE = re.compile(r"\d{6}-\d{6}")
_YYYY_SPAN_RE = re.compile(r"\d{4}-\d{4}")


def is_anomaly_file(path: Path) -> bool:
    return path.name.endswith(".anom_1x2.nc")


def candidate_priority(path: Path) -> tuple[int, int, int, str]:
    name = path.name
    return (
        0 if is_anomaly_file(path) else 1,
        0 if _YYYYMM_SPAN_RE.search(name) else 1,
        0 if _YYYY_SPAN_RE.search(name) else 1,
        name,
    )


def sort_preferred_paths(paths: Iterable[Path]) -> list[Path]:
    return sorted(paths, key=candidate_priority)


def select_preferred_path(paths: Sequence[Path]) -> Path | None:
    if not paths:
        return None
    return sort_preferred_paths(paths)[0]


def resolve_existing_path(root: Path, candidates: Sequence[str]) -> Path | None:
    for name in candidates:
        path = root / name
        if path.exists():
            return path
    return None


def require_existing_path(root: Path, candidates: Sequence[str], label: str) -> Path:
    path = resolve_existing_path(root, candidates)
    if path is None:
        joined = ", ".join(str(root / name) for name in candidates)
        raise FileNotFoundError(f"{label} not found; tried: {joined}")
    return path


def default_validation_target_candidates() -> list[Path]:
    return [INTERIM_ROOT / "soda" / SODA_ANOM_FILES["tos"]]


def default_test_target_candidates() -> list[Path]:
    return [
        INTERIM_ROOT / "godas" / GODAS_ANOM_FILES["tos"],
        INTERIM_ROOT / "godas" / "tos.1980-2025.anom_1x2.nc",
    ]
