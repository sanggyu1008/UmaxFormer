#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# 고정 설정
# ============================================================
OUTPUT_DIR = Path("/mnt/d/project/01_ENSO/03_output")
DEFAULT_ROOTS = {
    "cmip6": Path("/mnt/d/project/01_ENSO/01_data/01_raw/cmip6"),
    "era5": Path("/mnt/d/project/01_ENSO/01_data/01_raw/era5"),
    "godas": Path("/mnt/d/project/01_ENSO/01_data/01_raw/godas"),
    "oras5": Path("/mnt/d/project/01_ENSO/01_data/01_raw/oras5"),
}

ERA5_TARGETS = [
    {
        "label": "uas_day",
        "var": "uas",
        "expected_freq": "daily",
        "exact_name": "uas.day.195801-202512.nc",
        "glob_pattern": "uas.day.*.nc",
    },
    {
        "label": "uas_mon",
        "var": "uas",
        "expected_freq": "monthly",
        "exact_name": "uas.mon.195801-202512.nc",
        "glob_pattern": "uas.mon.*.nc",
    },
    {
        "label": "vas_mon",
        "var": "vas",
        "expected_freq": "monthly",
        "exact_name": "vas.mon.195801-202512.nc",
        "glob_pattern": "vas.mon.*.nc",
    },
    {
        "label": "psl_mon",
        "var": "psl",
        "expected_freq": "monthly",
        "exact_name": "psl.mon.195801-202512.nc",
        "glob_pattern": "psl.mon.*.nc",
    },
]

ATMOS_VARS = {"psl", "uas", "vas"}
OCEAN_VARS = {"mlotst", "ohc300", "sos", "tos", "uos", "vos"}
GODAS_EXCLUDE_BASENAME = "pottmp.198001-202512.nc"
TIME_BOUNDS_CANDIDATES = ("time_bnds", "time_bounds", "time_bound", "bounds_time")
COORD_LIKE = {
    "time", "time_bnds", "time_bounds", "lat", "lon", "latitude", "longitude",
    "lev", "olevel", "depth", "deptht", "depthu", "depthv", "depthw", "bnds",
    "bounds", "vertices_latitude", "vertices_longitude", "member_id", "height",
    "type", "axis_nbounds",
}
UNIT_RE = re.compile(r"^\s*([A-Za-z_]+)\s+since\s+(.+?)\s*$")
REF_RE = re.compile(
    r"^\s*(?P<year>[+-]?\d{1,6})-(?P<month>\d{1,2})-(?P<day>\d{1,2})"
    r"(?:[ T](?P<hour>\d{1,2})(?::(?P<minute>\d{1,2})(?::(?P<second>\d{1,2}(?:\.\d+)?))?)?)?"
    r"(?:\s*(?:Z|UTC|[+-]\d{2}:?\d{2}))?\s*$"
)
MONTH_OFFSETS_NOLEAP = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=np.int64)
MONTH_OFFSETS_LEAP = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335], dtype=np.int64)


# ============================================================
# 공통 유틸
# ============================================================
def safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def basename_only(path_str: str | None) -> str:
    if not path_str:
        return ""
    return Path(path_str).name


def safe_strings(values: Iterable[Any], limit: int = 10) -> list[str]:
    out: list[str] = []
    for x in values:
        out.append(str(x))
        if len(out) >= limit:
            break
    return out


def normalize_calendar(calendar: str | None) -> str:
    if not calendar:
        return "standard"
    c = str(calendar).strip().lower()
    mapping = {
        "gregorian": "standard",
        "standard": "standard",
        "proleptic_gregorian": "proleptic_gregorian",
        "noleap": "noleap",
        "365_day": "noleap",
        "365": "noleap",
        "all_leap": "all_leap",
        "366_day": "all_leap",
        "366": "all_leap",
        "360_day": "360_day",
        "julian": "julian",
    }
    return mapping.get(c, c)


def normalize_unit(unit: str) -> str:
    u = unit.strip().lower()
    mapping = {
        "day": "days",
        "days": "days",
        "d": "days",
        "hour": "hours",
        "hours": "hours",
        "hr": "hours",
        "hrs": "hours",
        "h": "hours",
        "minute": "minutes",
        "minutes": "minutes",
        "min": "minutes",
        "mins": "minutes",
        "second": "seconds",
        "seconds": "seconds",
        "sec": "seconds",
        "secs": "seconds",
        "s": "seconds",
        "month": "months",
        "months": "months",
        "mon": "months",
    }
    if u not in mapping:
        raise ValueError(f"지원하지 않는 time unit: {unit}")
    return mapping[u]


class RefDate(tuple):
    __slots__ = ()
    _fields = ("year", "month", "day", "hour", "minute", "second")

    def __new__(cls, year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: float = 0.0):
        return tuple.__new__(cls, (year, month, day, hour, minute, second))

    @property
    def year(self) -> int:
        return self[0]

    @property
    def month(self) -> int:
        return self[1]

    @property
    def day(self) -> int:
        return self[2]

    @property
    def hour(self) -> int:
        return self[3]

    @property
    def minute(self) -> int:
        return self[4]

    @property
    def second(self) -> float:
        return self[5]


class SimpleDate(tuple):
    __slots__ = ()
    _fields = ("year", "month", "day", "hour", "minute", "second")

    def __new__(cls, year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: float = 0.0):
        return tuple.__new__(cls, (year, month, day, hour, minute, second))

    @property
    def year(self) -> int:
        return self[0]

    @property
    def month(self) -> int:
        return self[1]

    @property
    def day(self) -> int:
        return self[2]

    @property
    def hour(self) -> int:
        return self[3]

    @property
    def minute(self) -> int:
        return self[4]

    @property
    def second(self) -> float:
        return self[5]

    def iso(self) -> str:
        sec_int = int(self.second)
        frac = self.second - sec_int
        if abs(frac) < 1e-9:
            return f"{self.year:04d}-{self.month:02d}-{self.day:02d} {self.hour:02d}:{self.minute:02d}:{sec_int:02d}"
        frac_str = f"{frac:.6f}".split(".")[1].rstrip("0")
        return (
            f"{self.year:04d}-{self.month:02d}-{self.day:02d} "
            f"{self.hour:02d}:{self.minute:02d}:{sec_int:02d}.{frac_str}"
        )


def parse_time_units(units: str) -> tuple[str, RefDate]:
    m = UNIT_RE.match(str(units))
    if not m:
        raise ValueError(f"CF time units 파싱 실패: {units}")
    unit = normalize_unit(m.group(1))
    ref = parse_ref_date(m.group(2))
    return unit, ref


def parse_ref_date(text: str) -> RefDate:
    s = text.strip().replace("T", " ")
    s = re.sub(r"\s+(UTC|utc)$", "", s)
    m = REF_RE.match(s)
    if not m:
        raise ValueError(f"reference date 파싱 실패: {text}")
    second = float(m.group("second") or 0.0)
    return RefDate(
        year=int(m.group("year")),
        month=int(m.group("month")),
        day=int(m.group("day")),
        hour=int(m.group("hour") or 0),
        minute=int(m.group("minute") or 0),
        second=second,
    )


def is_gregorian_leap(year: int) -> bool:
    return (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))


def is_julian_leap(year: int) -> bool:
    return year % 4 == 0


def days_in_month(year: int, month: int, calendar: str) -> int:
    cal = normalize_calendar(calendar)
    if cal == "360_day":
        return 30
    if cal == "noleap":
        feb = 28
    elif cal == "all_leap":
        feb = 29
    elif cal == "julian":
        feb = 29 if is_julian_leap(year) else 28
    else:
        feb = 29 if is_gregorian_leap(year) else 28
    month_days = [31, feb, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return month_days[month - 1]


def add_days_calendar(year: int, month: int, day: int, delta_days: int, calendar: str) -> tuple[int, int, int]:
    y, m, d = year, month, day
    n = int(delta_days)

    if n >= 0:
        while n > 0:
            dim = days_in_month(y, m, calendar)
            remain = dim - d
            if n <= remain:
                d += n
                n = 0
            else:
                n -= remain + 1
                d = 1
                m += 1
                if m > 12:
                    m = 1
                    y += 1
    else:
        while n < 0:
            if d + n >= 1:
                d += n
                n = 0
            else:
                n += d
                m -= 1
                if m < 1:
                    m = 12
                    y -= 1
                d = days_in_month(y, m, calendar)

    return y, m, d


def month_key(date: SimpleDate) -> int:
    return date.year * 12 + (date.month - 1)


def gregorian_days_before_month(year: int, month: int) -> int:
    offsets = MONTH_OFFSETS_LEAP if is_gregorian_leap(year) else MONTH_OFFSETS_NOLEAP
    return int(offsets[month - 1])


def julian_days_before_month(year: int, month: int) -> int:
    offsets = MONTH_OFFSETS_LEAP if is_julian_leap(year) else MONTH_OFFSETS_NOLEAP
    return int(offsets[month - 1])


def day_key(date: SimpleDate, calendar: str) -> int:
    cal = normalize_calendar(calendar)
    y, m, d = date.year, date.month, date.day

    if cal == "360_day":
        return (y - 1) * 360 + (m - 1) * 30 + (d - 1)
    if cal == "noleap":
        return (y - 1) * 365 + int(MONTH_OFFSETS_NOLEAP[m - 1]) + (d - 1)
    if cal == "all_leap":
        return (y - 1) * 366 + int(MONTH_OFFSETS_LEAP[m - 1]) + (d - 1)
    if cal == "julian":
        return 365 * (y - 1) + ((y - 1) // 4) + julian_days_before_month(y, m) + (d - 1)
    return 365 * (y - 1) + ((y - 1) // 4) - ((y - 1) // 100) + ((y - 1) // 400) + gregorian_days_before_month(y, m) + (d - 1)


def decode_cf_num(value: float, unit: str, ref: RefDate, calendar: str) -> SimpleDate:
    cal = normalize_calendar(calendar)
    v = float(value)

    if unit == "months":
        rounded = int(round(v))
        if abs(v - rounded) > 1e-8:
            raise ValueError(f"fractional months는 지원하지 않습니다: {value}")
        total = (ref.year * 12 + (ref.month - 1)) + rounded
        y = total // 12
        m = total % 12 + 1
        d = min(ref.day, days_in_month(y, m, cal))
        return SimpleDate(y, m, d, ref.hour, ref.minute, ref.second)

    seconds_per_unit = {
        "days": 86400.0,
        "hours": 3600.0,
        "minutes": 60.0,
        "seconds": 1.0,
    }[unit]

    total_seconds = ref.hour * 3600.0 + ref.minute * 60.0 + ref.second + v * seconds_per_unit
    day_offset = math.floor(total_seconds / 86400.0)
    seconds_of_day = total_seconds - day_offset * 86400.0

    if seconds_of_day < 0:
        day_offset -= 1
        seconds_of_day += 86400.0
    if abs(seconds_of_day - 86400.0) < 1e-8:
        day_offset += 1
        seconds_of_day = 0.0

    y, m, d = add_days_calendar(ref.year, ref.month, ref.day, int(day_offset), cal)
    hour = int(seconds_of_day // 3600.0)
    seconds_of_day -= hour * 3600.0
    minute = int(seconds_of_day // 60.0)
    seconds_of_day -= minute * 60.0
    second = round(seconds_of_day, 6)

    if abs(second - 60.0) < 1e-6:
        second = 0.0
        minute += 1
    if minute >= 60:
        minute -= 60
        hour += 1
    if hour >= 24:
        hour -= 24
        y, m, d = add_days_calendar(y, m, d, 1, cal)

    return SimpleDate(y, m, d, hour, minute, second)


def decode_time_values(time_values: np.ndarray, units: str, calendar: str) -> list[SimpleDate]:
    unit, ref = parse_time_units(units)
    return [decode_cf_num(v, unit, ref, calendar) for v in np.asarray(time_values)]


def infer_frequency_from_decoded(decoded: list[SimpleDate]) -> str:
    if len(decoded) < 2:
        return "unknown"

    month_diffs = []
    day_diffs = []
    for a, b in zip(decoded[:-1], decoded[1:]):
        month_diffs.append((b.year - a.year) * 12 + (b.month - a.month))
        day_diffs.append(day_key(b, "standard") - day_key(a, "standard"))

    if month_diffs:
        frac_monthly = sum(m == 1 for m in month_diffs) / len(month_diffs)
        if frac_monthly >= 0.8:
            return "monthly"
    if day_diffs:
        frac_daily = sum(d == 1 for d in day_diffs) / len(day_diffs)
        if frac_daily >= 0.8:
            return "daily"
    return "unknown"


def infer_frequency_from_time_values(time_values: np.ndarray, units: str, calendar: str) -> str:
    try:
        decoded = decode_time_values(time_values, units, calendar)
        if len(decoded) < 2:
            return "unknown"
        month_diffs = [(b.year - a.year) * 12 + (b.month - a.month) for a, b in zip(decoded[:-1], decoded[1:])]
        day_diffs = [day_key(b, calendar) - day_key(a, calendar) for a, b in zip(decoded[:-1], decoded[1:])]
        if month_diffs:
            frac_monthly = sum(m == 1 for m in month_diffs) / len(month_diffs)
            if frac_monthly >= 0.8:
                return "monthly"
        if day_diffs:
            frac_daily = sum(d == 1 for d in day_diffs) / len(day_diffs)
            if frac_daily >= 0.8:
                return "daily"
    except Exception:
        pass
    return "unknown"


def find_time_name(ds: xr.Dataset) -> str | None:
    for cand in ["time", "valid_time", "time_counter", "TIME", "t", "date"]:
        if cand in ds.coords or cand in ds.variables:
            return cand
    for name, var in ds.variables.items():
        axis = safe_str(var.attrs.get("axis", "")).upper()
        stdn = safe_str(var.attrs.get("standard_name", "")).lower()
        if axis == "T" or stdn == "time":
            return name
    return None


def find_time_bounds_name(ds: xr.Dataset, time_name: str) -> str:
    time_var = ds[time_name]
    bounds_name = time_var.attrs.get("bounds", "")
    if bounds_name and bounds_name in ds.variables:
        return str(bounds_name)
    for name in TIME_BOUNDS_CANDIDATES:
        if name in ds.variables:
            return name
    return ""


def classify_var(varname: str) -> str:
    if varname in ATMOS_VARS:
        return "atmos"
    if varname in OCEAN_VARS:
        return "ocean"
    return "unknown"


def infer_freq_class_from_path(path: Path) -> str:
    name = path.name.lower()
    parts = {p.lower() for p in path.parts}

    if "daily" in parts or ".day." in name or "_day_" in name:
        return "daily"
    if "monthly" in parts or ".mon." in name:
        return "monthly"

    tokens = path.name.split("_")
    if len(tokens) > 1:
        table = tokens[1].lower()
        if table == "day":
            return "daily"
        if table.endswith("mon"):
            return "monthly"

    return "unknown"


def find_target_file(root: Path, exact_name: str, glob_pattern: str) -> tuple[Path | None, str]:
    exact = root / exact_name
    if exact.exists():
        return exact, ""

    matches = sorted([p for p in root.glob(glob_pattern) if p.is_file()])
    if len(matches) == 0:
        return None, f"file not found: {exact_name}"
    if len(matches) == 1:
        return matches[0], f"exact file missing, fallback to {matches[0].name}"
    return matches[0], f"multiple matches found, fallback to {matches[0].name}"


def infer_var_from_filename(ncfile: Path) -> str:
    name = ncfile.name
    if "." in name:
        return name.split(".")[0]
    return name.split("_")[0]


def find_data_var(ds: xr.Dataset, path: Path, time_name: str, expected_var: str | None = None) -> str | None:
    if expected_var and expected_var in ds.data_vars:
        return expected_var

    stem_var = infer_var_from_filename(path)
    if stem_var in ds.data_vars:
        return stem_var

    candidates = []
    for name, da in ds.data_vars.items():
        lname = name.lower()
        if name in COORD_LIKE or "bnds" in lname or "bounds" in lname:
            continue
        if time_name not in da.dims:
            continue
        size = int(np.prod([da.sizes[d] for d in da.dims], dtype=np.int64))
        candidates.append((name, da.ndim, size))

    if not candidates:
        return None

    if expected_var:
        for name, _, _ in candidates:
            if expected_var in name or name in expected_var:
                return name

    for name, _, _ in candidates:
        if stem_var in name or name in stem_var:
            return name

    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return candidates[0][0]


def check_time_monotonic_and_duplicates(time_values: np.ndarray) -> tuple[bool, int]:
    try:
        vals = np.asarray(time_values, dtype=np.float64)
    except Exception:
        return False, -1

    if len(vals) < 2:
        return True, 0

    diffs = np.diff(vals)
    monotonic = bool(np.all(diffs > 0))
    duplicated = int(np.sum(diffs == 0))
    return monotonic, duplicated


def daily_gap_check(time_values: np.ndarray, units: str, max_items: int = 10) -> dict[str, Any]:
    unit, _ = parse_time_units(units)
    if unit == "months":
        raise ValueError("daily 자료인데 time unit이 months 입니다.")

    factor_to_days = {
        "days": 1.0,
        "hours": 1.0 / 24.0,
        "minutes": 1.0 / 1440.0,
        "seconds": 1.0 / 86400.0,
    }[unit]

    raw = np.asarray(time_values, dtype=float)
    if len(raw) <= 1:
        return {
            "freq_class": "daily",
            "gap_check_method": "raw_numeric_daily",
            "n_gap_steps": 0,
            "n_missing_steps_est": 0,
            "first_gap_indices": "",
            "first_gap_prev": "",
            "first_gap_next": "",
            "first_gap_detail": "",
        }

    delta_days = np.diff(raw) * factor_to_days
    tol = 1e-6
    bad = np.where(np.abs(delta_days - 1.0) > tol)[0]

    n_missing = int(sum(max(int(round(d)) - 1, 0) for d in delta_days if d > 1.0 + tol))
    return {
        "freq_class": "daily",
        "gap_check_method": "raw_numeric_daily",
        "n_gap_steps": int(len(bad)),
        "n_missing_steps_est": n_missing,
        "first_gap_indices": "; ".join(safe_strings((int(i) for i in bad), max_items)),
        "first_gap_prev": "; ".join(safe_strings((raw[i] for i in bad), max_items)),
        "first_gap_next": "; ".join(safe_strings((raw[i + 1] for i in bad), max_items)),
        "first_gap_detail": "; ".join(safe_strings((f"delta_days={delta_days[i]:.6f}" for i in bad), max_items)),
    }


def monthly_gap_check(time_values: np.ndarray, units: str, calendar: str, max_items: int = 10) -> dict[str, Any]:
    decoded = decode_time_values(np.asarray(time_values), units, calendar)
    if len(decoded) <= 1:
        return {
            "freq_class": "monthly",
            "gap_check_method": "decoded_monthly_year_month",
            "n_gap_steps": 0,
            "n_missing_steps_est": 0,
            "first_gap_indices": "",
            "first_gap_prev": "",
            "first_gap_next": "",
            "first_gap_detail": "",
        }

    month_keys = np.array([month_key(t) for t in decoded], dtype=np.int64)
    delta = np.diff(month_keys)
    bad = np.where(delta != 1)[0]
    n_missing = int(sum(max(int(d) - 1, 0) for d in delta if d > 1))

    return {
        "freq_class": "monthly",
        "gap_check_method": "decoded_monthly_year_month",
        "n_gap_steps": int(len(bad)),
        "n_missing_steps_est": n_missing,
        "first_gap_indices": "; ".join(safe_strings((int(i) for i in bad), max_items)),
        "first_gap_prev": "; ".join(safe_strings((decoded[i].iso() for i in bad), max_items)),
        "first_gap_next": "; ".join(safe_strings((decoded[i + 1].iso() for i in bad), max_items)),
        "first_gap_detail": "; ".join(safe_strings((f"delta_months={int(delta[i])}" for i in bad), max_items)),
    }


def unknown_gap_check(time_values: np.ndarray, units: str, calendar: str, max_items: int = 10) -> dict[str, Any]:
    raw = np.asarray(time_values, dtype=float)
    if len(raw) <= 2:
        return {
            "freq_class": "unknown",
            "gap_check_method": "unknown_freq_skipped",
            "n_gap_steps": 0,
            "n_missing_steps_est": 0,
            "first_gap_indices": "",
            "first_gap_prev": "",
            "first_gap_next": "",
            "first_gap_detail": "",
        }

    try:
        out = daily_gap_check(raw, units, max_items)
        if out["n_gap_steps"] == 0:
            out["gap_check_method"] = "raw_numeric_daily_inferred"
            return out
    except Exception:
        pass

    try:
        out = monthly_gap_check(raw, units, calendar, max_items)
        out["gap_check_method"] = "decoded_monthly_year_month_inferred"
        return out
    except Exception:
        pass

    return {
        "freq_class": "unknown",
        "gap_check_method": "unknown_freq_skipped",
        "n_gap_steps": -1,
        "n_missing_steps_est": -1,
        "first_gap_indices": "",
        "first_gap_prev": "",
        "first_gap_next": "",
        "first_gap_detail": "gap check unavailable",
    }


def gap_check(time_values: np.ndarray, units: str, calendar: str, freq_class: str, max_items: int = 10) -> dict[str, Any]:
    if freq_class == "daily":
        return daily_gap_check(time_values, units, max_items)
    if freq_class == "monthly":
        return monthly_gap_check(time_values, units, calendar, max_items)
    return unknown_gap_check(time_values, units, calendar, max_items)


def bounds_continuity_check(ds: xr.Dataset, time_name: str, time_values: np.ndarray, max_items: int = 10) -> dict[str, Any]:
    bounds_name = find_time_bounds_name(ds, time_name)
    if not bounds_name:
        return {
            "has_time_bounds": False,
            "time_bounds_name": "",
            "n_bounds_discontinuity": 0,
            "first_bounds_indices": "",
            "first_bounds_prev_end": "",
            "first_bounds_next_start": "",
            "first_bounds_detail": "",
            "bounds_check_note": "",
        }

    try:
        bvar = ds[bounds_name]
        if time_name not in bvar.dims or bvar.ndim != 2:
            return {
                "has_time_bounds": True,
                "time_bounds_name": bounds_name,
                "n_bounds_discontinuity": -1,
                "first_bounds_indices": "",
                "first_bounds_prev_end": "",
                "first_bounds_next_start": "",
                "first_bounds_detail": "",
                "bounds_check_note": f"{bounds_name}: 2차원({time_name}, bnds) 형태가 아님",
            }

        arr = np.asarray(bvar.values)
        time_axis = bvar.dims.index(time_name)
        if time_axis != 0:
            arr = np.moveaxis(arr, time_axis, 0)

        if arr.shape[1] != 2:
            return {
                "has_time_bounds": True,
                "time_bounds_name": bounds_name,
                "n_bounds_discontinuity": -1,
                "first_bounds_indices": "",
                "first_bounds_prev_end": "",
                "first_bounds_next_start": "",
                "first_bounds_detail": "",
                "bounds_check_note": f"{bounds_name}: 마지막 차원 길이가 2가 아님",
            }

        if len(arr) <= 1:
            return {
                "has_time_bounds": True,
                "time_bounds_name": bounds_name,
                "n_bounds_discontinuity": 0,
                "first_bounds_indices": "",
                "first_bounds_prev_end": "",
                "first_bounds_next_start": "",
                "first_bounds_detail": "",
                "bounds_check_note": "",
            }

        lower = np.asarray(arr[:, 0], dtype=float)
        upper = np.asarray(arr[:, 1], dtype=float)
        gap = lower[1:] - upper[:-1]

        raw_time = np.asarray(time_values, dtype=float)
        if len(raw_time) > 1:
            step_scale = float(np.nanmedian(np.abs(np.diff(raw_time))))
            if not np.isfinite(step_scale) or step_scale == 0:
                step_scale = 1.0
        else:
            step_scale = 1.0
        tol = max(1e-8, step_scale * 1e-6)
        bad = np.where(np.abs(gap) > tol)[0]

        return {
            "has_time_bounds": True,
            "time_bounds_name": bounds_name,
            "n_bounds_discontinuity": int(len(bad)),
            "first_bounds_indices": "; ".join(safe_strings((int(i) for i in bad), max_items)),
            "first_bounds_prev_end": "; ".join(safe_strings((upper[i] for i in bad), max_items)),
            "first_bounds_next_start": "; ".join(safe_strings((lower[i + 1] for i in bad), max_items)),
            "first_bounds_detail": "; ".join(safe_strings((gap[i] for i in bad), max_items)),
            "bounds_check_note": "",
        }
    except Exception as e:
        return {
            "has_time_bounds": True,
            "time_bounds_name": bounds_name,
            "n_bounds_discontinuity": -1,
            "first_bounds_indices": "",
            "first_bounds_prev_end": "",
            "first_bounds_next_start": "",
            "first_bounds_detail": "",
            "bounds_check_note": f"bounds_check_error: {e}",
        }


def estimate_da_nbytes(da: xr.DataArray) -> int:
    itemsize = np.dtype(da.dtype).itemsize
    n = 1
    for d in da.dims:
        n *= da.sizes[d]
    return int(n * itemsize)


def compute_nan_stats(da: xr.DataArray, time_name: str, time_block: int = 16) -> dict[str, Any]:
    other_dims = [d for d in da.dims if d != time_name]
    if time_name not in da.dims:
        return {
            "n_time": 0,
            "n_total_per_time": 0,
            "n_time_any_nan": -1,
            "n_time_all_nan": -1,
            "min_nan_count": -1,
            "max_nan_count": -1,
            "ref_nan_count": -1,
            "n_time_nan_changed": -1,
            "first_problem_indices": "",
            "first_problem_times": "",
        }

    n_time = int(da.sizes[time_name])
    n_total = int(np.prod([da.sizes[d] for d in other_dims], dtype=np.int64)) if other_dims else 1
    est_nbytes = estimate_da_nbytes(da)

    if est_nbytes <= 2 * 1024**3:
        if other_dims:
            valid_count = da.count(dim=other_dims).values
        else:
            valid_count = da.notnull().astype(np.int64).values
        valid_count = np.asarray(valid_count).astype(np.int64)
        nan_count = n_total - valid_count
        bad_idx = np.where((nan_count > 0) | (valid_count == 0))[0]
        return {
            "n_time": n_time,
            "n_total_per_time": n_total,
            "n_time_any_nan": int(np.sum(nan_count > 0)),
            "n_time_all_nan": int(np.sum(valid_count == 0)),
            "min_nan_count": int(np.min(nan_count)) if nan_count.size else 0,
            "max_nan_count": int(np.max(nan_count)) if nan_count.size else 0,
            "ref_nan_count": int(nan_count[0]) if nan_count.size else 0,
            "n_time_nan_changed": int(np.sum(nan_count != nan_count[0])) if nan_count.size else 0,
            "first_problem_indices": "; ".join(safe_strings((int(i) for i in bad_idx), 10)),
            "first_problem_times": "; ".join(safe_strings((int(i) for i in bad_idx), 10)),
        }

    min_nan_count: int | None = None
    max_nan_count: int | None = None
    n_time_any_nan = 0
    n_time_all_nan = 0
    n_time_nan_changed = 0
    first_problem_idx: list[int] = []
    first_nan_count: int | None = None

    for i0 in range(0, n_time, time_block):
        i1 = min(i0 + time_block, n_time)
        chunk = da.isel({time_name: slice(i0, i1)}).transpose(time_name, ...).values
        arr = np.asarray(chunk)
        if other_dims:
            valid = np.isfinite(arr).reshape(arr.shape[0], -1).sum(axis=1, dtype=np.int64)
        else:
            valid = np.isfinite(arr).astype(np.int64)
        nan_count = n_total - valid

        if min_nan_count is None:
            min_nan_count = int(nan_count.min())
            max_nan_count = int(nan_count.max())
            first_nan_count = int(nan_count[0])
        else:
            min_nan_count = min(min_nan_count, int(nan_count.min()))
            max_nan_count = max(max_nan_count, int(nan_count.max()))

        n_time_any_nan += int(np.sum(nan_count > 0))
        n_time_all_nan += int(np.sum(valid == 0))
        n_time_nan_changed += int(np.sum(nan_count != first_nan_count))

        bad_local = np.where((nan_count > 0) | (valid == 0))[0]
        for j in bad_local:
            if len(first_problem_idx) >= 10:
                break
            first_problem_idx.append(i0 + int(j))

    return {
        "n_time": n_time,
        "n_total_per_time": n_total,
        "n_time_any_nan": int(n_time_any_nan),
        "n_time_all_nan": int(n_time_all_nan),
        "min_nan_count": int(min_nan_count if min_nan_count is not None else 0),
        "max_nan_count": int(max_nan_count if max_nan_count is not None else 0),
        "ref_nan_count": int(first_nan_count if first_nan_count is not None else 0),
        "n_time_nan_changed": int(n_time_nan_changed),
        "first_problem_indices": "; ".join(str(i) for i in first_problem_idx),
        "first_problem_times": "; ".join(str(i) for i in first_problem_idx),
    }


# ============================================================
# Basic QC
# ============================================================
def analyze_basic_file(
    ncfile: Path,
    source: str,
    expected_var: str | None = None,
    expected_freq: str | None = None,
    note_prefix: str = "",
    max_problem_times: int = 10,
    time_block: int = 16,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source": source,
        "file": str(ncfile),
        "basename": ncfile.name,
        "label": "",
        "expected_var": safe_str(expected_var),
        "expected_freq": safe_str(expected_freq),
        "variable": safe_str(expected_var),
        "category": "",
        "freq_class": "",
        "detected_freq": "",
        "n_time": 0,
        "calendar": "",
        "time_name": "",
        "time_units": "",
        "monotonic_time": "",
        "duplicated_time": -1,
        "n_total_per_time": 0,
        "n_time_any_nan": 0,
        "n_time_all_nan": 0,
        "min_nan_count": 0,
        "max_nan_count": 0,
        "ref_nan_count": 0,
        "n_time_nan_changed": 0,
        "first_problem_indices": "",
        "first_problem_times": "",
        "gap_check_method": "",
        "n_gap_steps": 0,
        "n_missing_steps_est": 0,
        "first_gap_indices": "",
        "first_gap_prev": "",
        "first_gap_next": "",
        "first_gap_detail": "",
        "has_time_bounds": "",
        "time_bounds_name": "",
        "n_bounds_discontinuity": 0,
        "first_bounds_indices": "",
        "first_bounds_prev_end": "",
        "first_bounds_next_start": "",
        "first_bounds_detail": "",
        "status": "",
        "note": note_prefix,
    }

    try:
        ds = xr.open_dataset(ncfile, decode_times=False)
    except Exception as e:
        row["status"] = "ERROR"
        row["note"] = "; ".join([x for x in [note_prefix, f"{type(e).__name__}: {e}"] if x])
        return row

    try:
        time_name = find_time_name(ds)
        row["time_name"] = safe_str(time_name)
        if time_name is None:
            row["status"] = "ERROR"
            row["note"] = "; ".join([x for x in [note_prefix, "time coordinate not found"] if x])
            return row

        actual_var = find_data_var(ds, ncfile, time_name, expected_var)
        if actual_var is None:
            row["status"] = "ERROR"
            row["note"] = "; ".join([x for x in [note_prefix, f"data variable not found (expected={expected_var or infer_var_from_filename(ncfile)})"] if x])
            return row

        da = ds[actual_var]
        time_var = ds[time_name]
        raw_time = np.asarray(time_var.values)
        units = safe_str(time_var.attrs.get("units", ""))
        calendar = normalize_calendar(time_var.attrs.get("calendar", "standard"))
        category = classify_var(actual_var if actual_var else safe_str(expected_var))

        row["variable"] = actual_var
        row["category"] = category
        row["n_time"] = int(raw_time.size)
        row["time_units"] = units
        row["calendar"] = calendar

        monotonic, duplicated = check_time_monotonic_and_duplicates(raw_time)
        row["monotonic_time"] = "TRUE" if monotonic else "FALSE"
        row["duplicated_time"] = duplicated

        freq_hint = expected_freq or infer_freq_class_from_path(ncfile)
        gap_res = gap_check(raw_time, units, calendar, freq_hint, max_problem_times)
        bounds_res = bounds_continuity_check(ds, time_name, raw_time, max_problem_times)
        nan_stats = compute_nan_stats(da, time_name, time_block=time_block)

        row["freq_class"] = gap_res["freq_class"] if gap_res["freq_class"] != "unknown" else freq_hint
        row["detected_freq"] = infer_frequency_from_time_values(raw_time, units, calendar)
        row.update(nan_stats)
        row.update(gap_res)
        row.update(bounds_res)
        row["has_time_bounds"] = bounds_res["has_time_bounds"]
        row["time_bounds_name"] = bounds_res["time_bounds_name"]
        row["n_bounds_discontinuity"] = bounds_res["n_bounds_discontinuity"]

        problem_times_idx = []
        if row["first_problem_indices"]:
            try:
                problem_times_idx = [int(x.strip()) for x in row["first_problem_indices"].split(";") if x.strip()]
            except Exception:
                problem_times_idx = []
        if problem_times_idx:
            time_strings = safe_strings((raw_time[i] for i in problem_times_idx if i < len(raw_time)), max_problem_times)
            row["first_problem_times"] = "; ".join(time_strings)

        fail_reasons: list[str] = []
        warn_reasons: list[str] = []
        info_notes: list[str] = []

        if duplicated == -1:
            warn_reasons.append("monotonic_check_unavailable")
        else:
            if not monotonic:
                fail_reasons.append("time_not_strictly_increasing")
            if duplicated > 0:
                fail_reasons.append(f"duplicated_time={duplicated}")

        if row["n_gap_steps"] > 0:
            fail_reasons.append(f"time_gap_or_overlap={row['n_gap_steps']}")
        elif row["n_gap_steps"] < 0:
            warn_reasons.append("gap_check_unavailable")

        if row["n_bounds_discontinuity"] > 0:
            fail_reasons.append(f"time_bounds_discontinuity={row['n_bounds_discontinuity']}")
        elif row["n_bounds_discontinuity"] < 0:
            warn_reasons.append(bounds_res.get("bounds_check_note") or "bounds_check_unavailable")

        if row["n_time_all_nan"] > 0:
            fail_reasons.append(f"all_nan_timesteps={row['n_time_all_nan']}")

        if category == "atmos":
            if row["n_time_any_nan"] > 0:
                fail_reasons.append(f"atmos_var_has_nan={row['n_time_any_nan']}")
            else:
                info_notes.append("atmos variable: no NaN")
        elif category == "ocean":
            if row["n_time_nan_changed"] > 0:
                warn_reasons.append(f"nan_mask_changes_over_time={row['n_time_nan_changed']}")
            else:
                info_notes.append("ocean variable: fixed mask")
        else:
            if row["n_time_nan_changed"] > 0:
                warn_reasons.append(f"nan_count_changes_over_time={row['n_time_nan_changed']}")

        if expected_var and expected_var != actual_var:
            warn_reasons.append(f"expected_var={expected_var}, detected_var={actual_var}")

        if expected_freq and row["detected_freq"] not in ("", "unknown") and row["detected_freq"] != expected_freq:
            warn_reasons.append(f"expected_freq={expected_freq}, detected_freq={row['detected_freq']}")

        if fail_reasons:
            row["status"] = "FAIL"
        elif warn_reasons:
            row["status"] = "WARN"
        else:
            row["status"] = "PASS"

        notes = []
        if note_prefix:
            notes.append(note_prefix)
        notes.extend(fail_reasons)
        notes.extend(warn_reasons)
        if not fail_reasons and not warn_reasons:
            notes.extend(info_notes if info_notes else ["no issues"])
        row["note"] = "; ".join(notes)
        return row
    except Exception as e:
        row["status"] = "ERROR"
        row["note"] = "; ".join([x for x in [note_prefix, f"{type(e).__name__}: {e}"] if x])
        return row
    finally:
        ds.close()


def run_basic_cmip6(root: Path, pattern: str, max_problem_times: int, time_block: int) -> pd.DataFrame:
    files = sorted(root.rglob(pattern))
    if not files:
        raise SystemExit(f"No files found under {root} with pattern {pattern}")

    rows = []
    for i, path in enumerate(files, start=1):
        row = analyze_basic_file(path, source="cmip6", max_problem_times=max_problem_times, time_block=time_block)
        rows.append(row)
        print(f"[{i}/{len(files)}] {row['status']:5s} {path.name} gaps={row['n_gap_steps']} bounds={row['n_bounds_discontinuity']} {row['note']}")
    return pd.DataFrame(rows)


def run_basic_era5(root: Path, max_problem_times: int, time_block: int) -> pd.DataFrame:
    rows = []
    for target in ERA5_TARGETS:
        print(f"[QC] {target['exact_name']}")
        ncfile, file_note = find_target_file(root, target['exact_name'], target['glob_pattern'])
        if ncfile is None:
            rows.append({
                "source": "era5",
                "file": "",
                "basename": "",
                "label": target["label"],
                "expected_var": target["var"],
                "expected_freq": target["expected_freq"],
                "variable": target["var"],
                "category": classify_var(target["var"]),
                "freq_class": "",
                "detected_freq": "",
                "n_time": 0,
                "calendar": "",
                "time_name": "",
                "time_units": "",
                "monotonic_time": "",
                "duplicated_time": -1,
                "n_total_per_time": 0,
                "n_time_any_nan": 0,
                "n_time_all_nan": 0,
                "min_nan_count": 0,
                "max_nan_count": 0,
                "ref_nan_count": 0,
                "n_time_nan_changed": 0,
                "first_problem_indices": "",
                "first_problem_times": "",
                "gap_check_method": "",
                "n_gap_steps": -1,
                "n_missing_steps_est": -1,
                "first_gap_indices": "",
                "first_gap_prev": "",
                "first_gap_next": "",
                "first_gap_detail": "",
                "has_time_bounds": "",
                "time_bounds_name": "",
                "n_bounds_discontinuity": -1,
                "first_bounds_indices": "",
                "first_bounds_prev_end": "",
                "first_bounds_next_start": "",
                "first_bounds_detail": "",
                "status": "ERROR",
                "note": file_note,
            })
            continue

        row = analyze_basic_file(
            ncfile,
            source="era5",
            expected_var=target["var"],
            expected_freq=target["expected_freq"],
            note_prefix=file_note,
            max_problem_times=max_problem_times,
            time_block=time_block,
        )
        row["label"] = target["label"]
        rows.append(row)
    return pd.DataFrame(rows)


def run_basic_godas(root: Path, max_problem_times: int, time_block: int) -> pd.DataFrame:
    files = [p for p in sorted(root.glob("*.nc")) if p.is_file() and p.name != GODAS_EXCLUDE_BASENAME]
    if not files:
        raise SystemExit(f"No .nc files found in {root} (excluding {GODAS_EXCLUDE_BASENAME})")

    rows = []
    for path in files:
        expected_var = infer_var_from_filename(path)
        print(f"[QC] {path.name}")
        row = analyze_basic_file(
            path,
            source="godas",
            expected_var=expected_var,
            expected_freq=None,
            max_problem_times=max_problem_times,
            time_block=time_block,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def find_oras5_target_file(root: Path, expected_var: str) -> Path | None:
    matches = sorted([p for p in root.glob(f"{expected_var}*.nc") if p.is_file()])
    if not matches:
        return None
    exact = [p for p in matches if p.name.startswith(f"{expected_var}.")]
    return exact[0] if exact else matches[0]


def run_basic_oras5(root: Path, max_problem_times: int, time_block: int) -> pd.DataFrame:
    rows = []
    for var in sorted(OCEAN_VARS):
        print(f"[QC] {var}")
        ncfile = find_oras5_target_file(root, var)
        if ncfile is None:
            rows.append({
                "source": "oras5",
                "file": "",
                "basename": "",
                "label": var,
                "expected_var": var,
                "expected_freq": "",
                "variable": var,
                "category": "ocean",
                "freq_class": "",
                "detected_freq": "",
                "n_time": 0,
                "calendar": "",
                "time_name": "",
                "time_units": "",
                "monotonic_time": "",
                "duplicated_time": -1,
                "n_total_per_time": 0,
                "n_time_any_nan": 0,
                "n_time_all_nan": 0,
                "min_nan_count": 0,
                "max_nan_count": 0,
                "ref_nan_count": 0,
                "n_time_nan_changed": 0,
                "first_problem_indices": "",
                "first_problem_times": "",
                "gap_check_method": "",
                "n_gap_steps": -1,
                "n_missing_steps_est": -1,
                "first_gap_indices": "",
                "first_gap_prev": "",
                "first_gap_next": "",
                "first_gap_detail": "",
                "has_time_bounds": "",
                "time_bounds_name": "",
                "n_bounds_discontinuity": -1,
                "first_bounds_indices": "",
                "first_bounds_prev_end": "",
                "first_bounds_next_start": "",
                "first_bounds_detail": "",
                "status": "ERROR",
                "note": f"target file not found for variable: {var}",
            })
            continue

        row = analyze_basic_file(
            ncfile,
            source="oras5",
            expected_var=var,
            max_problem_times=max_problem_times,
            time_block=time_block,
        )
        row["label"] = var
        rows.append(row)
    return pd.DataFrame(rows)


def resolve_basic_csv(source: str, csv_arg: str | None) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if csv_arg:
        name = basename_only(csv_arg)
    else:
        name = f"{source}_qc_summary.csv"
    if not name.lower().endswith(".csv"):
        name = f"{name}.csv"
    return OUTPUT_DIR / name


def print_basic_summary(df: pd.DataFrame, out_csv: Path) -> None:
    status_order = {"FAIL": 0, "WARN": 1, "ERROR": 2, "PASS": 3}
    if "status" in df.columns:
        df["_status_order"] = df["status"].map(status_order).fillna(99)
        df.sort_values(["_status_order", "basename", "label"], inplace=True)
        df.drop(columns=["_status_order"], inplace=True)

    df.to_csv(out_csv, index=False)

    print("\n============================================================")
    print(f"Output CSV: {out_csv}")
    print("------------------------------------------------------------")
    if "status" in df.columns:
        print(df["status"].value_counts(dropna=False).to_string())
    print("============================================================\n")

    bad = df[df["status"].isin(["FAIL", "WARN", "ERROR"])].copy()
    if len(bad) > 0:
        show_cols = [
            "source", "status", "basename", "label", "expected_var", "variable", "expected_freq", "detected_freq",
            "n_gap_steps", "n_missing_steps_est", "n_time_any_nan", "n_time_all_nan", "n_time_nan_changed",
            "first_gap_prev", "first_gap_next", "note",
        ]
        show_cols = [c for c in show_cols if c in bad.columns]
        print("[Issues]")
        with pd.option_context("display.max_colwidth", 220, "display.width", 320):
            print(bad[show_cols].to_string(index=False))
    else:
        print("[Issues] none")


# ============================================================
# Mask QC
# ============================================================
def open_mask_dataarray(ncfile: Path, varname: str) -> tuple[xr.Dataset, xr.DataArray, str]:
    try:
        ds = xr.open_dataset(ncfile, decode_times=True)
    except Exception:
        ds = xr.open_dataset(ncfile, decode_times=False)

    time_name = find_time_name(ds)
    if time_name is None:
        ds.close()
        raise ValueError("time coordinate not found")

    if varname in ds.data_vars:
        da = ds[varname]
    else:
        actual_var = find_data_var(ds, ncfile, time_name, expected_var=varname)
        if actual_var is None:
            ds.close()
            raise ValueError(f"data variable not found (expected={varname})")
        da = ds[actual_var]
    return ds, da, time_name


def make_mask_change(da: xr.DataArray, time_name: str) -> xr.DataArray:
    return da.notnull().astype(np.int8).std(time_name) > 0


def make_flip_count(da: xr.DataArray, time_name: str) -> xr.DataArray:
    return np.abs(da.notnull().astype(np.int8).diff(time_name)).sum(time_name)


def make_nan_count(da: xr.DataArray, time_name: str) -> xr.DataArray:
    space_dims = [d for d in da.dims if d != time_name]
    return da.isnull().sum(dim=space_dims)


def resolve_mask_outdir(source: str, outdir_arg: str | None) -> Path:
    if outdir_arg:
        outdir = Path(outdir_arg)
    else:
        outdir = OUTPUT_DIR / f"qc_{source}_mask"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def summarize_mask_one(var: str, ncfile: Path | None, outdir: Path, source: str) -> dict[str, Any]:
    if ncfile is None:
        return {
            "source": source,
            "variable": var,
            "file": "",
            "status": "ERROR",
            "n_time": np.nan,
            "n_grid": np.nan,
            "changed_cells": np.nan,
            "fraction_changed": np.nan,
            "max_flip_count": np.nan,
            "mean_flip_on_changed_cells": np.nan,
            "nan_count_min": np.nan,
            "nan_count_max": np.nan,
            "nan_count_range": np.nan,
            "note": "file not found",
        }

    ds = None
    try:
        ds, da, time_name = open_mask_dataarray(ncfile, var)
        mask_change = make_mask_change(da, time_name)
        flip_count = make_flip_count(da, time_name)
        nan_count = make_nan_count(da, time_name)
        nan_count_delta = nan_count - nan_count.min()

        changed_cells = int(mask_change.sum().item())
        total_cells = int(mask_change.size)
        fraction_changed = changed_cells / total_cells if total_cells > 0 else np.nan
        max_flip = int(flip_count.max().item())
        changed_flip = flip_count.where(flip_count > 0)
        mean_flip_changed = float(changed_flip.mean().item()) if changed_cells > 0 else 0.0
        nan_min = int(nan_count.min().item())
        nan_max = int(nan_count.max().item())
        nan_rng = nan_max - nan_min

        plt.figure(figsize=(10, 4))
        mask_change.plot()
        plt.title(f"Cells where valid/missing status changes over time: {var}")
        plt.tight_layout()
        plt.savefig(outdir / f"{var}_mask_change.png", dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 4))
        flip_count.plot()
        plt.title(f"Number of valid/missing flips by grid cell: {var}")
        plt.tight_layout()
        plt.savefig(outdir / f"{var}_flip_count.png", dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 4))
        nan_count_delta.plot()
        plt.ticklabel_format(axis="y", style="plain", useOffset=False)
        plt.title(f"NaN count by time (delta from minimum): {var}")
        plt.ylabel("NaN count - min(NaN count)")
        plt.tight_layout()
        plt.savefig(outdir / f"{var}_nan_count_delta.png", dpi=150, bbox_inches="tight")
        plt.close()

        if fraction_changed < 0.01 and nan_rng <= 20:
            status = "OK"
            note = "small mask variation"
        else:
            status = "REVIEW"
            note = "inspect maps and time series"

        return {
            "source": source,
            "variable": var,
            "file": str(ncfile),
            "status": status,
            "n_time": int(da.sizes[time_name]),
            "n_grid": total_cells,
            "changed_cells": changed_cells,
            "fraction_changed": fraction_changed,
            "max_flip_count": max_flip,
            "mean_flip_on_changed_cells": mean_flip_changed,
            "nan_count_min": nan_min,
            "nan_count_max": nan_max,
            "nan_count_range": nan_rng,
            "note": note,
        }
    except Exception as e:
        return {
            "source": source,
            "variable": var,
            "file": "" if ncfile is None else str(ncfile),
            "status": "ERROR",
            "n_time": np.nan,
            "n_grid": np.nan,
            "changed_cells": np.nan,
            "fraction_changed": np.nan,
            "max_flip_count": np.nan,
            "mean_flip_on_changed_cells": np.nan,
            "nan_count_min": np.nan,
            "nan_count_max": np.nan,
            "nan_count_range": np.nan,
            "note": f"{type(e).__name__}: {e}",
        }
    finally:
        if ds is not None:
            ds.close()


def run_mask_godas(root: Path, outdir: Path) -> pd.DataFrame:
    rows = []
    for var in sorted(OCEAN_VARS):
        print(f"[CHECK] {var}")
        ncfile = root / f"{var}.198001-202512.nc"
        rows.append(summarize_mask_one(var, ncfile if ncfile.exists() else None, outdir, source="godas"))
    return pd.DataFrame(rows)


def run_mask_oras5(root: Path, outdir: Path) -> pd.DataFrame:
    rows = []
    for var in sorted(OCEAN_VARS):
        print(f"[CHECK] {var}")
        ncfile = find_oras5_target_file(root, var)
        rows.append(summarize_mask_one(var, ncfile, outdir, source="oras5"))
    return pd.DataFrame(rows)


def print_mask_summary(df: pd.DataFrame, outdir: Path, source: str) -> None:
    out_csv = outdir / f"{source}_mask_change_summary.csv"
    df.to_csv(out_csv, index=False)
    print("\n============================================================")
    print(f"Output dir : {outdir}")
    print(f"Summary CSV: {out_csv}")
    print("------------------------------------------------------------")
    print(df.to_string(index=False))
    print("============================================================\n")


# ============================================================
# CLI
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified QC script for CMIP6 / ERA5 / GODAS / ORAS5"
    )
    p.add_argument("--source", required=True, choices=["cmip6", "era5", "godas", "oras5"], help="data source")
    p.add_argument("--check", default="basic", choices=["basic", "mask", "all"], help="QC type to run")
    p.add_argument("root", nargs="?", default=None, help="root directory; if omitted, source-specific default is used")
    p.add_argument("--pattern", default="*.nc", help="recursive glob pattern for CMIP6 basic QC (default: *.nc)")
    p.add_argument("--csv", default=None, help="output CSV filename for basic QC (saved under fixed output dir)")
    p.add_argument("--outdir", default=None, help="output directory for mask QC plots/summary")
    p.add_argument("--max-problem-times", type=int, default=10, help="how many problematic timestamps/indices to store")
    p.add_argument("--time-block", type=int, default=16, help="time block size for streaming NaN stats on large arrays")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else DEFAULT_ROOTS[args.source]
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"[ERROR] directory not found: {root}")

    basic_df: pd.DataFrame | None = None
    mask_df: pd.DataFrame | None = None

    if args.check in ("basic", "all"):
        if args.source == "cmip6":
            basic_df = run_basic_cmip6(root, args.pattern, args.max_problem_times, args.time_block)
        elif args.source == "era5":
            basic_df = run_basic_era5(root, args.max_problem_times, args.time_block)
        elif args.source == "godas":
            basic_df = run_basic_godas(root, args.max_problem_times, args.time_block)
        elif args.source == "oras5":
            basic_df = run_basic_oras5(root, args.max_problem_times, args.time_block)
        else:
            raise SystemExit(f"[ERROR] unsupported source: {args.source}")

        out_csv = resolve_basic_csv(args.source, args.csv)
        print_basic_summary(basic_df, out_csv)

    if args.check in ("mask", "all"):
        if args.source not in {"godas", "oras5"}:
            if args.check == "mask":
                raise SystemExit("[ERROR] mask QC is currently supported only for godas/oras5")
            print(f"[INFO] mask QC skipped for source={args.source} (supported only for godas/oras5)")
        else:
            outdir = resolve_mask_outdir(args.source, args.outdir)
            if args.source == "godas":
                mask_df = run_mask_godas(root, outdir)
            else:
                mask_df = run_mask_oras5(root, outdir)
            print_mask_summary(mask_df, outdir, args.source)


if __name__ == "__main__":
    main()
