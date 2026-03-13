#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

OUTPUT_DIR = Path("/mnt/d/project/01_ENSO/03_output")
DEFAULT_CSV_NAME = "cmip6_qc_summary.csv"

ATMOS_VARS = {"psl", "uas", "vas"}
OCEAN_VARS = {"mlotst", "ohc300", "sos", "tos", "uos", "vos"}
COORD_LIKE = {
    "time", "time_bnds", "time_bounds", "lat", "lon", "latitude", "longitude",
    "lev", "olevel", "depth", "deptht", "depthu", "depthv", "depthw", "bnds",
    "bounds", "vertices_latitude", "vertices_longitude", "member_id", "height",
    "type", "axis_nbounds"
}
TIME_BOUNDS_CANDIDATES = ("time_bnds", "time_bounds")
UNIT_RE = re.compile(r"^\s*([A-Za-z_]+)\s+since\s+(.+?)\s*$")
REF_RE = re.compile(
    r"^\s*(?P<year>[+-]?\d{1,6})-(?P<month>\d{1,2})-(?P<day>\d{1,2})"
    r"(?:[ T](?P<hour>\d{1,2})(?::(?P<minute>\d{1,2})(?::(?P<second>\d{1,2}(?:\.\d+)?))?)?)?"
    r"(?:\s*(?:Z|UTC|[+-]\d{2}:?\d{2}))?\s*$"
)
MONTH_OFFSETS_NOLEAP = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=np.int64)
MONTH_OFFSETS_LEAP = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335], dtype=np.int64)


@dataclass(frozen=True)
class RefDate:
    year: int
    month: int
    day: int
    hour: int = 0
    minute: int = 0
    second: float = 0.0


@dataclass(frozen=True)
class SimpleDate:
    year: int
    month: int
    day: int
    hour: int = 0
    minute: int = 0
    second: float = 0.0

    def iso(self) -> str:
        sec_int = int(self.second)
        frac = self.second - sec_int
        if abs(frac) < 1e-9:
            return f"{self.year:04d}-{self.month:02d}-{self.day:02d} {self.hour:02d}:{self.minute:02d}:{sec_int:02d}"
        frac_str = f"{frac:.6f}".split(".")[1].rstrip("0")
        return f"{self.year:04d}-{self.month:02d}-{self.day:02d} {self.hour:02d}:{self.minute:02d}:{sec_int:02d}.{frac_str}"


@dataclass
class GapCheckResult:
    freq_class: str
    gap_check_method: str
    n_gap_steps: int
    n_missing_steps_est: int
    first_gap_indices: str
    first_gap_prev: str
    first_gap_next: str
    first_gap_detail: str


@dataclass
class BoundsCheckResult:
    has_time_bounds: bool
    time_bounds_name: str
    n_bounds_discontinuity: int
    first_bounds_indices: str
    first_bounds_prev_end: str
    first_bounds_next_start: str
    first_bounds_detail: str


@dataclass
class QCRow:
    file: str
    variable: str
    category: str
    freq_class: str
    nt: int
    calendar: str
    time_units: str
    monotonic_time: bool | str
    duplicated_time: bool | str
    total_cells_per_time: int
    n_time_any_nan: int
    n_time_all_nan: int
    min_nan_count: int
    max_nan_count: int
    ref_nan_count: int
    n_time_nan_changed: int
    first_problem_indices: str
    first_problem_times: str
    gap_check_method: str
    n_gap_steps: int
    n_missing_steps_est: int
    first_gap_indices: str
    first_gap_prev: str
    first_gap_next: str
    first_gap_detail: str
    has_time_bounds: bool | str
    time_bounds_name: str
    n_bounds_discontinuity: int
    first_bounds_indices: str
    first_bounds_prev_end: str
    first_bounds_next_start: str
    first_bounds_detail: str
    status: str
    note: str


def infer_varname(ds: xr.Dataset, path: Path) -> str:
    stem_var = path.name.split("_")[0]
    if stem_var in ds.data_vars:
        return stem_var

    candidates = []
    for name, da in ds.data_vars.items():
        if name in COORD_LIKE:
            continue
        if "time" in da.dims:
            candidates.append(name)

    if not candidates:
        raise ValueError("time 차원을 가진 data variable을 찾지 못했습니다.")

    if len(candidates) == 1:
        return candidates[0]

    for name in candidates:
        if stem_var in name or name in stem_var:
            return name

    return candidates[0]


def classify_var(varname: str) -> str:
    if varname in ATMOS_VARS:
        return "atmos"
    if varname in OCEAN_VARS:
        return "ocean"
    return "unknown"


def infer_freq_class(path: Path) -> str:
    name = path.name.lower()
    parts = {p.lower() for p in path.parts}

    if "daily" in parts or "_day_" in name:
        return "daily"
    if "monthly" in parts:
        return "monthly"

    tokens = path.name.split("_")
    if len(tokens) > 1:
        table = tokens[1].lower()
        if table == "day":
            return "daily"
        if table.endswith("mon"):
            return "monthly"

    return "unknown"


def safe_strings(values: Iterable, limit: int = 10) -> list[str]:
    out = []
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
    if month < 1 or month > 12:
        raise ValueError(f"잘못된 month: {month}")

    cal = normalize_calendar(calendar)
    if cal == "360_day":
        return 30

    if cal == "noleap":
        feb = 28
    elif cal == "all_leap":
        feb = 29
    elif cal == "julian":
        feb = 29 if is_julian_leap(year) else 28
    else:  # standard / proleptic_gregorian
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
                n -= (remain + 1)
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


def day_key(date: SimpleDate, calendar: str) -> int:
    cal = normalize_calendar(calendar)
    y, m, d = date.year, date.month, date.day

    if y <= 0:
        # SSP/CMIP6 실사용 범위에서는 거의 나오지 않지만, 음수/0년이면 비교 목적의 단순 누산으로 처리.
        if cal == "360_day":
            return y * 360 + (m - 1) * 30 + (d - 1)
        if cal == "noleap":
            return y * 365 + int(MONTH_OFFSETS_NOLEAP[m - 1]) + (d - 1)
        if cal == "all_leap":
            return y * 366 + int(MONTH_OFFSETS_LEAP[m - 1]) + (d - 1)

    if cal == "360_day":
        return (y - 1) * 360 + (m - 1) * 30 + (d - 1)
    if cal == "noleap":
        return (y - 1) * 365 + int(MONTH_OFFSETS_NOLEAP[m - 1]) + (d - 1)
    if cal == "all_leap":
        return (y - 1) * 366 + int(MONTH_OFFSETS_LEAP[m - 1]) + (d - 1)
    if cal == "julian":
        return 365 * (y - 1) + ((y - 1) // 4) + julian_days_before_month(y, m) + (d - 1)

    # standard / proleptic_gregorian
    return 365 * (y - 1) + ((y - 1) // 4) - ((y - 1) // 100) + ((y - 1) // 400) + gregorian_days_before_month(y, m) + (d - 1)


def gregorian_days_before_month(year: int, month: int) -> int:
    offsets = MONTH_OFFSETS_LEAP if is_gregorian_leap(year) else MONTH_OFFSETS_NOLEAP
    return int(offsets[month - 1])


def julian_days_before_month(year: int, month: int) -> int:
    offsets = MONTH_OFFSETS_LEAP if is_julian_leap(year) else MONTH_OFFSETS_NOLEAP
    return int(offsets[month - 1])


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

    total_seconds = (
        ref.hour * 3600.0
        + ref.minute * 60.0
        + ref.second
        + v * seconds_per_unit
    )

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


def find_time_bounds_name(ds: xr.Dataset) -> str:
    time_var = ds["time"]
    bounds_name = time_var.attrs.get("bounds", "")
    if bounds_name and bounds_name in ds.variables:
        return str(bounds_name)
    for name in TIME_BOUNDS_CANDIDATES:
        if name in ds.variables:
            return name
    return ""


def bounds_continuity_check(ds: xr.Dataset, time_values: np.ndarray, max_items: int = 10) -> BoundsCheckResult:
    bounds_name = find_time_bounds_name(ds)
    if not bounds_name:
        return BoundsCheckResult(False, "", 0, "", "", "", "")

    bvar = ds[bounds_name]
    b = np.asarray(bvar.values)
    if b.ndim != 2 or "time" not in bvar.dims:
        return BoundsCheckResult(True, bounds_name, 0, "", "", "", f"{bounds_name}: 2차원(time, bnds) 형태가 아님")

    time_axis = bvar.dims.index("time")
    if time_axis != 0:
        b = np.moveaxis(b, time_axis, 0)

    if b.shape[1] != 2:
        return BoundsCheckResult(True, bounds_name, 0, "", "", "", f"{bounds_name}: 마지막 차원 길이가 2가 아님")

    lower = np.asarray(b[:, 0], dtype=float)
    upper = np.asarray(b[:, 1], dtype=float)

    if len(lower) <= 1:
        return BoundsCheckResult(True, bounds_name, 0, "", "", "", "")

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
    if len(bad) == 0:
        return BoundsCheckResult(True, bounds_name, 0, "", "", "", "")

    first_idx = safe_strings((int(i) for i in bad), max_items)
    prev_end = safe_strings((upper[i] for i in bad), max_items)
    next_start = safe_strings((lower[i + 1] for i in bad), max_items)
    detail = safe_strings((gap[i] for i in bad), max_items)

    return BoundsCheckResult(
        has_time_bounds=True,
        time_bounds_name=bounds_name,
        n_bounds_discontinuity=int(len(bad)),
        first_bounds_indices="; ".join(first_idx),
        first_bounds_prev_end="; ".join(prev_end),
        first_bounds_next_start="; ".join(next_start),
        first_bounds_detail="; ".join(detail),
    )


def daily_gap_check(time_values: np.ndarray, units: str, max_items: int = 10) -> GapCheckResult:
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
        return GapCheckResult("daily", "raw_numeric_daily", 0, 0, "", "", "", "")

    delta_days = np.diff(raw) * factor_to_days
    tol = 1e-6
    bad = np.where(np.abs(delta_days - 1.0) > tol)[0]

    n_missing = int(sum(max(int(round(d)) - 1, 0) for d in delta_days if d > 1.0 + tol))
    first_idx = safe_strings((int(i) for i in bad), max_items)
    prev_vals = safe_strings((raw[i] for i in bad), max_items)
    next_vals = safe_strings((raw[i + 1] for i in bad), max_items)
    detail = safe_strings((f"delta_days={delta_days[i]:.6f}" for i in bad), max_items)

    return GapCheckResult(
        freq_class="daily",
        gap_check_method="raw_numeric_daily",
        n_gap_steps=int(len(bad)),
        n_missing_steps_est=n_missing,
        first_gap_indices="; ".join(first_idx),
        first_gap_prev="; ".join(prev_vals),
        first_gap_next="; ".join(next_vals),
        first_gap_detail="; ".join(detail),
    )


def monthly_gap_check(time_values: np.ndarray, units: str, calendar: str, max_items: int = 10) -> GapCheckResult:
    decoded = decode_time_values(np.asarray(time_values), units, calendar)
    if len(decoded) <= 1:
        return GapCheckResult("monthly", "decoded_monthly_year_month", 0, 0, "", "", "", "")

    month_keys = np.array([month_key(t) for t in decoded], dtype=np.int64)
    delta = np.diff(month_keys)
    bad = np.where(delta != 1)[0]

    n_missing = int(sum(max(int(d) - 1, 0) for d in delta if d > 1))
    first_idx = safe_strings((int(i) for i in bad), max_items)
    prev_vals = safe_strings((decoded[i].iso() for i in bad), max_items)
    next_vals = safe_strings((decoded[i + 1].iso() for i in bad), max_items)
    detail = safe_strings((f"delta_months={int(delta[i])}" for i in bad), max_items)

    return GapCheckResult(
        freq_class="monthly",
        gap_check_method="decoded_monthly_year_month",
        n_gap_steps=int(len(bad)),
        n_missing_steps_est=n_missing,
        first_gap_indices="; ".join(first_idx),
        first_gap_prev="; ".join(prev_vals),
        first_gap_next="; ".join(next_vals),
        first_gap_detail="; ".join(detail),
    )


def unknown_gap_check(time_values: np.ndarray, units: str, calendar: str, max_items: int = 10) -> GapCheckResult:
    raw = np.asarray(time_values, dtype=float)
    if len(raw) <= 2:
        return GapCheckResult("unknown", "unknown_freq_skipped", 0, 0, "", "", "", "")

    # 보조 휴리스틱: day-level 연속이면 daily로 간주, 아니면 monthly decode를 시도.
    try:
        daily = daily_gap_check(raw, units, max_items)
        if daily.n_gap_steps == 0:
            daily.freq_class = "daily"
            daily.gap_check_method = "raw_numeric_daily_inferred"
            return daily
    except Exception:
        pass

    try:
        monthly = monthly_gap_check(raw, units, calendar, max_items)
        monthly.freq_class = "monthly"
        monthly.gap_check_method = "decoded_monthly_year_month_inferred"
        return monthly
    except Exception:
        pass

    return GapCheckResult(
        freq_class="unknown",
        gap_check_method="unknown_freq_skipped",
        n_gap_steps=0,
        n_missing_steps_est=0,
        first_gap_indices="",
        first_gap_prev="",
        first_gap_next="",
        first_gap_detail="gap check unavailable",
    )


def gap_check(time_values: np.ndarray, units: str, calendar: str, freq_class: str, max_items: int = 10) -> GapCheckResult:
    if freq_class == "daily":
        return daily_gap_check(time_values, units, max_items)
    if freq_class == "monthly":
        return monthly_gap_check(time_values, units, calendar, max_items)
    return unknown_gap_check(time_values, units, calendar, max_items)


def resolve_output_csv(csv_arg: Path | str | None) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if csv_arg is None:
        name = DEFAULT_CSV_NAME
    else:
        p = Path(csv_arg)
        name = p.name if p.name else DEFAULT_CSV_NAME

    if not name.lower().endswith(".csv"):
        name = f"{name}.csv"

    return OUTPUT_DIR / name


def analyze_file(path: Path, max_problem_times: int = 10) -> QCRow:
    ds = xr.open_dataset(path, decode_times=False)
    try:
        if "time" not in ds.coords and "time" not in ds.dims:
            raise ValueError("time 축이 없습니다.")

        varname = infer_varname(ds, path)
        da = ds[varname]

        if "time" not in da.dims:
            raise ValueError(f"변수 {varname} 에 time 차원이 없습니다.")

        other_dims = [d for d in da.dims if d != "time"]
        category = classify_var(varname)
        freq_class = infer_freq_class(path)
        time_values = np.asarray(ds["time"].values)
        time_units = ds["time"].attrs.get("units", "")
        calendar = normalize_calendar(ds["time"].attrs.get("calendar", "standard"))

        monotonic_time = bool(np.all(np.diff(time_values) > 0)) if len(time_values) > 1 else True
        duplicated_time = bool(pd.Index(time_values).has_duplicates)

        n_total = int(math.prod(int(da.sizes[d]) for d in other_dims)) if other_dims else 1

        valid_counts = np.empty(int(da.sizes["time"]), dtype=np.int64)
        for i in range(int(da.sizes["time"])):
            valid_counts[i] = int(da.isel(time=i).count().item())

        nan_counts = n_total - valid_counts
        any_nan = nan_counts > 0
        all_nan = valid_counts == 0
        ref_nan = int(nan_counts[0]) if len(nan_counts) else 0
        changed_nan = nan_counts != ref_nan

        problem_times_idx = np.where(all_nan | changed_nan)[0]
        problem_time_strings = safe_strings(time_values[problem_times_idx], max_problem_times) if len(problem_times_idx) else []
        problem_idx_strings = safe_strings(problem_times_idx, max_problem_times) if len(problem_times_idx) else []

        n_time_any_nan = int(any_nan.sum())
        n_time_all_nan = int(all_nan.sum())
        n_time_nan_changed = int(changed_nan.sum())

        gap_res = gap_check(time_values, time_units, calendar, freq_class, max_problem_times)
        bounds_res = bounds_continuity_check(ds, time_values, max_problem_times)

        status = "PASS"
        notes: list[str] = []

        if n_time_all_nan > 0:
            status = "FAIL"
            notes.append("전체 시점이 NaN인 time 존재")
        if category == "atmos" and n_time_any_nan > 0:
            status = "FAIL"
            notes.append("대기 변수인데 NaN 존재")
        if not monotonic_time or duplicated_time:
            status = "FAIL"
            notes.append("time 좌표 비정상")
        if gap_res.n_gap_steps > 0:
            status = "FAIL"
            notes.append("time gap/overlap 감지")
        if bounds_res.n_bounds_discontinuity > 0:
            status = "FAIL"
            notes.append("time_bounds continuity 깨짐")

        if status == "PASS":
            if gap_res.gap_check_method == "unknown_freq_skipped":
                status = "WARN"
                notes.append("gap check unavailable")
            elif category == "ocean" and n_time_nan_changed > 0:
                status = "WARN"
                notes.append("ocean mask/valid cell 수가 time에 따라 변함")
            elif category == "unknown" and n_time_nan_changed > 0:
                status = "WARN"
                notes.append("NaN 개수가 time에 따라 변함")
            elif category == "ocean":
                notes.append("ocean 변수: 고정 mask 유지")
            elif category == "atmos":
                notes.append("대기 변수: NaN 없음")
            else:
                notes.append("특이사항 없음")

        return QCRow(
            file=str(path),
            variable=varname,
            category=category,
            freq_class=gap_res.freq_class if gap_res.freq_class != "unknown" else freq_class,
            nt=int(da.sizes["time"]),
            calendar=calendar,
            time_units=time_units,
            monotonic_time=monotonic_time,
            duplicated_time=duplicated_time,
            total_cells_per_time=n_total,
            n_time_any_nan=n_time_any_nan,
            n_time_all_nan=n_time_all_nan,
            min_nan_count=int(nan_counts.min()) if len(nan_counts) else 0,
            max_nan_count=int(nan_counts.max()) if len(nan_counts) else 0,
            ref_nan_count=ref_nan,
            n_time_nan_changed=n_time_nan_changed,
            first_problem_indices="; ".join(problem_idx_strings),
            first_problem_times="; ".join(problem_time_strings),
            gap_check_method=gap_res.gap_check_method,
            n_gap_steps=gap_res.n_gap_steps,
            n_missing_steps_est=gap_res.n_missing_steps_est,
            first_gap_indices=gap_res.first_gap_indices,
            first_gap_prev=gap_res.first_gap_prev,
            first_gap_next=gap_res.first_gap_next,
            first_gap_detail=gap_res.first_gap_detail,
            has_time_bounds=bounds_res.has_time_bounds,
            time_bounds_name=bounds_res.time_bounds_name,
            n_bounds_discontinuity=bounds_res.n_bounds_discontinuity,
            first_bounds_indices=bounds_res.first_bounds_indices,
            first_bounds_prev_end=bounds_res.first_bounds_prev_end,
            first_bounds_next_start=bounds_res.first_bounds_next_start,
            first_bounds_detail=bounds_res.first_bounds_detail,
            status=status,
            note="; ".join(notes),
        )
    finally:
        ds.close()


def main() -> None:
    p = argparse.ArgumentParser(description="CMIP6 netCDF files: NaN QC + time-gap QC along time dimension")
    p.add_argument("root", type=Path, help="root directory to scan recursively")
    p.add_argument("--pattern", default="*.nc", help="glob pattern (default: *.nc)")
    p.add_argument("--csv", type=Path, default=Path(DEFAULT_CSV_NAME), help="output CSV filename only; saved under fixed output dir")
    p.add_argument("--max-problem-times", type=int, default=10, help="how many problematic timestamps/indices to store")
    args = p.parse_args()

    output_csv = resolve_output_csv(args.csv)

    files = sorted(args.root.rglob(args.pattern))
    if not files:
        raise SystemExit(f"No files found under {args.root} with pattern {args.pattern}")

    rows: list[QCRow] = []
    for i, path in enumerate(files, start=1):
        try:
            row = analyze_file(path, max_problem_times=args.max_problem_times)
        except Exception as e:
            row = QCRow(
                file=str(path), variable="", category="", freq_class="", nt=0,
                calendar="", time_units="", monotonic_time="", duplicated_time="",
                total_cells_per_time=0, n_time_any_nan=0, n_time_all_nan=0,
                min_nan_count=0, max_nan_count=0, ref_nan_count=0, n_time_nan_changed=0,
                first_problem_indices="", first_problem_times="", gap_check_method="",
                n_gap_steps=0, n_missing_steps_est=0, first_gap_indices="",
                first_gap_prev="", first_gap_next="", first_gap_detail="",
                has_time_bounds="", time_bounds_name="", n_bounds_discontinuity=0,
                first_bounds_indices="", first_bounds_prev_end="", first_bounds_next_start="",
                first_bounds_detail="", status="ERROR", note=str(e)
            )
        rows.append(row)
        print(
            f"[{i}/{len(files)}] {row.status:5s} {path.name} "
            f"nan_changed={row.n_time_nan_changed} gaps={row.n_gap_steps} bounds={row.n_bounds_discontinuity} {row.note}"
        )

    df = pd.DataFrame(asdict(r) for r in rows)
    df.to_csv(output_csv, index=False)

    print("\n=== summary ===")
    print(df["status"].value_counts(dropna=False).to_string())
    print(f"\nCSV saved to: {output_csv}")

    bad = df[df["status"].isin(["FAIL", "WARN", "ERROR"])]
    if not bad.empty:
        cols = [
            "status", "file", "variable", "freq_class",
            "n_time_any_nan", "n_time_all_nan", "n_time_nan_changed",
            "n_gap_steps", "n_missing_steps_est", "n_bounds_discontinuity",
            "first_problem_indices", "first_gap_indices", "first_bounds_indices", "note"
        ]
        print("\n=== flagged files ===")
        with pd.option_context("display.max_colwidth", 180, "display.width", 280):
            print(bad[cols].to_string(index=False))


if __name__ == "__main__":
    main()
