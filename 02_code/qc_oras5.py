#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# 고정 설정
# ============================================================
DEFAULT_ROOT = Path("/mnt/d/project/01_ENSO/01_data/01_raw/oras5")
OUTPUT_DIR = Path("/mnt/d/project/01_ENSO/03_output")

OCEAN_VARS = {"mlotst", "ohc300", "sos", "tos", "uos", "vos"}
TIME_BOUNDS_CANDIDATES = ["time_bnds", "time_bounds", "time_bound", "bounds_time"]


# ============================================================
# 유틸
# ============================================================
def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def basename_only(path_str: str) -> str:
    return Path(path_str).name


def find_time_name(ds: xr.Dataset) -> str | None:
    for cand in ["time", "time_counter", "TIME", "t"]:
        if cand in ds.coords or cand in ds.variables:
            return cand

    for name, var in ds.variables.items():
        axis = safe_str(var.attrs.get("axis", "")).upper()
        stdn = safe_str(var.attrs.get("standard_name", "")).lower()
        if axis == "T" or stdn == "time":
            return name
    return None


def get_time_bounds_name(ds: xr.Dataset) -> str | None:
    for cand in TIME_BOUNDS_CANDIDATES:
        if cand in ds.variables:
            return cand
    return None


def parse_cf_units(units: str) -> tuple[str | None, str | None]:
    m = re.match(r"^\s*([A-Za-z_]+)\s+since\s+(.+?)\s*$", units or "")
    if not m:
        return None, None
    return m.group(1).strip().lower(), m.group(2).strip()


def try_parse_origin_timestamp(origin: str) -> pd.Timestamp | None:
    if not origin:
        return None

    txt = origin.strip().replace("T", " ").replace("Z", "")
    try:
        ts = pd.Timestamp(txt)
        if ts.tz is not None:
            ts = ts.tz_convert(None)
        return ts
    except Exception:
        pass

    try:
        return pd.Timestamp(txt.split()[0])
    except Exception:
        return None


def decode_time_values(
    time_values: np.ndarray,
    units: str,
    calendar: str | None = None
) -> list[Any] | None:
    calendar = (calendar or "standard").strip()

    try:
        import cftime  # type: ignore

        decoded = cftime.num2date(
            np.asarray(time_values),
            units=units,
            calendar=calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=False,
        )
        return list(decoded)
    except Exception:
        pass

    unit, origin = parse_cf_units(units)
    if unit is None or origin is None:
        return None

    origin_ts = try_parse_origin_timestamp(origin)
    if origin_ts is None:
        return None

    unit_map = {
        "day": "D", "days": "D",
        "hour": "h", "hours": "h",
        "minute": "m", "minutes": "m",
        "second": "s", "seconds": "s",
        "millisecond": "ms", "milliseconds": "ms",
    }
    if unit not in unit_map:
        return None

    cal_ok = calendar.lower() in {
        "standard", "gregorian", "proleptic_gregorian", "julian"
    }
    if not cal_ok:
        return None

    out: list[pd.Timestamp] = []
    for v in np.asarray(time_values):
        if pd.isna(v):
            out.append(pd.NaT)
            continue
        try:
            out.append(origin_ts + pd.to_timedelta(float(v), unit=unit_map[unit]))
        except Exception:
            return None
    return out


def infer_frequency_from_decoded(decoded: list[Any]) -> str:
    if decoded is None or len(decoded) < 2:
        return "unknown"

    vals = [x for x in decoded if not pd.isna(x)]
    if len(vals) < 2:
        return "unknown"

    month_diffs = []
    day_diffs = []

    for a, b in zip(vals[:-1], vals[1:]):
        try:
            mdiff = (b.year - a.year) * 12 + (b.month - a.month)
            month_diffs.append(mdiff)
        except Exception:
            pass

        try:
            ad = pd.Timestamp(f"{a.year:04d}-{a.month:02d}-{a.day:02d}")
            bd = pd.Timestamp(f"{b.year:04d}-{b.month:02d}-{b.day:02d}")
            day_diffs.append((bd - ad).days)
        except Exception:
            pass

    if month_diffs:
        frac_monthly = sum(m == 1 for m in month_diffs) / len(month_diffs)
        if frac_monthly >= 0.8:
            return "monthly"

    if day_diffs:
        frac_daily = sum(d == 1 for d in day_diffs) / len(day_diffs)
        if frac_daily >= 0.8:
            return "daily"

    return "unknown"


def check_time_monotonic_and_duplicates(time_values: np.ndarray) -> tuple[bool, int]:
    vals = np.asarray(time_values).astype(float)
    if len(vals) < 2:
        return True, 0

    diffs = np.diff(vals)
    monotonic = bool(np.all(diffs > 0))
    duplicated = int(np.sum(diffs == 0))
    return monotonic, duplicated


def check_gap_daily(decoded: list[Any]) -> tuple[int, int, str, str]:
    if decoded is None or len(decoded) < 2:
        return -1, -1, "", ""

    n_gap_steps = 0
    n_missing = 0
    first_prev = ""
    first_next = ""

    for a, b in zip(decoded[:-1], decoded[1:]):
        try:
            ad = pd.Timestamp(f"{a.year:04d}-{a.month:02d}-{a.day:02d}")
            bd = pd.Timestamp(f"{b.year:04d}-{b.month:02d}-{b.day:02d}")
            dd = (bd - ad).days
        except Exception:
            return -1, -1, "", ""

        if dd != 1:
            n_gap_steps += 1
            if dd > 1:
                n_missing += dd - 1
            if not first_prev:
                first_prev = f"{a.year:04d}-{a.month:02d}-{a.day:02d}"
                first_next = f"{b.year:04d}-{b.month:02d}-{b.day:02d}"

    return n_gap_steps, n_missing, first_prev, first_next


def check_gap_monthly(decoded: list[Any]) -> tuple[int, int, str, str]:
    if decoded is None or len(decoded) < 2:
        return -1, -1, "", ""

    n_gap_steps = 0
    n_missing = 0
    first_prev = ""
    first_next = ""

    for a, b in zip(decoded[:-1], decoded[1:]):
        try:
            mdiff = (b.year - a.year) * 12 + (b.month - a.month)
        except Exception:
            return -1, -1, "", ""

        if mdiff != 1:
            n_gap_steps += 1
            if mdiff > 1:
                n_missing += mdiff - 1
            if not first_prev:
                first_prev = f"{a.year:04d}-{a.month:02d}"
                first_next = f"{b.year:04d}-{b.month:02d}"

    return n_gap_steps, n_missing, first_prev, first_next


def check_time_bounds_continuity(ds: xr.Dataset, time_name: str | None) -> tuple[int, str]:
    bname = get_time_bounds_name(ds)
    if bname is None:
        return 0, ""

    try:
        b = ds[bname]
        if time_name is None or time_name not in b.dims:
            return 0, ""

        if b.ndim != 2:
            return 0, ""

        arr = np.asarray(b.values)
        if arr.shape[0] < 2 or arr.shape[1] < 2:
            return 0, ""

        left = arr[:-1, 1]
        right = arr[1:, 0]
        bad = np.where(left != right)[0]

        if len(bad) == 0:
            return 0, ""

        i0 = int(bad[0])
        first_msg = f"{bname}[{i0},1]={left[i0]} != {bname}[{i0+1},0]={right[i0]}"
        return int(len(bad)), first_msg

    except Exception as e:
        return -1, f"bounds_check_error: {e}"


def compute_nan_stats(da: xr.DataArray, time_name: str) -> dict[str, Any]:
    other_dims = [d for d in da.dims if d != time_name]

    if time_name not in da.dims:
        return {
            "n_time": 0,
            "n_total_per_time": 0,
            "n_time_any_nan": -1,
            "n_time_all_nan": -1,
            "min_nan_count": -1,
            "max_nan_count": -1,
            "n_time_nan_changed": -1,
            "first_problem_times": "",
        }

    if other_dims:
        valid_count = da.count(dim=other_dims).values
        n_total = int(np.prod([da.sizes[d] for d in other_dims], dtype=np.int64))
    else:
        valid_count = da.notnull().astype(np.int64).values
        n_total = 1

    valid_count = np.asarray(valid_count).astype(np.int64)
    nan_count = n_total - valid_count

    n_time_any_nan = int(np.sum(nan_count > 0))
    n_time_all_nan = int(np.sum(valid_count == 0))
    min_nan_count = int(np.min(nan_count)) if nan_count.size else 0
    max_nan_count = int(np.max(nan_count)) if nan_count.size else 0
    n_time_nan_changed = int(np.sum(nan_count != nan_count[0])) if nan_count.size else 0

    bad_idx = np.where((nan_count > 0) | (valid_count == 0))[0]
    first_problem_times = ",".join(str(int(i)) for i in bad_idx[:10]) if len(bad_idx) > 0 else ""

    return {
        "n_time": int(da.sizes[time_name]),
        "n_total_per_time": n_total,
        "n_time_any_nan": n_time_any_nan,
        "n_time_all_nan": n_time_all_nan,
        "min_nan_count": min_nan_count,
        "max_nan_count": max_nan_count,
        "n_time_nan_changed": n_time_nan_changed,
        "first_problem_times": first_problem_times,
    }


def find_data_var(ds: xr.Dataset, expected_var: str, time_name: str | None) -> str | None:
    if expected_var in ds.data_vars:
        return expected_var

    candidates = []
    for name, da in ds.data_vars.items():
        lname = name.lower()
        if "bnds" in lname or "bounds" in lname:
            continue
        if time_name is not None and time_name not in da.dims:
            continue
        candidates.append((name, da.ndim, int(np.prod([da.sizes[d] for d in da.dims], dtype=np.int64))))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return candidates[0][0]


def find_target_file(root: Path, expected_var: str) -> Path | None:
    matches = sorted([p for p in root.glob(f"{expected_var}*.nc") if p.is_file()])
    if not matches:
        return None

    exact = [p for p in matches if p.name.startswith(f"{expected_var}.")]
    return exact[0] if exact else matches[0]


def analyze_var(root: Path, expected_var: str) -> dict[str, Any]:
    ncfile = find_target_file(root, expected_var)

    row: dict[str, Any] = {
        "file": "" if ncfile is None else str(ncfile),
        "basename": "" if ncfile is None else ncfile.name,
        "variable": expected_var,
        "freq_class": "",
        "status": "",
        "note": "",

        "time_name": "",
        "time_units": "",
        "time_calendar": "",
        "n_time": -1,
        "monotonic_time": "",
        "duplicated_time": -1,

        "n_gap_steps": -1,
        "n_missing_steps_est": -1,
        "first_gap_prev": "",
        "first_gap_next": "",

        "n_bounds_discontinuity": -1,
        "bounds_note": "",

        "n_total_per_time": -1,
        "n_time_any_nan": -1,
        "n_time_all_nan": -1,
        "min_nan_count": -1,
        "max_nan_count": -1,
        "n_time_nan_changed": -1,
        "first_problem_times": "",
    }

    if ncfile is None:
        row["status"] = "ERROR"
        row["note"] = f"target file not found for variable: {expected_var}"
        return row

    try:
        ds = xr.open_dataset(ncfile, decode_times=False)

        time_name = find_time_name(ds)
        row["time_name"] = safe_str(time_name)

        if time_name is None:
            row["status"] = "ERROR"
            row["note"] = "time coordinate not found"
            ds.close()
            return row

        actual_var = find_data_var(ds, expected_var, time_name)
        if actual_var is None:
            row["status"] = "ERROR"
            row["note"] = f"data variable not found (expected={expected_var})"
            ds.close()
            return row

        row["variable"] = actual_var
        da = ds[actual_var]
        time_var = ds[time_name]

        raw_time = np.asarray(time_var.values)
        row["n_time"] = int(raw_time.size)

        units = safe_str(time_var.attrs.get("units", ""))
        calendar = safe_str(time_var.attrs.get("calendar", "standard"))
        row["time_units"] = units
        row["time_calendar"] = calendar

        monotonic, duplicated = check_time_monotonic_and_duplicates(raw_time)
        row["monotonic_time"] = "TRUE" if monotonic else "FALSE"
        row["duplicated_time"] = duplicated

        decoded = decode_time_values(raw_time, units=units, calendar=calendar)
        freq_class = infer_frequency_from_decoded(decoded)
        row["freq_class"] = freq_class

        if freq_class == "daily":
            n_gap_steps, n_missing, first_prev, first_next = check_gap_daily(decoded)
        elif freq_class == "monthly":
            n_gap_steps, n_missing, first_prev, first_next = check_gap_monthly(decoded)
        else:
            n_gap_steps, n_missing, first_prev, first_next = (-1, -1, "", "")

        row["n_gap_steps"] = n_gap_steps
        row["n_missing_steps_est"] = n_missing
        row["first_gap_prev"] = first_prev
        row["first_gap_next"] = first_next

        n_bounds_disc, bounds_note = check_time_bounds_continuity(ds, time_name)
        row["n_bounds_discontinuity"] = n_bounds_disc
        row["bounds_note"] = bounds_note

        nan_stats = compute_nan_stats(da, time_name)
        row.update(nan_stats)

        fail_reasons = []
        warn_reasons = []

        if not monotonic:
            fail_reasons.append("time_not_strictly_increasing")
        if duplicated > 0:
            fail_reasons.append(f"duplicated_time={duplicated}")
        if n_gap_steps > 0:
            fail_reasons.append(f"time_gap_or_overlap={n_gap_steps}")
        if n_bounds_disc > 0:
            fail_reasons.append(f"time_bounds_discontinuity={n_bounds_disc}")
        if row["n_time_all_nan"] > 0:
            fail_reasons.append(f"all_nan_timesteps={row['n_time_all_nan']}")

        if row["n_time_nan_changed"] > 0:
            warn_reasons.append(f"nan_mask_changes_over_time={row['n_time_nan_changed']}")

        if n_gap_steps == -1:
            warn_reasons.append("gap_check_unavailable")
        if n_bounds_disc == -1:
            warn_reasons.append("bounds_check_unavailable")
        if expected_var != actual_var:
            warn_reasons.append(f"filename_var={expected_var}, dataset_var={actual_var}")

        if fail_reasons:
            row["status"] = "FAIL"
        elif warn_reasons:
            row["status"] = "WARN"
        else:
            row["status"] = "PASS"

        row["note"] = "; ".join(fail_reasons + warn_reasons)

        ds.close()
        return row

    except Exception as e:
        row["status"] = "ERROR"
        row["note"] = f"{type(e).__name__}: {e}"
        return row


def main():
    parser = argparse.ArgumentParser(
        description="QC ORAS5 merged monthly nc files for mlotst/ohc300/sos/tos/uos/vos"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=str(DEFAULT_ROOT),
        help=f"directory containing ORAS5 merged nc files (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--csv",
        default="oras5_qc_summary.csv",
        help="output CSV file name only (saved under /mnt/d/project/01_ENSO/03_output)",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_name = basename_only(args.csv)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / out_name

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"[ERROR] directory not found: {root}")

    rows = []
    for var in sorted(OCEAN_VARS):
        print(f"[QC] {var}")
        rows.append(analyze_var(root, var))

    df = pd.DataFrame(rows)

    status_order = {"FAIL": 0, "WARN": 1, "ERROR": 2, "PASS": 3}
    if "status" in df.columns:
        df["_status_order"] = df["status"].map(status_order).fillna(99)
        df = df.sort_values(["_status_order", "basename"]).drop(columns=["_status_order"])

    df.to_csv(out_csv, index=False)

    print("\n============================================================")
    print(f"Input dir : {root}")
    print(f"Output CSV: {out_csv}")
    print("------------------------------------------------------------")
    if "status" in df.columns:
        print(df["status"].value_counts(dropna=False).to_string())
    print("============================================================\n")

    bad = df[df["status"].isin(["FAIL", "WARN", "ERROR"])].copy()
    if len(bad) > 0:
        show_cols = [
            "status", "basename", "variable", "freq_class",
            "n_gap_steps", "n_missing_steps_est",
            "n_time_any_nan", "n_time_all_nan", "n_time_nan_changed",
            "first_gap_prev", "first_gap_next", "note",
        ]
        show_cols = [c for c in show_cols if c in bad.columns]
        print("[Issues]")
        print(bad[show_cols].to_string(index=False))
    else:
        print("[Issues] none")


if __name__ == "__main__":
    main()